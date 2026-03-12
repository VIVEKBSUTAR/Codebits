"""
Unknown Cause Discovery Engine.

Uses anomaly detection and residual analysis to identify hidden/unknown
causal factors not present in the predefined Bayesian network.

Techniques:
1. Residual analysis: if BN predictions diverge from observed data, something unmeasured is at play
2. Anomaly detection: unusual patterns in time-series that can't be explained by known variables
3. Cross-variable correlation spikes: sudden correlations between normally independent variables
"""
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from database import db
from utils.logger import SystemLogger
from causal_engine.attribute_space import ATTRIBUTE_NAMES, KNOWN_CAUSES

logger = SystemLogger(module_name="unknown_cause")


def _detect_residual_anomalies(timeseries: List[Dict], variable: str,
                                known_predictors: List[str]) -> Dict:
    """
    Fit a linear model from known_predictors -> variable,
    check if residuals are larger than expected (unexplained variance).
    """
    if len(timeseries) < 10:
        return {"unexplained_ratio": 0, "anomaly": False}

    y = np.array([float(row.get(variable, 0)) for row in timeseries])
    if np.std(y) < 1e-8:
        return {"unexplained_ratio": 0, "anomaly": False}

    X_cols = []
    valid_preds = []
    for p in known_predictors:
        col = np.array([float(row.get(p, 0)) for row in timeseries])
        if np.std(col) > 1e-8:
            X_cols.append(col)
            valid_preds.append(p)

    if not X_cols:
        return {"unexplained_ratio": 1.0, "anomaly": True, "reason": "no_valid_predictors"}

    X = np.column_stack([np.ones(len(y))] + X_cols)
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        predicted = X @ beta
        residuals = y - predicted
        total_var = np.var(y)
        explained_var = np.var(predicted)
        unexplained_ratio = 1 - (explained_var / max(total_var, 1e-10))
        unexplained_ratio = max(0, min(1, unexplained_ratio))

        # Check for systematic residual patterns
        # Rolling mean of residuals - if it drifts, there's a hidden trend
        window = min(10, len(residuals) // 3)
        if window > 2:
            rolling_mean = np.convolve(residuals, np.ones(window) / window, mode='valid')
            drift = np.max(np.abs(rolling_mean)) / max(np.std(y), 1e-10)
        else:
            drift = 0

        is_anomaly = unexplained_ratio > 0.5 or drift > 0.3

        return {
            "unexplained_ratio": round(float(unexplained_ratio), 3),
            "residual_drift": round(float(drift), 3),
            "anomaly": is_anomaly,
            "predictors_used": valid_preds,
        }
    except Exception:
        return {"unexplained_ratio": 0, "anomaly": False}


def _detect_correlation_spikes(timeseries: List[Dict], variables: List[str],
                                window: int = 20) -> List[Dict]:
    """
    Detect sudden correlation changes between variable pairs.
    A spike suggests an external cause affecting both.
    """
    if len(timeseries) < window * 2:
        return []

    spikes = []
    data = {}
    for v in variables:
        col = [float(row.get(v, 0)) for row in timeseries]
        data[v] = np.array(col)

    n = len(timeseries)
    for i, v1 in enumerate(variables):
        for j, v2 in enumerate(variables):
            if j <= i:
                continue
            if np.std(data[v1]) < 1e-8 or np.std(data[v2]) < 1e-8:
                continue

            # Compute rolling correlation
            correlations = []
            for start in range(0, n - window, window // 2):
                end = start + window
                seg1 = data[v1][start:end]
                seg2 = data[v2][start:end]
                if np.std(seg1) > 1e-8 and np.std(seg2) > 1e-8:
                    r = np.corrcoef(seg1, seg2)[0, 1]
                    if not np.isnan(r):
                        correlations.append(r)

            if len(correlations) < 3:
                continue

            # Detect spike: is the last correlation significantly different from baseline?
            baseline = np.mean(correlations[:-1])
            latest = correlations[-1]
            change = abs(latest - baseline)

            if change > 0.4 and abs(latest) > 0.5:
                spikes.append({
                    "var1": v1,
                    "var2": v2,
                    "baseline_corr": round(float(baseline), 3),
                    "current_corr": round(float(latest), 3),
                    "change": round(float(change), 3),
                })

    return spikes


def _detect_sudden_shifts(timeseries: List[Dict], variable: str) -> Optional[Dict]:
    """Detect sudden level shifts (change points) in a variable."""
    values = np.array([float(row.get(variable, 0)) for row in timeseries])
    if len(values) < 20 or np.std(values) < 1e-8:
        return None

    # Simple CUSUM-based change detection
    mean_val = np.mean(values)
    std_val = np.std(values)
    cusum_pos = np.zeros(len(values))
    cusum_neg = np.zeros(len(values))
    threshold = 3 * std_val

    for i in range(1, len(values)):
        cusum_pos[i] = max(0, cusum_pos[i - 1] + (values[i] - mean_val) - 0.5 * std_val)
        cusum_neg[i] = max(0, cusum_neg[i - 1] - (values[i] - mean_val) - 0.5 * std_val)

    max_pos = np.max(cusum_pos)
    max_neg = np.max(cusum_neg)

    if max_pos > threshold or max_neg > threshold:
        change_idx = int(np.argmax(np.maximum(cusum_pos, cusum_neg)))
        direction = "increase" if cusum_pos[change_idx] > cusum_neg[change_idx] else "decrease"
        return {
            "variable": variable,
            "change_point_index": change_idx,
            "direction": direction,
            "magnitude": round(float(max(max_pos, max_neg) / std_val), 2),
            "timestamp": timeseries[change_idx].get("timestamp", ""),
        }
    return None


def discover_unknown_causes(zone: str, timeseries: Optional[List[Dict]] = None) -> Dict:
    """
    Run the full unknown cause discovery pipeline.

    Returns:
        {
            "zone": str,
            "discoveries": [...],
            "residual_analysis": {...},
            "correlation_spikes": [...],
            "change_points": [...],
            "timestamp": str,
        }
    """
    if timeseries is None:
        timeseries = db.get_timeseries(zone, hours=168)

    if len(timeseries) < 10:
        return {
            "zone": zone,
            "discoveries": [],
            "residual_analysis": {},
            "correlation_spikes": [],
            "change_points": [],
            "message": "Insufficient data — need at least 10 time-series points",
            "timestamp": datetime.now().isoformat(),
        }

    variables = ATTRIBUTE_NAMES

    # 1. Residual analysis for each target variable
    residual_results = {}
    discoveries = []
    for target, predictors in KNOWN_CAUSES.items():
        result = _detect_residual_anomalies(timeseries, target, predictors)
        residual_results[target] = result
        if result.get("anomaly"):
            cause_desc = f"Unexplained variance in {target} — {result['unexplained_ratio']:.0%} not explained by known causes ({', '.join(predictors)})"
            discoveries.append({
                "event_type": target,
                "discovered_cause": cause_desc,
                "confidence": round(result["unexplained_ratio"], 2),
                "evidence_type": "residual_analysis",
            })
            # Store to DB
            try:
                db.store_unknown_cause({
                    "zone": zone,
                    "event_type": target,
                    "discovered_cause": cause_desc,
                    "confidence": result["unexplained_ratio"],
                    "evidence": result,
                    "algorithm": "residual_analysis",
                })
            except Exception:
                pass

    # 2. Correlation spikes
    corr_spikes = _detect_correlation_spikes(timeseries, variables)
    for spike in corr_spikes:
        cause_desc = f"Sudden correlation change between {spike['var1']} and {spike['var2']} (Δ={spike['change']:.2f}) suggests hidden common cause"
        discoveries.append({
            "event_type": f"{spike['var1']}/{spike['var2']}",
            "discovered_cause": cause_desc,
            "confidence": round(min(spike["change"], 1.0), 2),
            "evidence_type": "correlation_spike",
        })

    # 3. Change point detection  
    change_points = []
    for v in variables:
        cp = _detect_sudden_shifts(timeseries, v)
        if cp:
            change_points.append(cp)
            cause_desc = f"Sudden {cp['direction']} in {v} at data point {cp['change_point_index']} (magnitude: {cp['magnitude']}σ) — likely external cause"
            discoveries.append({
                "event_type": v,
                "discovered_cause": cause_desc,
                "confidence": round(min(cp["magnitude"] / 5.0, 1.0), 2),
                "evidence_type": "change_point",
            })

    logger.log(f"Unknown cause discovery for {zone}: {len(discoveries)} findings")

    return {
        "zone": zone,
        "discoveries": discoveries,
        "residual_analysis": residual_results,
        "correlation_spikes": corr_spikes,
        "change_points": change_points,
        "total_findings": len(discoveries),
        "timestamp": datetime.now().isoformat(),
    }
