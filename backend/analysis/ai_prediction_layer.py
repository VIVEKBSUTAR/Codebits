"""
AI Root-Cause Prediction Layer.

When Granger, PCMCI, and NOTEARS cannot identify the root cause
of an event (no significant edges found for a target variable),
this layer uses ML-based approaches to predict the most likely cause:

1. Correlation ranking: rank all attributes by correlation with the target
2. Mutual information: non-linear dependency scoring
3. Anomaly coincidence: find attributes that spiked at the same time as the event
4. Ensemble scoring: combine all signals into a confidence-weighted prediction
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from causal_engine.attribute_space import ATTRIBUTE_NAMES, KNOWN_CAUSES, prepare_matrix
from utils.logger import SystemLogger

logger = SystemLogger(module_name="ai_predictor")


def _correlation_ranking(data: np.ndarray, target_idx: int,
                         var_names: List[str]) -> List[Dict]:
    """Rank variables by absolute Pearson correlation with target."""
    T, V = data.shape
    target = data[:, target_idx]
    if np.std(target) < 1e-8:
        return []
    rankings = []
    for j in range(V):
        if j == target_idx:
            continue
        if np.std(data[:, j]) < 1e-8:
            continue
        r = np.corrcoef(target, data[:, j])[0, 1]
        if np.isnan(r):
            continue
        rankings.append({
            "variable": var_names[j],
            "correlation": round(float(r), 4),
            "abs_correlation": round(abs(float(r)), 4),
            "method": "pearson_correlation",
        })
    rankings.sort(key=lambda x: x["abs_correlation"], reverse=True)
    return rankings


def _mutual_information_ranking(data: np.ndarray, target_idx: int,
                                var_names: List[str], n_bins: int = 10) -> List[Dict]:
    """
    Estimate mutual information between each variable and the target
    using histogram-based approach (captures non-linear dependencies).
    """
    T, V = data.shape
    target = data[:, target_idx]
    if np.std(target) < 1e-8:
        return []

    rankings = []
    for j in range(V):
        if j == target_idx:
            continue
        x = data[:, j]
        if np.std(x) < 1e-8:
            continue
        try:
            # 2D histogram
            c_xy, _, _ = np.histogram2d(x, target, bins=n_bins)
            c_xy = c_xy / c_xy.sum()  # joint probability
            c_x = c_xy.sum(axis=1)    # marginal x
            c_y = c_xy.sum(axis=0)    # marginal y

            # MI = sum p(x,y) * log(p(x,y) / (p(x)*p(y)))
            mi = 0.0
            for i in range(n_bins):
                for k in range(n_bins):
                    if c_xy[i, k] > 1e-10 and c_x[i] > 1e-10 and c_y[k] > 1e-10:
                        mi += c_xy[i, k] * np.log(c_xy[i, k] / (c_x[i] * c_y[k]))

            rankings.append({
                "variable": var_names[j],
                "mutual_information": round(float(max(0, mi)), 4),
                "method": "mutual_information",
            })
        except Exception:
            continue
    rankings.sort(key=lambda x: x["mutual_information"], reverse=True)
    return rankings


def _anomaly_coincidence(data: np.ndarray, target_idx: int,
                         var_names: List[str], z_threshold: float = 2.0) -> List[Dict]:
    """
    Find variables that had anomalous spikes at the same timestamps
    as anomalous spikes in the target (coincidence analysis).
    """
    T, V = data.shape
    target = data[:, target_idx]
    if np.std(target) < 1e-8:
        return []

    # Find anomaly timestamps in target
    t_mean = np.mean(target)
    t_std = np.std(target)
    target_anomalies = set(np.where(np.abs(target - t_mean) > z_threshold * t_std)[0])
    if not target_anomalies:
        return []

    coincidences = []
    for j in range(V):
        if j == target_idx:
            continue
        x = data[:, j]
        if np.std(x) < 1e-8:
            continue
        x_mean = np.mean(x)
        x_std = np.std(x)
        x_anomalies = set(np.where(np.abs(x - x_mean) > z_threshold * x_std)[0])
        if not x_anomalies:
            continue

        # Count overlapping anomaly timestamps (with ±1 window)
        overlap = 0
        for t in target_anomalies:
            for offset in [-1, 0, 1]:
                if (t + offset) in x_anomalies:
                    overlap += 1
                    break

        if overlap > 0:
            score = overlap / max(len(target_anomalies), 1)
            coincidences.append({
                "variable": var_names[j],
                "coincidence_score": round(float(score), 4),
                "overlapping_anomalies": overlap,
                "total_target_anomalies": len(target_anomalies),
                "method": "anomaly_coincidence",
            })
    coincidences.sort(key=lambda x: x["coincidence_score"], reverse=True)
    return coincidences


def predict_root_cause(timeseries: List[Dict],
                       target_variable: str,
                       causal_edges: Optional[List[Dict]] = None) -> Dict:
    """
    AI prediction layer that identifies the most likely root cause
    when standard causal discovery algorithms fail.

    Args:
        timeseries: Time-series data (list of dicts with attribute values)
        target_variable: The variable/event to explain
        causal_edges: Edges from causal discovery (if any were found)

    Returns:
        {
            "target": str,
            "predicted_causes": [...],
            "confidence": float,
            "method_details": {...},
            "causal_discovery_found": bool,
        }
    """
    data, valid_vars = prepare_matrix(timeseries)
    if data.size == 0 or target_variable not in valid_vars:
        return {
            "target": target_variable,
            "predicted_causes": [],
            "confidence": 0,
            "method_details": {},
            "causal_discovery_found": False,
            "message": f"Insufficient data or '{target_variable}' has no variance",
        }

    target_idx = valid_vars.index(target_variable)

    # Check if causal discovery already found edges
    causal_found = False
    if causal_edges:
        incoming = [e for e in causal_edges if e.get("to") == target_variable]
        if incoming:
            causal_found = True

    # Run all three prediction methods
    corr_ranking = _correlation_ranking(data, target_idx, valid_vars)
    mi_ranking = _mutual_information_ranking(data, target_idx, valid_vars)
    anomaly_ranking = _anomaly_coincidence(data, target_idx, valid_vars)

    # Ensemble: merge rankings with weights
    # correlation=0.3, MI=0.4, anomaly_coincidence=0.3
    scores = {}
    for r in corr_ranking:
        v = r["variable"]
        scores[v] = scores.get(v, {"variable": v, "total_score": 0, "methods": []})
        scores[v]["total_score"] += 0.3 * r["abs_correlation"]
        scores[v]["methods"].append({"method": "correlation", "score": r["abs_correlation"], "detail": r["correlation"]})

    for r in mi_ranking:
        v = r["variable"]
        scores[v] = scores.get(v, {"variable": v, "total_score": 0, "methods": []})
        # Normalize MI to 0-1 range using max
        max_mi = mi_ranking[0]["mutual_information"] if mi_ranking else 1
        norm_mi = r["mutual_information"] / max(max_mi, 1e-10)
        scores[v]["total_score"] += 0.4 * norm_mi
        scores[v]["methods"].append({"method": "mutual_information", "score": round(norm_mi, 4), "raw_mi": r["mutual_information"]})

    for r in anomaly_ranking:
        v = r["variable"]
        scores[v] = scores.get(v, {"variable": v, "total_score": 0, "methods": []})
        scores[v]["total_score"] += 0.3 * r["coincidence_score"]
        scores[v]["methods"].append({"method": "anomaly_coincidence", "score": r["coincidence_score"], "overlaps": r["overlapping_anomalies"]})

    # Sort by ensemble score
    predicted = sorted(scores.values(), key=lambda x: x["total_score"], reverse=True)
    for p in predicted:
        p["total_score"] = round(p["total_score"], 4)

    # Top confidence
    top_confidence = predicted[0]["total_score"] if predicted else 0

    # Mark which are known vs unknown causes
    known = KNOWN_CAUSES.get(target_variable, [])
    for p in predicted:
        p["is_known_cause"] = p["variable"] in known

    logger.log(
        f"AI Prediction for '{target_variable}': "
        f"{len(predicted)} candidates, top={predicted[0]['variable'] if predicted else 'none'} "
        f"(score={top_confidence:.3f}), causal_discovery_found={causal_found}"
    )

    return {
        "target": target_variable,
        "predicted_causes": predicted[:10],  # top 10
        "confidence": round(top_confidence, 4),
        "method_details": {
            "correlation_candidates": len(corr_ranking),
            "mi_candidates": len(mi_ranking),
            "anomaly_candidates": len(anomaly_ranking),
        },
        "causal_discovery_found": causal_found,
        "data_points": data.shape[0],
        "variables_analyzed": len(valid_vars),
    }


def predict_all_targets(timeseries: List[Dict],
                        causal_result: Optional[Dict] = None) -> Dict:
    """
    Run AI prediction for ALL target variables that causal discovery
    didn't find significant edges for.

    Args:
        timeseries: Time-series data
        causal_result: Result from run_all_causal_discovery (consensus_edges)

    Returns dict with predictions for each unresolved target.
    """
    from causal_engine.attribute_space import KNOWN_CAUSES

    consensus_edges = []
    if causal_result and "consensus_edges" in causal_result:
        consensus_edges = causal_result["consensus_edges"]

    # Find targets not explained by causal discovery
    resolved_targets = set()
    for e in consensus_edges:
        if e.get("agreement", 0) >= 2:
            resolved_targets.add(e["to"])

    predictions = {}
    for target in KNOWN_CAUSES:
        edges_for_target = [e for e in consensus_edges if e.get("to") == target]
        result = predict_root_cause(timeseries, target, edges_for_target)
        result["resolved_by_discovery"] = target in resolved_targets
        predictions[target] = result

    return {
        "predictions": predictions,
        "resolved_count": len(resolved_targets),
        "unresolved_count": len(KNOWN_CAUSES) - len(resolved_targets),
        "total_targets": len(KNOWN_CAUSES),
    }
