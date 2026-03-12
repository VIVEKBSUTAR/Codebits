"""
Granger Causality Test for time-series causal discovery.

Tests whether past values of one variable help predict another,
using VAR (Vector Auto-Regression) framework.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.logger import SystemLogger
from causal_engine.attribute_space import ATTRIBUTE_NAMES, prepare_matrix

logger = SystemLogger(module_name="granger")


def _prepare_matrix(timeseries: List[Dict], variables: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Delegate to common attribute space."""
    return prepare_matrix(timeseries, variables)


def _ols_residuals(y: np.ndarray, X: np.ndarray) -> Tuple[np.ndarray, float]:
    """OLS regression, returns residuals and RSS."""
    if X.shape[0] <= X.shape[1]:
        return y, float(np.sum(y ** 2))
    try:
        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        residuals = y - X @ beta
        rss = float(np.sum(residuals ** 2))
        return residuals, rss
    except np.linalg.LinAlgError:
        return y, float(np.sum(y ** 2))


def granger_test_pair(x: np.ndarray, y: np.ndarray, max_lag: int = 3) -> Dict:
    """
    Test if x Granger-causes y.

    Restricted model:  y_t = a0 + a1*y_{t-1} + ... + ap*y_{t-p}
    Unrestricted model: y_t = a0 + a1*y_{t-1} + ... + ap*y_{t-p} + b1*x_{t-1} + ... + bp*x_{t-p}

    F-test: ((RSS_r - RSS_u) / p) / (RSS_u / (T - 2p - 1))
    """
    T = len(y)
    if T < max_lag * 3 + 5:
        return {"f_stat": 0, "p_value": 1.0, "significant": False, "lag": max_lag}

    best_result = {"f_stat": 0, "p_value": 1.0, "significant": False, "lag": 1}

    for lag in range(1, max_lag + 1):
        n = T - lag
        if n < 2 * lag + 3:
            continue

        # Build lagged arrays
        y_target = y[lag:]
        y_lags = np.column_stack([y[lag - i - 1: T - i - 1] for i in range(lag)])
        x_lags = np.column_stack([x[lag - i - 1: T - i - 1] for i in range(lag)])

        # Restricted: only y lags
        X_r = np.column_stack([np.ones(n), y_lags])
        _, rss_r = _ols_residuals(y_target, X_r)

        # Unrestricted: y lags + x lags
        X_u = np.column_stack([np.ones(n), y_lags, x_lags])
        _, rss_u = _ols_residuals(y_target, X_u)

        df1 = lag
        df2 = n - 2 * lag - 1
        if df2 <= 0 or rss_u <= 0:
            continue

        f_stat = ((rss_r - rss_u) / df1) / (rss_u / df2)
        f_stat = max(0, f_stat)

        # Approximate p-value using F-distribution CDF
        p_value = _f_survival(f_stat, df1, df2)

        if p_value < best_result["p_value"]:
            best_result = {
                "f_stat": round(f_stat, 4),
                "p_value": round(p_value, 6),
                "significant": p_value < 0.05,
                "lag": lag,
            }

    return best_result


def _f_survival(f: float, df1: int, df2: int) -> float:
    """Approximate F-distribution survival function using Beta incomplete function."""
    try:
        from scipy.stats import f as f_dist
        return float(f_dist.sf(f, df1, df2))
    except ImportError:
        # Rough approximation when scipy not available
        # Use Wilson-Hilferty normal approximation
        if f <= 0:
            return 1.0
        a = df1 / 2
        b = df2 / 2
        x = df2 / (df2 + df1 * f)
        # Very rough: use threshold-based
        if f > 4:
            return 0.01
        elif f > 2.5:
            return 0.05
        elif f > 1.5:
            return 0.15
        return 0.3


def run_granger_full(timeseries: List[Dict], max_lag: int = 3,
                     variables: Optional[List[str]] = None) -> Dict:
    """
    Run pairwise Granger causality tests on all variable pairs.

    Returns:
        {
            "edges": [{"from": "x", "to": "y", "lag": 2, "f_stat": ..., "p_value": ...}, ...],
            "matrix": { "x->y": { "f_stat": ..., "p_value": ..., ... }, ... },
            "significant_edges": [...],
            "data_points": T,
        }
    """
    data, valid_vars = _prepare_matrix(timeseries, variables)
    if data.size == 0 or len(valid_vars) < 2:
        return {"edges": [], "matrix": {}, "significant_edges": [], "data_points": 0}

    T, V = data.shape
    edges = []
    matrix = {}

    for i in range(V):
        for j in range(V):
            if i == j:
                continue
            x_name = valid_vars[i]
            y_name = valid_vars[j]
            result = granger_test_pair(data[:, i], data[:, j], max_lag=max_lag)
            key = f"{x_name}->{y_name}"
            matrix[key] = result
            if result["significant"]:
                edges.append({
                    "from": x_name,
                    "to": y_name,
                    "lag": result["lag"],
                    "f_stat": result["f_stat"],
                    "p_value": result["p_value"],
                })

    logger.log(f"Granger: {len(edges)} significant edges from {T} data points, {V} variables")
    return {
        "edges": edges,
        "matrix": matrix,
        "significant_edges": edges,
        "data_points": T,
        "variables": valid_vars,
        "max_lag": max_lag,
    }
