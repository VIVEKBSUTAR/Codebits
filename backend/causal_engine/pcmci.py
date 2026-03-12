"""
PCMCI (Peter-Clark Momentary Conditional Independence) for time-series causal discovery.

A constraint-based method that:
1. PC-stable algorithm to find skeleton (remove spurious edges via conditional independence)
2. MCI (Momentary Conditional Independence) to orient edges with lag structure

Reference: Runge et al. (2019) "Detecting and quantifying causal associations in large nonlinear time series datasets"
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from utils.logger import SystemLogger
from causal_engine.attribute_space import ATTRIBUTE_NAMES, prepare_matrix

logger = SystemLogger(module_name="pcmci")


def _prepare_matrix(timeseries: List[Dict], variables: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Delegate to common attribute space."""
    return prepare_matrix(timeseries, variables)


def _partial_correlation(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Tuple[float, float]:
    """
    Compute partial correlation between x and y given z.
    Uses linear regression residualization.
    Returns (partial_corr, p_value).
    """
    n = len(x)
    if z.size == 0 or z.ndim == 1 and len(z) == 0:
        # Simple correlation
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0, 1.0
        r = np.corrcoef(x, y)[0, 1]
        if np.isnan(r):
            return 0.0, 1.0
        # Fisher z-transform for p-value
        return float(r), _corr_p_value(r, n)

    if z.ndim == 1:
        z = z.reshape(-1, 1)

    # Residualize x and y on z
    Z = np.column_stack([np.ones(n), z])
    if Z.shape[0] <= Z.shape[1]:
        return 0.0, 1.0
    try:
        beta_x = np.linalg.lstsq(Z, x, rcond=None)[0]
        beta_y = np.linalg.lstsq(Z, y, rcond=None)[0]
        res_x = x - Z @ beta_x
        res_y = y - Z @ beta_y
    except np.linalg.LinAlgError:
        return 0.0, 1.0

    if np.std(res_x) < 1e-10 or np.std(res_y) < 1e-10:
        return 0.0, 1.0

    r = np.corrcoef(res_x, res_y)[0, 1]
    if np.isnan(r):
        return 0.0, 1.0
    effective_n = n - z.shape[1]
    return float(r), _corr_p_value(r, effective_n)


def _corr_p_value(r: float, n: int) -> float:
    """P-value for correlation using t-distribution approximation."""
    if n <= 3:
        return 1.0
    t_stat = r * np.sqrt((n - 2) / max(1 - r ** 2, 1e-10))
    # Approximate using normal for large n
    try:
        from scipy.stats import t as t_dist
        return float(2 * t_dist.sf(abs(t_stat), n - 2))
    except ImportError:
        # Rough approximation
        abs_t = abs(t_stat)
        if abs_t > 3.5:
            return 0.001
        elif abs_t > 2.5:
            return 0.01
        elif abs_t > 2.0:
            return 0.05
        elif abs_t > 1.5:
            return 0.15
        return 0.4


def _build_lagged_data(data: np.ndarray, tau_max: int) -> Tuple[np.ndarray, int]:
    """
    Build lagged data matrix.
    Each variable at each lag becomes a separate column.
    Returns (lagged_data, effective_T).
    """
    T, V = data.shape
    effective_T = T - tau_max
    if effective_T < 5:
        return np.array([]), 0

    # Columns: var0_lag0, var0_lag1, ..., var0_lagP, var1_lag0, ...
    cols = []
    for v in range(V):
        for lag in range(tau_max + 1):
            start = tau_max - lag
            end = T - lag
            cols.append(data[start:end, v])
    return np.column_stack(cols), effective_T


def pc_stable_skeleton(data: np.ndarray, var_names: List[str], tau_max: int = 2,
                       alpha: float = 0.05) -> Dict[Tuple, Dict]:
    """
    PC-stable algorithm for skeleton discovery.
    Tests conditional independence at increasing conditioning set sizes.
    """
    T, V = data.shape
    effective_T = T - tau_max
    if effective_T < 10:
        return {}

    # Initialize: all lagged pairs are potential edges
    # Links: (i, -tau) -> j  means var_i at lag tau causes var_j at lag 0
    links = {}
    for j in range(V):
        for i in range(V):
            for tau in range(0, tau_max + 1):
                if i == j and tau == 0:
                    continue
                links[(i, -tau, j)] = {"val": 1.0, "pval": 0.0, "removed": False}

    # Iteratively test conditional independence with increasing set size
    for cond_size in range(0, min(V * tau_max, 4)):
        for (i, neg_tau, j), info in list(links.items()):
            if info["removed"]:
                continue
            tau = -neg_tau
            # Get potential conditioning parents of j (excluding the tested link)
            parents_j = [
                (pi, pt, pj)
                for (pi, pt, pj), pinfo in links.items()
                if pj == j and not pinfo["removed"] and (pi, pt) != (i, neg_tau)
            ]
            if len(parents_j) < cond_size:
                continue

            # Test all subsets of size cond_size
            removed = False
            for subset in combinations(parents_j, cond_size):
                # Build conditioning set data
                y = data[tau_max:, j]
                x = data[tau_max - tau: T - tau if tau > 0 else T, i]

                z_cols = []
                for (pi, pt, _) in subset:
                    p_tau = -pt
                    z_cols.append(data[tau_max - p_tau: T - p_tau if p_tau > 0 else T, pi])

                z = np.column_stack(z_cols) if z_cols else np.array([])
                r, p_val = _partial_correlation(x, y, z)
                links[(i, neg_tau, j)]["val"] = abs(r)
                links[(i, neg_tau, j)]["pval"] = p_val

                if p_val > alpha:
                    links[(i, neg_tau, j)]["removed"] = True
                    removed = True
                    break

            if removed:
                continue

    return links


def mci_test(data: np.ndarray, links: Dict, var_names: List[str],
             tau_max: int = 2, alpha: float = 0.05) -> List[Dict]:
    """
    MCI (Momentary Conditional Independence) test.
    For each surviving link X(t-tau) -> Y(t), condition on:
    - All parents of Y(t) except X(t-tau)
    - All parents of X(t-tau)
    """
    T, V = data.shape
    results = []

    # Collect parents for each variable
    parents = {j: [] for j in range(V)}
    for (i, neg_tau, j), info in links.items():
        if not info["removed"]:
            parents[j].append((i, -neg_tau))

    for (i, neg_tau, j), info in links.items():
        if info["removed"]:
            continue
        tau = -neg_tau

        # Conditioning set: parents of Y \ {X(t-tau)} + parents of X(t-tau)
        cond_vars = []
        for (pi, p_tau) in parents[j]:
            if (pi, p_tau) != (i, tau):
                cond_vars.append((pi, p_tau))
        for (pi, p_tau) in parents[i]:
            cond_vars.append((pi, p_tau + tau))

        # Build arrays
        y = data[tau_max:, j]
        x = data[tau_max - tau: T - tau if tau > 0 else T, i]

        z_cols = []
        for (pi, p_tau) in cond_vars:
            if 0 <= tau_max - p_tau and tau_max - p_tau < T:
                end = T - p_tau if p_tau > 0 else T
                start = tau_max - p_tau
                if start >= 0 and end > start:
                    col = data[start:end, pi]
                    if len(col) == len(y):
                        z_cols.append(col)

        z = np.column_stack(z_cols) if z_cols else np.array([])
        r, p_val = _partial_correlation(x, y, z)

        if p_val < alpha:
            results.append({
                "from": var_names[i],
                "to": var_names[j],
                "lag": tau,
                "mci_value": round(abs(r), 4),
                "p_value": round(p_val, 6),
                "significant": True,
            })

    return results


def run_pcmci(timeseries: List[Dict], tau_max: int = 2, alpha: float = 0.05,
              variables: Optional[List[str]] = None) -> Dict:
    """
    Run full PCMCI algorithm on time-series data.

    Returns:
        {
            "edges": [...],
            "significant_edges": [...],
            "skeleton_links": int,
            "data_points": int,
            "variables": [...],
            "tau_max": int,
        }
    """
    data, valid_vars = _prepare_matrix(timeseries, variables)
    if data.size == 0 or len(valid_vars) < 2:
        return {"edges": [], "significant_edges": [], "skeleton_links": 0, "data_points": 0}

    T, V = data.shape
    logger.log(f"PCMCI: {T} time points, {V} variables, tau_max={tau_max}")

    # Step 1: PC-stable skeleton
    links = pc_stable_skeleton(data, valid_vars, tau_max=tau_max, alpha=alpha)
    skeleton_count = sum(1 for v in links.values() if not v["removed"])

    # Step 2: MCI test
    edges = mci_test(data, links, valid_vars, tau_max=tau_max, alpha=alpha)

    # Sort by strength
    edges.sort(key=lambda e: e["mci_value"], reverse=True)

    logger.log(f"PCMCI: skeleton={skeleton_count}, final edges={len(edges)}")
    return {
        "edges": edges,
        "significant_edges": edges,
        "skeleton_links": skeleton_count,
        "data_points": T,
        "variables": valid_vars,
        "tau_max": tau_max,
        "alpha": alpha,
    }
