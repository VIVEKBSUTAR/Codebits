"""
NOTEARS (Non-combinatorial Optimization via Trace Exponential and Augmented lagRangian for Structure learning)

A score-based method that learns DAG structure from observational data by solving a
continuous optimization problem with an acyclicity constraint.

Reference: Zheng et al. (2018) "DAGs with NO TEARS: Continuous Optimization for Structure Learning"

Objective:  min  F(W) = 0.5/n * ||X - XW||^2_F + lambda * ||W||_1
            s.t. h(W) = tr(e^{W ◦ W}) - d = 0   (DAG constraint)

Solved via augmented Lagrangian method.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from utils.logger import SystemLogger
from causal_engine.attribute_space import ATTRIBUTE_NAMES, prepare_matrix_standardized

logger = SystemLogger(module_name="notears")


def _prepare_matrix(timeseries: List[Dict], variables: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Delegate to common attribute space (standardized for NOTEARS)."""
    return prepare_matrix_standardized(timeseries, variables)


def _h(W: np.ndarray) -> float:
    """Acyclicity constraint: h(W) = tr(e^{W◦W}) - d.
    h(W) = 0 iff W encodes a DAG."""
    d = W.shape[0]
    M = W * W  # element-wise square
    # Matrix exponential via eigendecomposition for numerical stability
    try:
        eigenvalues = np.linalg.eigvalsh(M)
        return float(np.sum(np.exp(eigenvalues)) - d)
    except np.linalg.LinAlgError:
        # Fallback: power series up to order 10
        E = np.eye(d)
        power = np.eye(d)
        for k in range(1, 11):
            power = power @ M / k
            E += power
        return float(np.trace(E) - d)


def _h_grad(W: np.ndarray) -> np.ndarray:
    """Gradient of h(W) w.r.t. W: ∇h = 2W ◦ (e^{W◦W})^T."""
    d = W.shape[0]
    M = W * W
    try:
        # Matrix exponential
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        exp_eig = np.diag(np.exp(eigenvalues))
        E = eigenvectors @ exp_eig @ eigenvectors.T
    except np.linalg.LinAlgError:
        E = np.eye(d)
        power = np.eye(d)
        for k in range(1, 11):
            power = power @ M / k
            E += power
    return 2 * W * E.T


def _loss_and_grad(W: np.ndarray, X: np.ndarray, l1_lambda: float) -> Tuple[float, np.ndarray]:
    """
    Compute least-squares loss + L1 penalty and gradient.
    F(W) = 0.5/n * ||X - XW||^2_F
    """
    n, d = X.shape
    R = X - X @ W  # residuals
    loss = 0.5 / n * np.sum(R ** 2)
    grad = -1.0 / n * (X.T @ R)
    # L1 penalty gradient (sub-gradient)
    loss += l1_lambda * np.sum(np.abs(W))
    grad += l1_lambda * np.sign(W)
    return float(loss), grad


def notears_linear(X: np.ndarray, l1_lambda: float = 0.01,
                   max_iter: int = 100, h_tol: float = 1e-8,
                   rho_max: float = 1e+16, w_threshold: float = 0.3) -> np.ndarray:
    """
    Solve the NOTEARS optimization via augmented Lagrangian.

    Args:
        X: (n, d) data matrix
        l1_lambda: L1 sparsity penalty
        max_iter: max outer iterations
        h_tol: convergence threshold for acyclicity
        rho_max: max penalty parameter
        w_threshold: threshold to zero out small weights

    Returns:
        W: (d, d) weighted adjacency matrix (DAG)
    """
    n, d = X.shape
    W = np.zeros((d, d))
    rho = 1.0  # augmented Lagrangian penalty
    alpha = 0.0  # Lagrange multiplier
    h_prev = np.inf

    for iteration in range(max_iter):
        # Inner optimization: minimize augmented Lagrangian w.r.t. W
        # L(W, alpha, rho) = F(W) + alpha * h(W) + 0.5 * rho * h(W)^2
        for _ in range(20):  # inner gradient descent steps
            loss, grad_loss = _loss_and_grad(W, X, l1_lambda)
            h_val = _h(W)
            grad_h = _h_grad(W)

            # Full gradient of augmented Lagrangian
            grad = grad_loss + (alpha + rho * h_val) * grad_h

            # Adaptive step size with line search
            step = 0.001
            W_new = W - step * grad
            np.fill_diagonal(W_new, 0)  # no self-loops

            # Simple decrease check
            h_new = _h(W_new)
            loss_new, _ = _loss_and_grad(W_new, X, l1_lambda)
            aug_old = loss + alpha * h_val + 0.5 * rho * h_val ** 2
            aug_new = loss_new + alpha * h_new + 0.5 * rho * h_new ** 2

            if aug_new < aug_old + 1e-4:
                W = W_new
            else:
                # Smaller step
                W = W - step * 0.1 * grad
                np.fill_diagonal(W, 0)

        h_val = _h(W)

        if h_val > 0.25 * h_prev:
            rho = min(rho * 10, rho_max)
        alpha += rho * h_val
        h_prev = h_val

        if abs(h_val) < h_tol:
            logger.log(f"NOTEARS converged at iteration {iteration}, h={h_val:.2e}")
            break

    # Threshold small weights
    W[np.abs(W) < w_threshold] = 0
    np.fill_diagonal(W, 0)
    return W


def run_notears(timeseries: List[Dict], l1_lambda: float = 0.01,
                w_threshold: float = 0.3,
                variables: Optional[List[str]] = None) -> Dict:
    """
    Run NOTEARS on time-series data to discover causal DAG structure.

    Returns:
        {
            "edges": [{"from": ..., "to": ..., "weight": ...}, ...],
            "adjacency_matrix": [[...], ...],
            "variables": [...],
            "data_points": int,
        }
    """
    data, valid_vars = _prepare_matrix(timeseries, variables)
    if data.size == 0 or len(valid_vars) < 2:
        return {"edges": [], "adjacency_matrix": [], "variables": [], "data_points": 0}

    T, V = data.shape
    logger.log(f"NOTEARS: {T} data points, {V} variables, lambda={l1_lambda}")

    W = notears_linear(data, l1_lambda=l1_lambda, w_threshold=w_threshold)

    edges = []
    for i in range(V):
        for j in range(V):
            if abs(W[i, j]) > 0:
                edges.append({
                    "from": valid_vars[i],
                    "to": valid_vars[j],
                    "weight": round(float(W[i, j]), 4),
                    "abs_weight": round(abs(float(W[i, j])), 4),
                })

    # Sort by absolute weight
    edges.sort(key=lambda e: e["abs_weight"], reverse=True)

    logger.log(f"NOTEARS: discovered {len(edges)} edges")
    return {
        "edges": edges,
        "adjacency_matrix": W.tolist(),
        "variables": valid_vars,
        "data_points": T,
        "l1_lambda": l1_lambda,
        "w_threshold": w_threshold,
        "h_value": round(_h(W), 8),
    }
