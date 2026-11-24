"""Semidefinite relaxation of the sparse generalised eigenvalue problem.

Implements Section 3.2 of the paper. The lifted problem

    maximise   Tr(A X) / Tr(B X)
    s.t.       Card(X) <= k^2, Tr(X) = 1, X >= 0, Rank(X) = 1

is relaxed to the convex SDP

    maximise   Tr(A Y)
    s.t.       1^T |Y| 1 - k z <= 0
               Tr(Y) - z = 0
               Tr(B Y) = 1
               Y >= 0,

in the variables ``Y`` (n x n PSD) and ``z >= 0``. After solving, a primal
sparse portfolio ``x`` is recovered from the leading eigenvector of ``Y``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


@dataclass(frozen=True)
class SDPSolution:
    """Output of :func:`sdp_sparse`.

    weights:   length-``n`` vector, unit norm, with at most ``k`` entries
               above the sparsity tolerance after recovery.
    objective: generalised Rayleigh quotient of ``weights``.
    sdp_bound: optimal value of the relaxed SDP (an upper bound on the true
               maximum Rayleigh quotient for the given cardinality).
    status:    solver status string.
    """

    weights: np.ndarray
    objective: float
    sdp_bound: float
    status: str


def _rayleigh(x: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    den = float(x @ B @ x)
    if den <= 0:
        return 0.0
    return float(x @ A @ x) / den


def sdp_sparse(
    A: np.ndarray,
    B: np.ndarray,
    k: int,
    *,
    maximise: bool = True,
    solver: str = "SCS",
    sparsity_tol: float = 1e-3,
) -> SDPSolution:
    """Solve the SDP relaxation (paper Section 3.2) for a single cardinality.

    When ``maximise=True`` the SDP targets the largest generalised
    eigenvalue; set ``maximise=False`` (equivalent to swapping ``A`` and ``B``)
    for the smallest. For mean reversion the paper swaps the roles of ``A``
    and ``B`` (minimise predictability).
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or B.shape != (n, n):
        raise ValueError("A and B must be square matrices of the same size.")
    if not (1 <= k <= n):
        raise ValueError("k must lie in [1, n].")

    Y = cp.Variable((n, n), symmetric=True)
    z = cp.Variable(nonneg=True)
    constraints = [
        Y >> 0,
        cp.sum(cp.abs(Y)) - k * z <= 0,
        cp.trace(Y) - z == 0,
        cp.trace(B @ Y) == 1,
    ]
    obj = cp.Maximize(cp.trace(A @ Y)) if maximise else cp.Minimize(cp.trace(A @ Y))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=solver, verbose=False)

    if Y.value is None:
        raise RuntimeError(f"SDP failed to solve (status {prob.status}).")

    Y_val = np.asarray(Y.value)
    Y_val = 0.5 * (Y_val + Y_val.T)
    eigvals, eigvecs = np.linalg.eigh(Y_val)
    idx = int(np.argmax(np.abs(eigvals)))
    x = eigvecs[:, idx]
    # Truncate to the k largest entries (sparsity recovery, paper page 8).
    order = np.argsort(np.abs(x))[::-1]
    keep = order[:k]
    x_sparse = np.zeros_like(x)
    x_sparse[keep] = x[keep]
    # Drop any very small spurious entries below ``sparsity_tol`` relative scale.
    max_abs = float(np.max(np.abs(x_sparse))) if x_sparse.size else 0.0
    if max_abs > 0:
        x_sparse[np.abs(x_sparse) < sparsity_tol * max_abs] = 0.0
    nrm = float(np.linalg.norm(x_sparse))
    if nrm > 0:
        x_sparse /= nrm
    return SDPSolution(
        weights=x_sparse,
        objective=_rayleigh(x_sparse, A, B),
        sdp_bound=float(prob.value) if prob.value is not None else float("nan"),
        status=str(prob.status),
    )


def sdp_path(
    A: np.ndarray,
    B: np.ndarray,
    *,
    maximise: bool = True,
    solver: str = "SCS",
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Full cardinality-1 to cardinality-n SDP path.

    Returns a tuple ``(weights, objective, bound)`` mirroring
    :class:`src.greedy.GreedyPath` for an easy side-by-side comparison.
    """
    n = A.shape[0]
    weights: list[np.ndarray] = []
    objective = np.empty(n)
    bound = np.empty(n)
    for k in range(1, n + 1):
        sol = sdp_sparse(A, B, k, maximise=maximise, solver=solver)
        weights.append(sol.weights)
        objective[k - 1] = sol.objective
        bound[k - 1] = sol.sdp_bound
    return weights, objective, bound
