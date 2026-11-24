"""Greedy search for sparse generalized eigenvalue problems (paper, Section 3.1).

Solves

    maximise   x^T A x / x^T B x
    subject to Card(x) <= k, ||x|| = 1,

for every target cardinality ``k = 1, ..., n`` by iteratively adding the asset
whose inclusion yields the largest increase (or decrease) of the generalised
Rayleigh quotient on the restricted support set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class GreedyPath:
    """Full path of sparse solutions produced by :func:`greedy_search`.

    weights:  list of length ``n``; ``weights[k-1]`` is a length-``n`` vector
              with at most ``k`` non-zero entries and unit norm.
    objective: array of shape (n,); ``objective[k-1]`` is the generalised
              Rayleigh quotient reached at cardinality ``k``.
    supports: list of length ``n``; ``supports[k-1]`` is the sorted tuple of
              column indices selected by the greedy algorithm at step ``k``.
    """

    weights: list[np.ndarray]
    objective: np.ndarray
    supports: list[tuple[int, ...]]


def _restricted_ratio(
    A: np.ndarray, B: np.ndarray, support: list[int], minimise: bool
) -> tuple[float, np.ndarray]:
    """Optimal generalised Rayleigh quotient on a given support set."""
    idx = np.asarray(support, dtype=int)
    sub_A = A[np.ix_(idx, idx)]
    sub_B = B[np.ix_(idx, idx)]
    # scipy.linalg.eigh returns ascending generalised eigenvalues of A z = lam B z.
    try:
        eigvals, eigvecs = eigh(sub_A, sub_B)
    except np.linalg.LinAlgError:
        # Fall back to a regularised solve when B is near-singular.
        eps = 1e-10 * np.trace(sub_B) / max(len(idx), 1)
        eigvals, eigvecs = eigh(sub_A, sub_B + eps * np.eye(len(idx)))
    k_pick = 0 if minimise else -1
    eigval = float(eigvals[k_pick])
    z = eigvecs[:, k_pick]
    n = A.shape[0]
    x_full = np.zeros(n)
    x_full[idx] = z
    nrm = np.linalg.norm(x_full)
    if nrm > 0:
        x_full /= nrm
    return eigval, x_full


def greedy_search(
    A: np.ndarray,
    B: np.ndarray,
    *,
    minimise: bool = True,
) -> GreedyPath:
    """Forward greedy sparse generalised eigenvalue search (paper Section 3.1).

    Parameters
    ----------
    A, B : symmetric matrices of shape (n, n).
        For mean reversion the paper uses ``A = S_hat^T S_hat`` and
        ``B = S_t^T S_t`` and selects the smallest generalised eigenvalue.
    minimise : bool
        When ``True`` the algorithm greedily reduces the predictability
        (mean reverting portfolios); when ``False`` it maximises it (momentum
        portfolios). Section 3.1 writes the max form and notes the mean
        reverting case is symmetric.

    Returns
    -------
    GreedyPath with the full cardinality-1 to cardinality-n path.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    n = A.shape[0]
    if A.shape != (n, n) or B.shape != (n, n):
        raise ValueError("A and B must be square matrices of the same size.")

    # k = 1 initial step: pick argmax/argmin of A_ii / B_ii.
    diag = np.array([A[i, i] / B[i, i] for i in range(n)])
    first = int(np.argmin(diag)) if minimise else int(np.argmax(diag))
    support = [first]
    weights_list: list[np.ndarray] = []
    objective = np.empty(n)
    supports_list: list[tuple[int, ...]] = []

    obj0, w0 = _restricted_ratio(A, B, support, minimise)
    objective[0] = obj0
    weights_list.append(w0)
    supports_list.append(tuple(sorted(support)))

    for k in range(1, n):
        best_obj = np.inf if minimise else -np.inf
        best_i = -1
        best_w = w0
        for i in range(n):
            if i in support:
                continue
            trial = support + [i]
            cand_obj, cand_w = _restricted_ratio(A, B, trial, minimise)
            if minimise:
                if cand_obj < best_obj:
                    best_obj, best_i, best_w = cand_obj, i, cand_w
            else:
                if cand_obj > best_obj:
                    best_obj, best_i, best_w = cand_obj, i, cand_w
        if best_i < 0:
            raise RuntimeError("Greedy search failed to add an asset.")
        support.append(best_i)
        objective[k] = best_obj
        weights_list.append(best_w)
        supports_list.append(tuple(sorted(support)))
    return GreedyPath(
        weights=weights_list,
        objective=objective,
        supports=supports_list,
    )
