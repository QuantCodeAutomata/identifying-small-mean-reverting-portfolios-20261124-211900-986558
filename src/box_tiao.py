"""Box & Tiao (1977) canonical decomposition for mean reverting portfolios.

Implements the full procedure of Section 2 of the paper:
   1. VAR(1) OLS estimate of A (Eq. 6).
   2. Predictability matrix M = Gamma^{-1/2} A^T Gamma A Gamma^{-1/2} (Eq. 7).
   3. Generalized eigenvalue decomposition to produce the n canonical
      portfolios ranked by predictability.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.linalg import eigh


@dataclass(frozen=True)
class BoxTiaoResult:
    """Output of a Box & Tiao canonical decomposition.

    weights:        shape (n, n). Column k holds the k-th portfolio weights,
                    normalised to ``||x||=1``. Columns are ordered from the
                    most mean reverting (smallest predictability) to the least.
    predictability: shape (n,). The generalised eigenvalue associated with
                    each column, equal to ``x^T A^T Gamma A x / x^T Gamma x``.
    A_hat:          shape (n, n). The OLS VAR(1) matrix (Eq. 6).
    gamma_hat:      shape (n, n). The symmetric matrix ``S_t^T S_t / m``
                    used as the plug-in estimate of ``Gamma``.
    """

    weights: np.ndarray
    predictability: np.ndarray
    A_hat: np.ndarray
    gamma_hat: np.ndarray


def estimate_var1(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OLS estimate of the VAR(1) matrix ``A`` and the predicted series ``S_hat``.

    Parameters
    ----------
    S : ndarray of shape (m, n)
        Asset time series with ``m`` observations (rows in time order) and
        ``n`` assets (columns).

    Returns
    -------
    A_hat : ndarray of shape (n, n)
        Satisfies ``S_t = S_{t-1} A + Z_t`` (paper Eq. 2 and 6).
    S_hat : ndarray of shape (m-1, n)
        ``S_hat = S_{t-1} A_hat``.
    """
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] < 3:
        raise ValueError("S must be a 2-D array with at least three rows.")
    S_lag = S[:-1]
    S_cur = S[1:]
    # Eq. (6): A_hat = (S_{t-1}^T S_{t-1})^{-1} S_{t-1}^T S_t
    gram = S_lag.T @ S_lag
    A_hat = np.linalg.solve(gram, S_lag.T @ S_cur)
    S_hat = S_lag @ A_hat
    return A_hat, S_hat


def canonical_decomposition(S: np.ndarray) -> BoxTiaoResult:
    """Full Box & Tiao canonical decomposition of a VAR(1) series.

    The generalised eigenvalue problem solved is
    ``det(lambda * (S_t^T S_t) - (S_hat^T S_hat)) = 0`` (paper Eq. 5 and 7);
    the eigenvectors ``z`` of ``M = Gamma^{-1/2} A^T Gamma A Gamma^{-1/2}``
    are mapped back to portfolio weights ``x = Gamma^{-1/2} z``.
    """
    S = np.asarray(S, dtype=float)
    A_hat, S_hat = estimate_var1(S)
    S_cur = S[1:]  # paper aligns both sums on the common sample (t = 1..m-1)
    B = S_cur.T @ S_cur  # S_t^T S_t, the Gamma proxy of Eq. (7)
    A = S_hat.T @ S_hat  # S_hat^T S_hat
    # scipy.linalg.eigh solves the generalised symmetric eigenvalue problem
    # A z = lambda B z with ascending eigenvalues; smallest => most mean reverting.
    eigvals, eigvecs = eigh(A, B)
    # Normalise each portfolio weight to unit Euclidean norm.
    norms = np.linalg.norm(eigvecs, axis=0)
    weights = eigvecs / np.where(norms > 0, norms, 1.0)
    return BoxTiaoResult(
        weights=weights,
        predictability=np.asarray(eigvals, dtype=float),
        A_hat=A_hat,
        gamma_hat=B / S_cur.shape[0],
    )


def predictability(x: np.ndarray, A_hat: np.ndarray, gamma: np.ndarray) -> float:
    """Predictability of a portfolio ``x`` (paper, variance ratio before Eq. 4)."""
    x = np.asarray(x, dtype=float).ravel()
    num = float(x @ A_hat.T @ gamma @ A_hat @ x)
    den = float(x @ gamma @ x)
    if den <= 0:
        raise ValueError("Portfolio variance is non-positive.")
    return num / den
