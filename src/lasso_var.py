"""LASSO-penalised VAR(1) estimation (paper Section 4.2, Eq. 15).

Replaces the OLS column-wise regression of ``A`` by an L1-penalised one,
tuning the penalty so that approximately a target fraction of the n*n
coefficients is driven to zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Lasso


@dataclass(frozen=True)
class LassoVARResult:
    """Output of :func:`lasso_var_fit`."""

    A: np.ndarray
    gamma: float
    zero_fraction: float


def _fit_one_column(S_lag: np.ndarray, target: np.ndarray, gamma: float) -> np.ndarray:
    # sklearn's objective is (1/(2n)) * ||y - Xw||^2 + alpha ||w||_1.
    # Paper Eq. 15: argmin ||S_it - S_{t-1} x||^2 + gamma ||x||_1.
    # Hence alpha = gamma / (2n). We set alpha directly from this mapping.
    n_obs = S_lag.shape[0]
    alpha = gamma / (2.0 * n_obs) if n_obs > 0 else gamma
    model = Lasso(alpha=alpha, fit_intercept=False, max_iter=50000, tol=1e-6)
    model.fit(S_lag, target)
    return model.coef_


def lasso_var_fit(
    S: np.ndarray,
    *,
    zero_fraction: float = 0.20,
    tol: float = 0.005,
    max_bisect: int = 60,
) -> LassoVARResult:
    """Estimate a sparse VAR(1) matrix with a target fraction of zeros.

    The LASSO penalty ``gamma`` is tuned by bisection so that approximately
    ``zero_fraction`` of the ``n * n`` coefficients of ``A`` are zeroed out
    (paper: "set to zero out 20% of the regression coefficients").
    """
    S = np.asarray(S, dtype=float)
    S_lag = S[:-1]
    S_cur = S[1:]
    n = S.shape[1]

    # Rough starting range: zero penalty -> OLS (0% zeros); very large penalty
    # -> all zeros (100%).
    gamma_lo, gamma_hi = 0.0, max(1.0, 10.0 * np.linalg.norm(S_lag.T @ S_cur, np.inf))

    def frac_zeros(gamma: float) -> tuple[float, np.ndarray]:
        cols = [_fit_one_column(S_lag, S_cur[:, j], gamma) for j in range(n)]
        A = np.column_stack(cols)
        frac = float(np.mean(A == 0.0))
        return frac, A

    frac_hi, A_hi = frac_zeros(gamma_hi)
    while frac_hi < zero_fraction and gamma_hi < 1e12:
        gamma_hi *= 10.0
        frac_hi, A_hi = frac_zeros(gamma_hi)

    best_gamma, best_A, best_frac = gamma_hi, A_hi, frac_hi
    for _ in range(max_bisect):
        gamma_mid = 0.5 * (gamma_lo + gamma_hi)
        frac_mid, A_mid = frac_zeros(gamma_mid)
        if abs(frac_mid - zero_fraction) < abs(best_frac - zero_fraction):
            best_gamma, best_A, best_frac = gamma_mid, A_mid, frac_mid
        if abs(frac_mid - zero_fraction) <= tol:
            return LassoVARResult(A=A_mid, gamma=gamma_mid, zero_fraction=frac_mid)
        if frac_mid < zero_fraction:
            gamma_lo = gamma_mid
        else:
            gamma_hi = gamma_mid
    return LassoVARResult(A=best_A, gamma=best_gamma, zero_fraction=best_frac)


def var_predicted(S: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Return the predicted series ``S_hat_t = S_{t-1} A`` of length ``m - 1``."""
    return S[:-1] @ A
