"""Shared rolling-window machinery for the Section 5 empirical experiments.

Both the swap-rate study (Figure 7) and the FX study (Figure 9) use the same
pattern: an estimation window of length ``L_in``, an out-of-sample window of
length ``L_out`` immediately after it, and a step of ``step`` days between
successive windows (paper page 16). The functions in this module iterate over
such windows and compute the greedy sparse portfolios, their in-sample and
out-of-sample mean reversion coefficients, and the out-of-sample price range.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.box_tiao import estimate_var1
from src.config import DELTA_T
from src.greedy import greedy_search
from src.lasso_var import lasso_var_fit, var_predicted
from src.ou import estimate_ou


@dataclass(frozen=True)
class WindowResult:
    """Summary of a single rolling-window pass."""

    start: int
    in_lam: np.ndarray            # shape (n,)
    out_lam: np.ndarray           # shape (n,)
    out_range: np.ndarray         # shape (n,); max - min of the portfolio values
    weights: list[np.ndarray]     # length n; column k has the k-th portfolio weights


def _build_AB_unpenalized(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, S_hat = estimate_var1(S)
    S_cur = S[1:]
    A = S_hat.T @ S_hat
    B = S_cur.T @ S_cur
    return A, B


def _build_AB_penalized(
    S: np.ndarray,
    rho: float,
    zero_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (A, B) using LASSO VAR for A and penalised Gaussian MLE for Gamma.

    The precision estimated by the covariance selection step is used to form a
    well-conditioned positive-definite plug-in for ``Gamma`` (paper Section 4.3).
    """
    from src.covariance_selection import covariance_selection

    S_lag = S[:-1]
    S_cur = S[1:]
    # Penalised Gamma comes from covariance selection on the centred series.
    centred = S_cur - S_cur.mean(axis=0, keepdims=True)
    cov_res = covariance_selection(centred, rho=rho)
    gamma_est = np.asarray(cov_res.covariance)
    gamma = S_cur.shape[0] * gamma_est  # match the scale of S_cur^T S_cur
    lasso = lasso_var_fit(S, zero_fraction=zero_fraction)
    S_hat = S_lag @ lasso.A
    A = S_hat.T @ S_hat
    return A, gamma


def rolling_sparse_study(
    S: np.ndarray,
    est_window: int,
    oos_window: int,
    step: int,
    *,
    penalized: bool = False,
    rho: float = 0.01,
    zero_fraction: float = 0.20,
    delta_t: float = DELTA_T,
) -> list[WindowResult]:
    """Run the Section 5.2 rolling-window procedure.

    For each start index the function:
       1. Builds matrices ``A`` and ``B`` from the estimation window (either
          OLS/sample covariance or LASSO VAR + covariance-selection);
       2. Runs the greedy search for every cardinality ``k = 1, ..., n``;
       3. Measures the in-sample and out-of-sample mean reversion
          coefficients and the out-of-sample price range for each portfolio.
    """
    S = np.asarray(S, dtype=float)
    m, n = S.shape
    results: list[WindowResult] = []
    i = 0
    while i + est_window + oos_window <= m:
        S_in = S[i: i + est_window]
        S_out = S[i + est_window: i + est_window + oos_window]
        if penalized:
            A, B = _build_AB_penalized(S_in, rho=rho, zero_fraction=zero_fraction)
        else:
            A, B = _build_AB_unpenalized(S_in)
        try:
            path = greedy_search(A, B, minimise=True)
        except Exception:
            i += step
            continue
        in_lam = np.empty(n)
        out_lam = np.empty(n)
        out_range = np.empty(n)
        for k, x in enumerate(path.weights):
            p_in = S_in @ x
            p_out = S_out @ x
            try:
                in_lam[k] = estimate_ou(p_in, delta_t=delta_t).lam
            except Exception:
                in_lam[k] = np.nan
            try:
                out_lam[k] = estimate_ou(p_out, delta_t=delta_t).lam
            except Exception:
                out_lam[k] = np.nan
            out_range[k] = float(p_out.max() - p_out.min())
        results.append(WindowResult(
            start=i,
            in_lam=in_lam,
            out_lam=out_lam,
            out_range=out_range,
            weights=path.weights,
        ))
        i += step
    return results


def aggregate_windows(results: list[WindowResult]) -> dict[str, np.ndarray]:
    """Return per-cardinality mean and std of the rolling-window outputs."""
    if not results:
        return {"in_lam_mean": np.array([]),
                "in_lam_std": np.array([]),
                "out_lam_mean": np.array([]),
                "out_lam_std": np.array([]),
                "out_range_mean": np.array([]),
                "out_range_std": np.array([])}
    in_mat = np.vstack([r.in_lam for r in results])
    out_mat = np.vstack([r.out_lam for r in results])
    rng_mat = np.vstack([r.out_range for r in results])
    return {
        "in_lam_mean": np.nanmean(in_mat, axis=0),
        "in_lam_std": np.nanstd(in_mat, axis=0, ddof=1),
        "out_lam_mean": np.nanmean(out_mat, axis=0),
        "out_lam_std": np.nanstd(out_mat, axis=0, ddof=1),
        "out_range_mean": np.nanmean(rng_mat, axis=0),
        "out_range_std": np.nanstd(rng_mat, axis=0, ddof=1),
    }
