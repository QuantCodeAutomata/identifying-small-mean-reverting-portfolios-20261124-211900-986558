"""Ornstein-Uhlenbeck parameter estimation and derived quantities.

Implements the appendix of the paper (page 18):
    P_t = Pbar + exp(-lambda dt) (P_{t-dt} - Pbar) + noise
with the regression-based estimators of ``(mu, lambda, sigma)``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import adfuller

from src.config import DELTA_T


@dataclass(frozen=True)
class OUEstimate:
    """Regression-based estimates of an Ornstein-Uhlenbeck process."""

    mu: float           # long-run mean P-bar
    lam: float          # mean-reversion speed (per year with DELTA_T = 1/252)
    sigma: float        # instantaneous volatility
    half_life: float    # log(2)/lam
    p_value: float      # augmented Dickey-Fuller p-value
    volatility: float   # sample standard deviation of the series
    delta_t: float


def estimate_ou(series: np.ndarray, delta_t: float = DELTA_T) -> OUEstimate:
    """Estimate an OU process from a scalar time series.

    Uses the appendix formulas (page 18) for ``mu`` and ``lambda``, and the
    corrected normalisation for the residual standard deviation ``sigma``.
    """
    series = np.asarray(series, dtype=float).ravel()
    n = series.size
    if n < 4:
        raise ValueError("OU estimation needs at least 4 observations.")
    mu = float(np.mean(series))
    centred = series - mu
    p_cur = centred[1:]
    p_lag = centred[:-1]
    num = float(np.sum(p_cur * p_lag))
    den = float(np.sum(p_lag * p_lag))
    if den <= 0:
        raise ValueError("Degenerate variance in OU regression.")
    ratio = num / den
    # A non-positive or near-unit AR(1) ratio corresponds to the two edge cases
    # the OU model cannot represent: an overshooting (anti-correlated) series or
    # a non-stationary one. The paper's appendix (page 18) formulas break down
    # there. Cap the reported lambda at a half-life of half a trading day so
    # window-averaged statistics stay finite and comparable.
    lam_cap = float(np.log(2.0) / max(0.5 * delta_t, 1e-12))
    if ratio <= 0:
        lam = lam_cap
    else:
        lam = -float(np.log(max(ratio, 1e-12))) / delta_t
        lam = min(lam, lam_cap)
    if lam > 0 and np.isfinite(lam):
        residuals = p_cur - np.exp(-lam * delta_t) * p_lag
        var = float(np.sum(residuals ** 2))
        # Appendix: sigma^2 = 2 lam / ((1 - exp(-2 lam dt))(N - 2)) * sum(residuals^2)
        denom = (1.0 - np.exp(-2.0 * lam * delta_t)) * max(n - 2, 1)
        sigma = float(np.sqrt(2.0 * lam * var / denom)) if denom > 0 else float("nan")
    else:
        sigma = float("nan")
    half_life = float(np.log(2.0) / lam) if lam > 0 and np.isfinite(lam) else float("inf")
    # Augmented Dickey-Fuller to flag non-stationarity (i.e. no mean reversion).
    try:
        adf = adfuller(series, maxlag=1, regression="c", autolag=None)
        p_value = float(adf[1])
    except Exception:
        p_value = float("nan")
    return OUEstimate(
        mu=mu,
        lam=lam,
        sigma=sigma,
        half_life=half_life,
        p_value=p_value,
        volatility=float(np.std(series, ddof=1)),
        delta_t=delta_t,
    )
