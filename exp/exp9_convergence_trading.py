"""Experiment 9: convergence trading on sparse swap-rate portfolios.

Reproduces Figure 10 of the paper (Section 5.3): roll a 100-day estimation
window followed by a 50-day out-of-sample window, estimate the
Ornstein-Uhlenbeck parameters of every greedy sparse portfolio, apply the
Jurek-Yang log-utility optimal strategy out of sample, and report the average
Sharpe ratio with and without a 1-bp bid-ask spread.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_swap_rates, to_matrix
from src.box_tiao import estimate_var1
from src.config import (
    DELTA_T,
    EXP9_BID_ASK_BP,
    EXP9_EST_WINDOW,
    EXP9_OOS_WINDOW,
    EXP9_RISK_FREE_RATE,
    EXP9_STEP,
    RESULTS_DIR,
    TRADING_DAYS,
)
from src.greedy import greedy_search
from src.ou import estimate_ou


def _convergence_trade(
    prices: np.ndarray,
    mu: float,
    lam: float,
    sigma: float,
    r: float,
    weights_turnover: float,
    bid_ask_bp: float = 0.0,
) -> np.ndarray:
    """Return daily P&L per unit of wealth under the log-utility optimal rule.

    The Jurek-Yang (2006) log-utility optimal number of shares is
    ``N_t = (lam * (mu - P_t) - r * P_t) / sigma^2``. Here wealth-normalised
    returns are the exposure times the one-step price change; the bid-ask
    cost ``bid_ask_bp`` (basis points of the traded notional) is subtracted on
    every day the exposure changes.
    """
    if sigma <= 0 or not np.isfinite(sigma):
        return np.zeros(len(prices) - 1)
    N = (lam * (mu - prices) - r * prices) / (sigma ** 2)
    dP = np.diff(prices)
    gross = N[:-1] * dP
    if bid_ask_bp > 0:
        turnover = np.abs(np.diff(N))
        # bid_ask_bp is a cost per traded unit; weights_turnover scales for the
        # effective traded notional of the underlying swap-rate combination.
        cost = (bid_ask_bp * 1e-4) * turnover * weights_turnover
        return gross - cost
    return gross


def main() -> dict:
    frame = load_swap_rates()
    S = to_matrix(frame)
    m, n = S.shape
    ks = np.arange(1, n + 1)
    sharpes = {False: [[] for _ in range(n)], True: [[] for _ in range(n)]}

    i = 0
    processed = 0
    while i + EXP9_EST_WINDOW + EXP9_OOS_WINDOW <= m:
        S_in = S[i: i + EXP9_EST_WINDOW]
        S_out = S[i + EXP9_EST_WINDOW: i + EXP9_EST_WINDOW + EXP9_OOS_WINDOW]
        _, S_hat = estimate_var1(S_in)
        A = S_hat.T @ S_hat
        B = S_in[1:].T @ S_in[1:]
        try:
            path = greedy_search(A, B, minimise=True)
        except Exception:
            i += EXP9_STEP
            continue
        for k, x in enumerate(path.weights):
            p_in = S_in @ x
            p_out = S_out @ x
            ou = estimate_ou(p_in, delta_t=DELTA_T)
            if not (np.isfinite(ou.lam) and ou.lam > 0 and np.isfinite(ou.sigma)
                    and ou.sigma > 0):
                for cost_on in (False, True):
                    sharpes[cost_on][k].append(np.nan)
                continue
            w_turnover = float(np.sum(np.abs(x)))
            for cost_on in (False, True):
                bp = EXP9_BID_ASK_BP if cost_on else 0.0
                pnl = _convergence_trade(p_out, ou.mu, ou.lam, ou.sigma,
                                         EXP9_RISK_FREE_RATE, w_turnover, bp)
                if pnl.size < 2 or pnl.std(ddof=1) <= 0:
                    sharpes[cost_on][k].append(np.nan)
                    continue
                sharpes[cost_on][k].append(
                    float(pnl.mean() / pnl.std(ddof=1) * np.sqrt(TRADING_DAYS))
                )
        processed += 1
        i += EXP9_STEP

    def _stats(samples: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        mean = np.array([np.nanmean(s) if s else np.nan for s in samples])
        std = np.array([np.nanstd(s, ddof=1) if len(s) > 1 else np.nan for s in samples])
        return mean, std

    mean_fric, std_fric = _stats(sharpes[False])
    mean_cost, std_cost = _stats(sharpes[True])
    table = pd.DataFrame({
        "k": ks,
        "sharpe_frictionless_mean": mean_fric,
        "sharpe_frictionless_std": std_fric,
        "sharpe_with_cost_mean": mean_cost,
        "sharpe_with_cost_std": std_cost,
    })
    table.to_csv(os.path.join(RESULTS_DIR, "exp9_convergence_trading.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.errorbar(ks, mean_fric, yerr=std_fric, marker="o", label="frictionless")
    ax.errorbar(ks, mean_cost, yerr=std_cost, marker="s", label=f"{EXP9_BID_ASK_BP}bp cost")
    ax.set_xlabel("cardinality k")
    ax.set_ylabel("annualised Sharpe ratio")
    ax.set_title("Convergence trading on sparse swap portfolios")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp9_convergence_trading.png"), dpi=120)
    plt.close(fig)

    summary = {
        "n_windows": int(processed),
        "sharpe_frictionless_mean": mean_fric.tolist(),
        "sharpe_with_cost_mean": mean_cost.tolist(),
        "bid_ask_bp": EXP9_BID_ASK_BP,
    }
    with open(os.path.join(RESULTS_DIR, "exp9_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("exp9 done:", summary)
    return summary


if __name__ == "__main__":
    main()
