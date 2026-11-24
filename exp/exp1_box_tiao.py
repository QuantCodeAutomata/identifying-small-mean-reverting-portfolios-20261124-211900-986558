"""Experiment 1: Box & Tiao canonical decomposition of U.S. swap rates.

Reproduces Table 1 of the paper (Section 2): eight canonical portfolios
ranked by predictability, with each one's mean-reversion coefficient,
Dickey-Fuller p-value, volatility and half-life.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_swap_rates, to_matrix
from src.box_tiao import canonical_decomposition
from src.config import DELTA_T, EXP1_N_PORTFOLIOS, ILLUSTRATIVE_N_DAYS, RESULTS_DIR
from src.ou import estimate_ou


def main() -> dict:
    # Load the full available sample, then use the last 100 trading days as
    # the "100 days of daily data" the paper uses for Figure 1 and Table 1.
    frame = load_swap_rates()
    window = frame.iloc[-ILLUSTRATIVE_N_DAYS:]
    S = to_matrix(window)
    result = canonical_decomposition(S)

    rows = []
    for k in range(EXP1_N_PORTFOLIOS):
        x = result.weights[:, k]
        portfolio = S @ x
        ou = estimate_ou(portfolio, delta_t=DELTA_T)
        rows.append({
            "rank": k + 1,
            "predictability": float(result.predictability[k]),
            "mean_reversion_lambda": ou.lam,
            "half_life_days": ou.half_life / DELTA_T,
            "volatility": ou.volatility,
            "adf_pvalue": ou.p_value,
            **{f"w_{label}": float(x[i]) for i, label in enumerate(frame.columns)},
        })
    table = pd.DataFrame(rows)
    out_csv = os.path.join(RESULTS_DIR, "exp1_box_tiao_table.csv")
    table.to_csv(out_csv, index=False)

    # Figure: the most mean-reverting portfolio over the 100 days.
    best = S @ result.weights[:, 0]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(window.index, best, label="Most mean-reverting portfolio")
    ax.set_title("Box & Tiao: most mean-reverting portfolio of U.S. swap rates")
    ax.set_xlabel("date")
    ax.set_ylabel("portfolio value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp1_box_tiao_best_portfolio.png"), dpi=120)
    plt.close(fig)

    summary = {
        "window_start": str(window.index.min().date()),
        "window_end": str(window.index.max().date()),
        "n_observations": int(len(window)),
        "columns": list(frame.columns),
        "mean_reversion_lambda_min": float(table["mean_reversion_lambda"].min()),
        "mean_reversion_lambda_max": float(table["mean_reversion_lambda"].max()),
        "half_life_days_min": float(table["half_life_days"].min()),
    }
    with open(os.path.join(RESULTS_DIR, "exp1_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("exp1 done:", summary)
    return summary


if __name__ == "__main__":
    main()
