"""Experiment 2: greedy search for sparse mean-reverting swap portfolios.

Reproduces the left panel of Figure 5 and the left panel of Figure 6 of the
paper (Section 3.1 and Section 5.1): mean-reversion coefficient vs. target
cardinality on 100 days of U.S. swap rates.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_swap_rates, to_matrix
from src.box_tiao import estimate_var1
from src.config import DELTA_T, ILLUSTRATIVE_N_DAYS, RESULTS_DIR
from src.greedy import greedy_search
from src.ou import estimate_ou


def build_AB(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, S_hat = estimate_var1(S)
    S_cur = S[1:]
    return S_hat.T @ S_hat, S_cur.T @ S_cur


def main() -> dict:
    frame = load_swap_rates()
    window = frame.iloc[-ILLUSTRATIVE_N_DAYS:]
    S = to_matrix(window)
    A, B = build_AB(S)
    path = greedy_search(A, B, minimise=True)

    rows = []
    for k, x in enumerate(path.weights, start=1):
        portfolio = S @ x
        ou = estimate_ou(portfolio, delta_t=DELTA_T)
        rows.append({
            "k": k,
            "support": ",".join(frame.columns[i] for i in path.supports[k - 1]),
            "objective": float(path.objective[k - 1]),
            "mean_reversion_lambda": ou.lam,
            "half_life_days": ou.half_life / DELTA_T,
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "exp2_greedy_path.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["k"], table["mean_reversion_lambda"], "o-")
    ax.set_xlabel("portfolio cardinality k")
    ax.set_ylabel("mean reversion coefficient (per year)")
    ax.set_title("Greedy search on U.S. swap rates")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp2_greedy_mean_reversion.png"), dpi=120)
    plt.close(fig)

    summary = {
        "window_start": str(window.index.min().date()),
        "window_end": str(window.index.max().date()),
        "mean_reversion_at_k1": float(table["mean_reversion_lambda"].iloc[0]),
        "mean_reversion_at_k_n": float(table["mean_reversion_lambda"].iloc[-1]),
        "supports": [list(path.supports[k - 1]) for k in range(1, len(path.supports) + 1)],
    }
    with open(os.path.join(RESULTS_DIR, "exp2_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("exp2 done:", summary)
    return summary


if __name__ == "__main__":
    main()
