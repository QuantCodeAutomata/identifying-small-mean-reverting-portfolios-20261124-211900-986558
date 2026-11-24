"""Experiment 7: mean reversion vs. sparsity on U.S. swap rates.

Reproduces Figure 7 and Table 2 of the paper (Section 5.2): rolling greedy
search on 200-day estimation windows with a 50-day step, measuring in-sample
and out-of-sample mean reversion coefficients and the out-of-sample price
range for every cardinality.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_swap_rates, to_matrix
from src.config import (
    EXP7_EST_WINDOW,
    EXP7_OOS_WINDOW,
    EXP7_STEP,
    RESULTS_DIR,
)
from src.rolling import aggregate_windows, rolling_sparse_study


def main() -> dict:
    frame = load_swap_rates()
    S = to_matrix(frame)
    windows = rolling_sparse_study(
        S,
        est_window=EXP7_EST_WINDOW,
        oos_window=EXP7_OOS_WINDOW,
        step=EXP7_STEP,
        penalized=False,
    )
    agg = aggregate_windows(windows)
    n = S.shape[1]
    ks = np.arange(1, n + 1)

    table = pd.DataFrame({
        "k": ks,
        "in_lambda_mean": agg["in_lam_mean"],
        "in_lambda_std": agg["in_lam_std"],
        "out_lambda_mean": agg["out_lam_mean"],
        "out_lambda_std": agg["out_lam_std"],
        "out_range_mean_bp": 1e4 * agg["out_range_mean"],
        "out_range_std_bp": 1e4 * agg["out_range_std"],
    })
    table.to_csv(os.path.join(RESULTS_DIR, "exp7_swap_sparsity.csv"), index=False)

    # Table 2-like portfolio composition from the median window.
    if windows:
        mid = len(windows) // 2
        comp = pd.DataFrame(
            np.column_stack(windows[mid].weights),
            index=list(frame.columns),
            columns=[f"k={k}" for k in ks],
        )
        comp.to_csv(os.path.join(RESULTS_DIR, "exp7_portfolio_composition.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].errorbar(ks, agg["in_lam_mean"], yerr=agg["in_lam_std"], marker="o",
                     label="in-sample")
    axes[0].errorbar(ks, agg["out_lam_mean"], yerr=agg["out_lam_std"], marker="s",
                     label="out-of-sample")
    axes[0].set_xlabel("cardinality k")
    axes[0].set_ylabel("mean reversion coefficient")
    axes[0].set_title("In- vs out-of-sample mean reversion")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    axes[1].errorbar(ks, 1e4 * agg["out_range_mean"], yerr=1e4 * agg["out_range_std"],
                     marker="o")
    axes[1].set_xlabel("cardinality k")
    axes[1].set_ylabel("out-of-sample price range (bp)")
    axes[1].set_title("Out-of-sample portfolio range")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp7_swap_sparsity.png"), dpi=120)
    plt.close(fig)

    summary = {
        "n_windows": len(windows),
        "in_lambda_mean": agg["in_lam_mean"].tolist(),
        "out_lambda_mean": agg["out_lam_mean"].tolist(),
        "out_range_mean_bp": (1e4 * agg["out_range_mean"]).tolist(),
    }
    with open(os.path.join(RESULTS_DIR, "exp7_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("exp7 done:", summary)
    return summary


if __name__ == "__main__":
    main()
