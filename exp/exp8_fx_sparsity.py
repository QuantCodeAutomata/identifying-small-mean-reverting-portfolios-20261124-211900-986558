"""Experiment 8: mean reversion vs. sparsity on FX rates with penalised estimation.

Reproduces Figure 9 of the paper (Section 5.2): on the covariance-selection
cluster of U.S. dollar exchange rates, compare out-of-sample mean reversion
and portfolio range of sparse portfolios built from (a) unpenalised sample
estimates of (Gamma, A) and (b) penalised estimates (covariance selection for
Gamma, LASSO for A).
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_fx_rates, to_matrix
from src.config import (
    COV_SELECT_RHO_FX,
    EXP8_EST_WINDOW,
    EXP8_OOS_WINDOW,
    EXP8_STEP,
    LASSO_VAR_ZERO_FRAC,
    RESULTS_DIR,
)
from src.covariance_selection import (
    covariance_selection,
    largest_connected_cluster,
    precision_to_graph,
)
from src.rolling import aggregate_windows, rolling_sparse_study


def _cluster(fx: pd.DataFrame) -> list[str]:
    centred = to_matrix(fx.apply(lambda c: c - c.mean()))
    res = covariance_selection(centred, rho=COV_SELECT_RHO_FX)
    graph = precision_to_graph(res.precision, list(fx.columns))
    cluster = largest_connected_cluster(graph)
    return sorted(cluster) if cluster else list(fx.columns)


def main() -> dict:
    fx = load_fx_rates()
    cluster = _cluster(fx)
    data = fx[cluster]
    S = to_matrix(data)

    runs = {}
    for label, penalized in (("unpenalized", False), ("penalized", True)):
        wins = rolling_sparse_study(
            S,
            est_window=EXP8_EST_WINDOW,
            oos_window=EXP8_OOS_WINDOW,
            step=EXP8_STEP,
            penalized=penalized,
            rho=COV_SELECT_RHO_FX,
            zero_fraction=LASSO_VAR_ZERO_FRAC,
        )
        runs[label] = (wins, aggregate_windows(wins))

    n = S.shape[1]
    ks = np.arange(1, n + 1)
    table = pd.DataFrame({"k": ks})
    for label, (_, agg) in runs.items():
        table[f"{label}_out_lambda_mean"] = agg["out_lam_mean"]
        table[f"{label}_out_lambda_std"] = agg["out_lam_std"]
        table[f"{label}_out_range_mean"] = agg["out_range_mean"]
    table.to_csv(os.path.join(RESULTS_DIR, "exp8_fx_sparsity.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for label, (_, agg) in runs.items():
        axes[0].plot(ks, agg["out_lam_mean"], "o-", label=label)
        axes[1].plot(ks, agg["out_range_mean"], "s-", label=label)
    axes[0].set_xlabel("cardinality k")
    axes[0].set_ylabel("out-of-sample mean reversion")
    axes[0].set_title("Penalised vs. unpenalised, FX cluster")
    axes[0].grid(True, alpha=0.3); axes[0].legend()
    axes[1].set_xlabel("cardinality k")
    axes[1].set_ylabel("out-of-sample portfolio range")
    axes[1].set_title("Portfolio range vs. cardinality")
    axes[1].grid(True, alpha=0.3); axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp8_fx_sparsity.png"), dpi=120)
    plt.close(fig)

    summary = {
        "cluster": cluster,
        "n_windows": {k: len(v[0]) for k, v in runs.items()},
        "unpenalized_out_lambda_mean": runs["unpenalized"][1]["out_lam_mean"].tolist(),
        "penalized_out_lambda_mean": runs["penalized"][1]["out_lam_mean"].tolist(),
    }
    with open(os.path.join(RESULTS_DIR, "exp8_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("exp8 done:", summary)
    return summary


if __name__ == "__main__":
    main()
