"""Experiment 6: LASSO-penalised VAR(1) estimation on FX rates.

Reproduces the sparse VAR step of Section 4.2 and 4.3: given the cluster of
currencies isolated by covariance selection, tune the LASSO penalty gamma so
that approximately 20% of the n*n entries of A are driven to zero.
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
    LASSO_VAR_ZERO_FRAC,
    RESULTS_DIR,
)
from src.covariance_selection import (
    covariance_selection,
    largest_connected_cluster,
    precision_to_graph,
)
from src.lasso_var import lasso_var_fit


def _fx_cluster(fx: pd.DataFrame) -> list[str]:
    centred = to_matrix(fx.apply(lambda c: c - c.mean()))
    result = covariance_selection(centred, rho=COV_SELECT_RHO_FX)
    graph = precision_to_graph(result.precision, list(fx.columns))
    cluster = largest_connected_cluster(graph)
    return sorted(cluster) if cluster else list(fx.columns)


def main() -> dict:
    fx = load_fx_rates()
    cluster = _fx_cluster(fx)
    data = fx[cluster]
    S = to_matrix(data)

    lasso = lasso_var_fit(S, zero_fraction=LASSO_VAR_ZERO_FRAC)
    sparse_mask = lasso.A != 0.0
    pd.DataFrame(lasso.A, index=cluster, columns=cluster).to_csv(
        os.path.join(RESULTS_DIR, "exp6_lasso_A.csv"))
    pd.DataFrame(sparse_mask.astype(int), index=cluster, columns=cluster).to_csv(
        os.path.join(RESULTS_DIR, "exp6_lasso_A_mask.csv"))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sparse_mask.astype(int), cmap="Greys", aspect="auto")
    ax.set_xticks(range(len(cluster)))
    ax.set_xticklabels(cluster, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cluster)))
    ax.set_yticklabels(cluster, fontsize=7)
    ax.set_title(f"LASSO VAR nonzero pattern (zero fraction {lasso.zero_fraction:.2f})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp6_lasso_sparsity.png"), dpi=120)
    plt.close(fig)

    summary = {
        "cluster": cluster,
        "gamma": lasso.gamma,
        "zero_fraction": lasso.zero_fraction,
        "target_zero_fraction": LASSO_VAR_ZERO_FRAC,
    }
    with open(os.path.join(RESULTS_DIR, "exp6_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("exp6 done:", summary)
    return summary


if __name__ == "__main__":
    main()
