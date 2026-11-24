"""Experiment 5: covariance selection graphs for swap rates and FX rates.

Reproduces Figures 2 and 8 of the paper (Section 4.1): the sparse inverse
covariance of the U.S. swap rate panel (rho = 0.1) and of the FX panel
(rho = 0.01), and their induced conditional-dependence graphs.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from data.data_loader import load_fx_rates, load_swap_rates, to_matrix
from src.config import COV_SELECT_RHO_FX, COV_SELECT_RHO_SWAP, RESULTS_DIR
from src.covariance_selection import (
    covariance_selection,
    largest_connected_cluster,
    precision_to_graph,
)


def _draw_graph(graph: nx.Graph, out_png: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    if graph.number_of_nodes() == 0:
        plt.close(fig)
        return
    pos = nx.spring_layout(graph, seed=1, k=1.0)
    edge_colors, widths = [], []
    for _, _, data in graph.edges(data=True):
        edge_colors.append("green" if data["weight"] > 0 else "red")
        widths.append(1.0 + 4.0 * min(1.0, abs(data["weight"])))
    nx.draw_networkx_nodes(graph, pos, node_size=300, node_color="lightblue", ax=ax)
    nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=widths, ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def _run_block(data: pd.DataFrame, rho: float, label: str) -> dict:
    # Covariance selection assumes a stationary Gaussian sample; swap and FX
    # levels are non-stationary, so we fit it on daily first differences
    # (the innovation process of the VAR model of Section 4.1).
    diffs = data.diff().dropna(how="any")
    # Scale each column to unit standard deviation so the L1 penalty is on
    # partial correlations rather than raw covariances; this is standard
    # practice and makes the ``rho`` values the paper reports comparable
    # across assets with very different scales (percent swap rates vs. FX).
    scales = diffs.std(ddof=0).replace(0.0, 1.0)
    X = to_matrix((diffs - diffs.mean()) / scales)
    penalised = covariance_selection(X, rho=rho)
    sample_prec = np.linalg.pinv(penalised.sample_covariance)
    graph_pen = precision_to_graph(penalised.precision, list(data.columns))
    graph_unp = precision_to_graph(sample_prec, list(data.columns),
                                   threshold=1e-6 * float(np.max(np.abs(sample_prec))))
    _draw_graph(
        graph_pen,
        os.path.join(RESULTS_DIR, f"exp5_{label}_penalised_graph.png"),
        f"{label}: covariance selection (rho={rho})",
    )
    _draw_graph(
        graph_unp,
        os.path.join(RESULTS_DIR, f"exp5_{label}_unpenalised_graph.png"),
        f"{label}: sample inverse covariance",
    )
    cluster = largest_connected_cluster(graph_pen)
    pd.DataFrame({
        "asset": list(data.columns),
    }).to_csv(os.path.join(RESULTS_DIR, f"exp5_{label}_assets.csv"), index=False)
    np.save(os.path.join(RESULTS_DIR, f"exp5_{label}_precision.npy"), penalised.precision)
    return {
        "rho": rho,
        "n_assets": int(data.shape[1]),
        "n_observations": int(data.shape[0]),
        "n_edges_penalised": int(graph_pen.number_of_edges()),
        "n_edges_unpenalised": int(graph_unp.number_of_edges()),
        "largest_cluster_size": int(len(cluster)),
        "largest_cluster": sorted(cluster),
    }


def main() -> dict:
    summary: dict = {}
    swaps = load_swap_rates()
    summary["swap"] = _run_block(swaps, COV_SELECT_RHO_SWAP, "swap")
    fx = load_fx_rates()
    summary["fx"] = _run_block(fx, COV_SELECT_RHO_FX, "fx")
    summary["fx"]["dropped_currencies"] = list(fx.attrs.get("dropped", []))
    with open(os.path.join(RESULTS_DIR, "exp5_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("exp5 done:", summary)
    return summary


if __name__ == "__main__":
    main()
