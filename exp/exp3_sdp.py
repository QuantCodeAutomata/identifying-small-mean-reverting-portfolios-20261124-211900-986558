"""Experiment 3: semidefinite relaxation for sparse mean-reverting portfolios.

Reproduces the SDP curve of Figure 5 (Section 5.1) on 100 days of U.S. swap
rates and prints both the recovered Rayleigh quotient and the SDP upper bound.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_swap_rates, to_matrix
from src.config import DELTA_T, EXP3_SOLVER, ILLUSTRATIVE_N_DAYS, RESULTS_DIR
from src.ou import estimate_ou
from src.sdp import sdp_sparse
from exp.exp2_greedy import build_AB


def main() -> dict:
    frame = load_swap_rates()
    window = frame.iloc[-ILLUSTRATIVE_N_DAYS:]
    S = to_matrix(window)
    A, B = build_AB(S)
    # Mean reversion = minimise predictability => minimise x^T A x / x^T B x.
    # The SDP primal maximises Tr(A Y); passing (B, A) instead flips the ratio.
    n = S.shape[1]
    rows = []
    for k in range(1, n + 1):
        sol = sdp_sparse(B, A, k, maximise=True, solver=EXP3_SOLVER)
        x = sol.weights
        portfolio = S @ x
        ou = estimate_ou(portfolio, delta_t=DELTA_T)
        rows.append({
            "k": k,
            "sdp_status": sol.status,
            "sdp_primal": sol.sdp_bound,
            "recovered_ratio": sol.objective,
            "mean_reversion_lambda": ou.lam,
            "support_size": int(np.sum(np.abs(x) > 1e-6)),
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "exp3_sdp_path.csv"), index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(table["k"], table["mean_reversion_lambda"], "s-", label="SDP relaxation")
    ax.set_xlabel("portfolio cardinality k")
    ax.set_ylabel("mean reversion coefficient (per year)")
    ax.set_title("SDP relaxation on U.S. swap rates")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp3_sdp_mean_reversion.png"), dpi=120)
    plt.close(fig)

    summary = {
        "solver": EXP3_SOLVER,
        "mean_reversion_at_k1": float(table["mean_reversion_lambda"].iloc[0]),
        "mean_reversion_at_k_n": float(table["mean_reversion_lambda"].iloc[-1]),
    }
    with open(os.path.join(RESULTS_DIR, "exp3_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("exp3 done:", summary)
    return summary


if __name__ == "__main__":
    main()
