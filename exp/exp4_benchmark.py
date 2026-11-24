"""Experiment 4: greedy vs. SDP numerical comparison.

Reproduces Figure 5 of the paper (Section 5.1): on 100 days of U.S. swap
rates, plot the mean-reversion coefficient obtained by greedy search and by
the SDP relaxation as a function of cardinality; on synthetic VAR(1) datasets
of growing size, record the CPU time of the greedy search for every
cardinality from 1 to n.
"""

from __future__ import annotations

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data.data_loader import load_swap_rates, to_matrix
from src.box_tiao import estimate_var1
from src.config import (
    DELTA_T,
    EXP3_SOLVER,
    EXP4_SYNTHETIC_N_OBS,
    EXP4_SYNTHETIC_SEED,
    EXP4_UNIVERSE_SIZES,
    ILLUSTRATIVE_N_DAYS,
    RESULTS_DIR,
)
from src.greedy import greedy_search
from src.ou import estimate_ou
from src.sdp import sdp_sparse
from exp.exp2_greedy import build_AB


def quality_comparison() -> pd.DataFrame:
    frame = load_swap_rates()
    window = frame.iloc[-ILLUSTRATIVE_N_DAYS:]
    S = to_matrix(window)
    A, B = build_AB(S)
    greedy_path = greedy_search(A, B, minimise=True)
    n = S.shape[1]
    rows = []
    for k in range(1, n + 1):
        x_g = greedy_path.weights[k - 1]
        lam_g = estimate_ou(S @ x_g, delta_t=DELTA_T).lam
        sol = sdp_sparse(B, A, k, maximise=True, solver=EXP3_SOLVER)
        lam_s = estimate_ou(S @ sol.weights, delta_t=DELTA_T).lam
        rows.append({
            "k": k,
            "greedy_lambda": lam_g,
            "sdp_lambda": lam_s,
        })
    return pd.DataFrame(rows)


def _random_var1(n: int, m: int, seed: int) -> np.ndarray:
    """Draw a stationary VAR(1) series with ``n`` assets and ``m`` observations."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    A /= 1.5 * np.linalg.norm(A, ord=2)  # keep the spectral radius < 1
    chol = np.tril(rng.standard_normal((n, n))) / np.sqrt(n)
    S = np.zeros((m + 50, n))
    S[0] = rng.standard_normal(n)
    for t in range(1, m + 50):
        S[t] = S[t - 1] @ A + rng.standard_normal(n) @ chol.T
    return S[50:]


def cpu_benchmark() -> pd.DataFrame:
    rows = []
    for n in EXP4_UNIVERSE_SIZES:
        S = _random_var1(n, EXP4_SYNTHETIC_N_OBS, EXP4_SYNTHETIC_SEED + n)
        _, S_hat = estimate_var1(S)
        A = S_hat.T @ S_hat
        B = S[1:].T @ S[1:]
        t0 = time.perf_counter()
        greedy_search(A, B, minimise=True)
        rows.append({"n": n, "cpu_seconds_greedy": time.perf_counter() - t0})
    return pd.DataFrame(rows)


def main() -> dict:
    quality = quality_comparison()
    quality.to_csv(os.path.join(RESULTS_DIR, "exp4_quality.csv"), index=False)
    cpu = cpu_benchmark()
    cpu.to_csv(os.path.join(RESULTS_DIR, "exp4_cpu.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(quality["k"], quality["greedy_lambda"], "o-", label="greedy")
    axes[0].plot(quality["k"], quality["sdp_lambda"], "s--", label="SDP")
    axes[0].set_xlabel("cardinality k")
    axes[0].set_ylabel("mean reversion coefficient")
    axes[0].set_title("Quality on U.S. swap rates")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(cpu["n"], cpu["cpu_seconds_greedy"], "o-")
    axes[1].set_xlabel("number of assets n")
    axes[1].set_ylabel("CPU time (seconds)")
    axes[1].set_title("Greedy search CPU cost")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "exp4_benchmark.png"), dpi=120)
    plt.close(fig)

    summary = {
        "quality_rows": quality.to_dict(orient="records"),
        "cpu_rows": cpu.to_dict(orient="records"),
    }
    with open(os.path.join(RESULTS_DIR, "exp4_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("exp4 done:", summary)
    return summary


if __name__ == "__main__":
    main()
