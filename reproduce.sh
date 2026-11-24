#!/usr/bin/env bash
# Reproduce the replication end-to-end from a clean checkout.
#
# Runs the test suite, then every experiment script in the paper's order.
# Experiment outputs land in results/.
set -eu

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

python -m pytest -m "not market_data"

python -m exp.exp1_box_tiao
python -m exp.exp2_greedy
python -m exp.exp3_sdp
python -m exp.exp4_benchmark
python -m exp.exp5_covariance_selection
python -m exp.exp6_lasso_var
python -m exp.exp7_swap_sparsity
python -m exp.exp8_fx_sparsity
python -m exp.exp9_convergence_trading
