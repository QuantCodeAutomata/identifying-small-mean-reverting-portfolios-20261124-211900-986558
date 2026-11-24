# Identifying Small Mean Reverting Portfolios

Replication of **"Identifying Small Mean Reverting Portfolios"** by Alexandre
d'Aspremont (2011).

This repository reproduces the nine experiments of the paper on U.S. constant-
maturity swap rates and U.S. dollar exchange rates:

| # | Title | Section |
|---|-------|---------|
| 1 | Box & Tiao canonical decomposition for dense mean reverting portfolios | §2 / Table 1 |
| 2 | Greedy search algorithm for sparse mean reverting portfolios | §3.1 / Figures 5, 6 |
| 3 | Semidefinite relaxation for sparse mean reverting portfolios | §3.2 / Figure 5 |
| 4 | Numerical performance: greedy search vs. SDP | §5.1 / Figure 5 |
| 5 | Covariance selection and graphical model estimation | §4.1 / Figures 2, 8 |
| 6 | LASSO-penalized VAR estimation | §4.2 |
| 7 | Mean reversion vs. sparsity on U.S. swap rates | §5.2 / Figure 7, Table 2 |
| 8 | Mean reversion vs. sparsity on FX rates with penalized estimation | §5.2 / Figure 9 |
| 9 | Convergence trading on sparse mean reverting portfolios | §5.3 / Figure 10 |

## Layout

```
.
├── src/     # Library (algorithms, estimators, rolling backtest)
├── exp/     # Executable experiment scripts (python -m exp.<name>)
├── tests/   # Unit and integration tests
├── data/    # Data access (pandas-datareader / FRED; cache gitignored)
├── results/ # Tables, figures, metrics, and RESULTS.md
├── requirements.txt
├── reproduce.sh
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # optional; only needed by the QuantX / Massive loaders
```

## Running the experiments

Every script runs from the repository root as a module:

```bash
python -m exp.exp1_box_tiao
python -m exp.exp2_greedy
python -m exp.exp3_sdp
python -m exp.exp4_benchmark
python -m exp.exp5_covariance_selection
python -m exp.exp6_lasso_var
python -m exp.exp7_swap_sparsity
python -m exp.exp8_fx_sparsity
python -m exp.exp9_convergence_trading
```

The scripts above write their tables and figures into `results/`, overwriting
any previous run. The numbers and verdicts in `results/RESULTS.md` must be
checked against the newly written files and updated by hand.

`./reproduce.sh` runs the tests and every experiment in order.

## Tests

```bash
python -m pytest                        # all tests
python -m pytest -m "not market_data"   # skip tests that need a live download
```

Tests marked `market_data` fetch a small slice from FRED and are skipped when
network access is not available.

## Data

Constant-maturity U.S. swap rates (eight maturities, 1998–2005, daily) and
U.S. dollar exchange rates (April 2002 – April 2007, daily) are pulled from
the Federal Reserve Economic Data service via `pandas-datareader`. Downloaded
series are cached under `data/cache/` and never committed. See
`data/data_loader.py` for the ticker lists and inversion conventions.

The FRED constant-maturity swap rate series start on 2000-07-03, so the
realised swap-rate sample covers 2000-07 through 2005-12 rather than the
1998–2005 window the paper reports; this and any other deviation are listed
in `results/RESULTS.md`.

## Paper configuration

All experiment parameters live in `src/config.py` as module-level upper-case
variables with a comment stating their meaning, unit, and paper reference.
`src/config.py` holds the full paper configuration; no parameters need to be
edited to reproduce the headline figures.
