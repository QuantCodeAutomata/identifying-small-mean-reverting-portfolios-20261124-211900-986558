# AGENTS.md

Guide for coding agents and contributors working on this repository.

## What this repository implements

Replication of the nine experiments of **"Identifying Small Mean Reverting
Portfolios"** by Alexandre d'Aspremont (2011): dense Box & Tiao canonical
decomposition, greedy and semidefinite search for sparse mean-reverting
portfolios, covariance selection, LASSO-penalised VAR estimation, rolling in-
and out-of-sample mean reversion studies on U.S. swap rates and U.S. dollar
exchange rates, and the convergence-trading backtest of Section 5.3.

## Layout (binding)

```
src/     flat library modules (algorithms, estimators, rolling backtest)
exp/     one script per experiment, exp.<name> callable from the repo root
data/    data access (data_loader.py); download cache (data/cache/) ignored
tests/   pytest tests for src/ and exp/
results/ generated tables, figures, metrics, and RESULTS.md
```

No other top-level directories; each of `src/`, `exp/`, `data/` contains an
empty `__init__.py`.

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env            # only needed by the QuantX / Massive paths

python -m pytest                        # full suite
python -m pytest -m "not market_data"   # skip live-download tests

python -m exp.exp1_box_tiao
python -m exp.exp2_greedy
python -m exp.exp3_sdp
python -m exp.exp4_benchmark
python -m exp.exp5_covariance_selection
python -m exp.exp6_lasso_var
python -m exp.exp7_swap_sparsity
python -m exp.exp8_fx_sparsity
python -m exp.exp9_convergence_trading

./reproduce.sh                          # runs tests + every experiment
```

Each experiment script overwrites its outputs in `results/`. After a rerun
the numbers and verdicts in `results/RESULTS.md` must be checked against the
new files and updated by hand.

## Conventions

* All experiment parameters live in `src/config.py` as module-level upper-case
  variables; each has a comment with its meaning, unit, and paper reference.
  No parameter dicts or dataclasses.
* All market data is loaded through `data/data_loader.py`. Experiments must
  not reach past `data/` for raw downloads.
* Tests use small fixtures or synthetic inputs. Integration tests that touch
  FRED are marked `@pytest.mark.market_data` (registered in
  `tests/conftest.py`) and are skipped when network access is unavailable.
* API keys (if used) come from `.env` via `python-dotenv`; `.env` is
  gitignored. `.env.example` holds only empty placeholders.
* All code and comments in English, type hints on public functions, no
  secrets in files or logs.
