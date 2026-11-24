# Results

**Paper:** Alexandre d'Aspremont, *"Identifying Small Mean Reverting Portfolios"*,
Quantitative Finance (2011).

This file is written and updated by hand from the artefacts saved by the
experiment scripts in `results/`. After every rerun the numbers and the
verdicts below must be checked against the new output files.

Verdict scale (applied to every compared metric):
* **reproduced** - within 10% of the paper's value and same sign, or inside
  the paper's reported range or confidence interval;
* **partially reproduced** - same qualitative behaviour, deviation above 10%
  but below the materiality threshold;
* **not reproduced** - opposite sign, more than 50% away, outside the paper's
  reported range, or a qualitative result of the paper does not hold.

---

## Experiment 1 - Box & Tiao canonical decomposition for dense mean reverting portfolios

See `results/exp1_box_tiao_table.csv` and `results/exp1_box_tiao_portfolios.png`.
Numbers and verdicts are filled in after the final run of
`python -m exp.exp1_box_tiao`.

## Experiment 2 - Greedy search algorithm for sparse mean reverting portfolios

See `results/exp2_greedy_path.csv` and `results/exp2_greedy_mean_reversion.png`.

## Experiment 3 - Semidefinite relaxation for sparse mean reverting portfolios

See `results/exp3_sdp_path.csv` and `results/exp3_sdp_vs_greedy.png`.

## Experiment 4 - Numerical performance: greedy search vs. SDP relaxation

See `results/exp4_cpu_time.csv` and `results/exp4_cpu_time.png`.

## Experiment 5 - Covariance selection and graphical model estimation

See `results/exp5_swap_graph.png`, `results/exp5_fx_graph.png`, and
`results/exp5_fx_cluster.csv`.

## Experiment 6 - LASSO-penalised VAR estimation

See `results/exp6_lasso_sparsity.csv` and `results/exp6_lasso_support.png`.

## Experiment 7 - Mean reversion vs. sparsity on U.S. swap rates

See `results/exp7_swap_mean_reversion.csv`, `results/exp7_swap_portfolio.csv`,
`results/exp7_swap_curves.png`, and `results/exp7_swap_range.png`.

## Experiment 8 - Mean reversion vs. sparsity on FX rates with penalised estimation

See `results/exp8_fx_mean_reversion.csv`, `results/exp8_fx_range.csv`, and
`results/exp8_fx_curves.png`.

## Experiment 9 - Convergence trading on sparse mean reverting portfolios

See `results/exp9_sharpe.csv` and `results/exp9_sharpe.png`.

---

## Deviations from the paper

* The FRED constant-maturity swap-rate series (DSWP1, DSWP2, ..., DSWP30)
  begin on 2000-07-03, so the realised swap-rate sample used for the empirical
  study of Section 5 covers 2000-07 through 2005-12 rather than the
  1998-2005 window the paper reports. The methodology, frequency, and
  maturities (1Y, 2Y, 3Y, 4Y, 5Y, 7Y, 10Y, 30Y) are kept.

* The paper does not state the exact 100-day window of Figure 1; the
  illustrative decomposition uses the final 100 trading days of the realised
  swap-rate sample.

* The paper quotes 45 U.S. dollar exchange rates for the covariance-selection
  study; the H.10 FRED series provide 20-23 of them in the April 2002 to
  April 2007 window. The replication lists the dropped currencies in
  `results/exp5_fx_dropped.csv`. The methodology (penalised MLE with
  rho = 0.01) and frequency are unchanged.

* Experiment 8 reuses the Section 5.2 rolling-window parameters
  (200-day estimation, 200-day out-of-sample, 50-day step) because the paper
  does not state the FX rolling-window parameters separately.

* The risk-free rate r used in the Jurek-Yang optimal-trading rule of
  Experiment 9 is set to zero because the paper does not fix a value for it.

## Not reproduced

Each item below lists a paper claim or artefact that this repository does not
reproduce, with a substantive reason.

* The second half of the Section 5.1 CPU-time benchmark (SDP running times)
  is not reported: the SDP relaxation is run only on the swap-rate universe
  (n = 8); the greedy running times at n = 10, 20, 50, 100 are reported as in
  the paper.
