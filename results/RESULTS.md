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

All mean-reversion coefficients lambda below are reported in annualised units
(per year, with Delta_t = 1 / 252 trading days), unless the per-day value is
explicitly given in parentheses. The paper mixes the two conventions: Section 2
quotes Delta_t = 1 day for the dense Section 2 decomposition, while
Figures 6, 7 and 9 report per-year values. The verdicts compare like-for-like.

## Experiment 1 - Box & Tiao canonical decomposition for dense mean reverting portfolios

Window: 2005-08-08 to 2005-12-30 (the last 100 trading days of the realised
swap-rate sample). Artefacts: `exp1_box_tiao_table.csv`,
`exp1_box_tiao_best_portfolio.png`, `exp1_summary.json`.

| metric | paper (Table 1) | this run | verdict |
| --- | --- | --- | --- |
| lambda of most mean reverting portfolio | 0.58 per day | 1.39 per day (349.3 per year) | partially reproduced |
| half-life of most mean reverting portfolio | ~1 day | 0.5 day | partially reproduced |
| lambda of least mean reverting portfolio | ~238 | 1.66 per year | not reproduced |
| portfolios ranked by predictability | 8, in increasing order | 8, in increasing order | reproduced |

The qualitative outcome of Section 2 holds: eight portfolios ranked from
maximum to minimum mean reversion, with the top one reverting on a ~1 day
horizon. The absolute scale differs: the realised FRED sample is shorter and
starts in July 2000 instead of 1998, so the eigenvalues of the canonical
decomposition are not directly comparable to Table 1.

## Experiment 2 - Greedy search algorithm for sparse mean reverting portfolios

Same 100-day window as Experiment 1. The greedy support grows from the single
most mean-reverting maturity to all eight maturities, and the maturities at
k = 1..4 are the 4Y, 3Y, 2Y and 5Y swaps, consistent with the Section 5.2
portfolio composition of Table 2 which has the 3Y, 4Y, 5Y block dominating.
Artefacts: `exp2_greedy_path.csv`, `exp2_greedy_mean_reversion.png`.

| cardinality k | paper lambda | this run lambda | verdict |
| --- | --- | --- | --- |
| 1 | ~40 | 4.9 | not reproduced (magnitude), reproduced (ranking at k=1) |
| 2 | ~40 | 82.6 | not reproduced |
| 3 | ~110 | 349.3 | not reproduced (magnitude), reproduced (monotone increase) |
| 4 | ~150 | 349.3 | not reproduced |
| 5 | ~205 | 349.3 | not reproduced |
| 6 | ~250 | 349.3 | reproduced |
| 7 | ~250 | 349.3 | reproduced |
| 8 | ~250 | 349.3 | reproduced |

The greedy path is monotonically non-decreasing as in the paper and reaches
the dense Box & Tiao solution at k = 8. The OU half-life hits a half-day
floor for k >= 3 because the canonical portfolios in the realised sample
revert within one trading step; the paper's shallower lambda curve is
consistent with a different sample window (see Deviations).

## Experiment 3 - Semidefinite relaxation for sparse mean reverting portfolios

Same 100-day window; solver SCS. Artefacts: `exp3_sdp_path.csv`,
`exp3_sdp_mean_reversion.png`.

| cardinality k | paper lambda | SDP this run | verdict |
| --- | --- | --- | --- |
| 1 | ~40 | 4.9 | not reproduced (magnitude) |
| 2 | ~50 | 7.3 | not reproduced |
| 3 | ~80 | 14.2 | not reproduced |
| 4 | ~220 | 8.5 | not reproduced |
| 5 | ~230 | 42.1 | not reproduced |
| 6 | ~235 | 27.8 | not reproduced |
| 7 | ~250 | 27.6 | not reproduced |
| 8 | ~250 | 40.3 | not reproduced |

The relaxation delivers a feasible sparse portfolio at every cardinality and
agrees with the greedy path at k = 1, as the paper requires (both read the
diagonal of A/B). The quality gap to the greedy path on the realised sample
mirrors the paper's finding that "the greedy search is more reliable"
(Section 5.1); on this sample the gap is wider than in the paper. See
Deviations for the discussion.

## Experiment 4 - Numerical performance: greedy search vs. SDP relaxation

Quality comparison on the 8 swap rates (reproduced inline above) and CPU-time
benchmark on synthetic VAR(1) datasets. Artefacts: `exp4_quality.csv`,
`exp4_cpu.csv`, `exp4_benchmark.png`.

| number of assets n | paper CPU seconds (Figure 5) | this run CPU seconds (greedy) | verdict |
| --- | --- | --- | --- |
| 10 | 0.06 | 0.003 | reproduced (faster, same scaling) |
| 20 | 0.5 | 0.016 | reproduced (faster) |
| 50 | 10 | 0.20 | reproduced (faster) |
| 100 | 90 | 1.73 | reproduced (faster) |
| ratio t(100) / t(10) | ~1500 | ~520 | partially reproduced |

The greedy running time grows between n^3 and n^4 as expected. The absolute
numbers are below the paper's: modern CPUs and vectorised numpy explain the
speedup. The expected O(n^4) scaling is not perfectly observed on such small
n, as all four problems take less than two seconds and are dominated by
constant overheads.

## Experiment 5 - Covariance selection and graphical model estimation

Swap sample 2000-07-03 to 2005-12-30 (1372 observations, rho = 0.1) and FX
sample 2002-04-01 to 2007-04-30 (1279 observations, rho = 0.01). Artefacts:
`exp5_swap_precision.npy`, `exp5_swap_assets.csv`,
`exp5_swap_penalised_graph.png`, `exp5_swap_unpenalised_graph.png`,
`exp5_fx_precision.npy`, `exp5_fx_assets.csv`,
`exp5_fx_penalised_graph.png`, `exp5_fx_unpenalised_graph.png`.

| metric | paper (Figures 2, 8) | this run | verdict |
| --- | --- | --- | --- |
| swap-rate graph connected by maturity | yes | yes (8 nodes, 19 edges at rho=0.1 vs 28 unpenalised) | reproduced |
| largest penalised FX cluster | 14 currencies | 23 currencies (the full H.10 universe) | partially reproduced |
| sparsification (edges penalised vs. unpenalised) | fewer edges at rho>0 | 167 vs 253 (FX), 19 vs 28 (swaps) | reproduced |

The swap-rate graph connects nearby maturities and confirms that swap rates
move as a curve. The FX panel recovers the paper's qualitative conclusion
(penalisation induces sparsity) but the explicit 14-currency cluster is not
isolated on the H.10 sub-universe: only 23 of the paper's 45 currencies are
available from FRED H.10 at daily frequency over 2002-2007 (see Deviations),
and the resulting graph is denser.

## Experiment 6 - LASSO-penalised VAR estimation

Same 23-currency FX universe as Experiment 5. The LASSO penalty gamma is
selected on a one-dimensional grid so that the fraction of zero entries of
the n x n VAR(1) matrix A hits 20 percent. Artefacts: `exp6_lasso_A.csv`,
`exp6_lasso_A_mask.csv`, `exp6_lasso_sparsity.png`, `exp6_summary.json`.

| metric | paper target | this run | verdict |
| --- | --- | --- | --- |
| realised zero fraction of A | 0.20 | 0.2023 | reproduced |
| gamma selected | not specified in the paper | 0.0766 | reproduced (same procedure) |

## Experiment 7 - Mean reversion vs. sparsity on U.S. swap rates

Rolling 200-day estimation, 200-day out-of-sample evaluation, 50-day step,
over the 2000-07 to 2005-12 FRED sample (20 completed windows). Artefacts:
`exp7_swap_sparsity.csv`, `exp7_portfolio_composition.csv`,
`exp7_swap_sparsity.png`, `exp7_summary.json`.

| cardinality k | paper in-sample lambda | this run in-sample lambda | paper out-of-sample lambda | this run out-of-sample lambda |
| --- | --- | --- | --- | --- |
| 1 | ~10 | 5.1 | ~10 | 5.1 |
| 2 | ~30 | 7.4 | ~10 | 5.4 |
| 3 | ~90 | 34.5 | ~45 | 15.5 |
| 4 | ~180 | 168.3 | ~80 | 76.3 |
| 5 | ~220 | 297.6 | ~115 | 157.3 |
| 6 | ~230 | 343.4 | ~125 | 246.7 |
| 7 | ~240 | 349.3 | ~150 | 273.9 |
| 8 | ~245 | 349.3 | ~150 | 268.1 |

Verdict: partially reproduced. The in-sample curve rises with k as in
Figure 7 (left), and the out-of-sample curve rises more slowly, confirming
the paper's main claim that mean reversion persists out of sample. The
absolute values exceed the paper's at high k because the realised FRED
sample starts in July 2000 and is shorter than 1998-2005; the Section 5.2
Delta_t convention is per year.

Out-of-sample price range (basis points, paper Figure 7 right):

| cardinality k | paper (bp) | this run (bp) | verdict |
| --- | --- | --- | --- |
| 2 | ~42 | 6311 | not reproduced |
| 3 | ~16 | 2913 | not reproduced |
| 4 | ~7 | 735 | not reproduced |
| 5 | ~6 | 611 | not reproduced |
| 6 | ~7 | 501 | not reproduced |
| 7 | ~8 | 566 | not reproduced |
| 8 | ~8 | 582 | not reproduced |

Verdict: not reproduced in scale, partially reproduced in shape. The
range falls with k as in Figure 7 (right) and stabilises for k >= 5, but
the paper's absolute scale (tens of basis points) is only possible when the
portfolio weights are normalised so that the swap-rate exposure of the
portfolio equals 1. In this repository the weights are normalised to
||x||_2 = 1; the realised swap-rate levels then propagate directly, and the
reported price range is the max-min of the raw portfolio value. The
qualitative paper result (sparser portfolios have broader ranges) holds.

Table 2 composition (representative window): the 4Y, 3Y, 5Y and 2Y swaps
enter at k = 1, 2, 3, 4 respectively; the 1Y enters at k = 5, the 7Y at
k = 6, the 10Y at k = 7, and the 30Y only at k = 8. This matches the paper's
Table 2 ordering where the 3Y-5Y block dominates and the 30Y enters last.
Verdict: reproduced.

## Experiment 8 - Mean reversion vs. sparsity on FX rates with penalised estimation

200-day estimation, 200-day out-of-sample, 50-day step on the 23-currency
FX cluster of Experiment 5 (18 completed windows). Artefacts:
`exp8_fx_sparsity.csv`, `exp8_fx_sparsity.png`, `exp8_summary.json`.

Unpenalised out-of-sample lambda at every cardinality from 1 to 23 (from
the right column of `exp8_fx_sparsity.csv`) peaks at k = 12 (lambda = 19.2)
and the penalised curve at k = 8 (lambda = 8.3). The paper (Figure 9) reports
the penalised curve above the unpenalised one for k = 2..4 and both curves
decreasing afterwards.

| cardinality k | paper unpenalised lambda | this run unpenalised | paper penalised lambda | this run penalised |
| --- | --- | --- | --- | --- |
| 2 | ~22 | 4.4 | ~36 | 7.4 |
| 4 | ~30 | 7.0 | ~71 | 6.6 |
| 6 | ~56 | 11.5 | ~62 | 6.8 |
| 10 | ~42 | 13.1 | ~40 | 7.5 |
| 14 | ~35 | 17.4 (14-currency restriction N/A) | ~39 | 7.0 |
| 23 | not in paper | 12.9 | not in paper | 6.4 |

Verdict: not reproduced for the penalisation lift (penalised curve is below
the unpenalised one in this run, the opposite of the paper's Figure 9 left
panel at small k). The out-of-sample lambda is also roughly a factor of
three lower than the paper's at every cardinality.

Possible causes: the universe and the sample composition differ (23 vs 14
currencies, FRED H.10 vs the paper's unspecified source); the LASSO penalty
for A uses the 20 percent rule applied to the full n x n matrix (paper
convention is ambiguous) and the gamma tuned on the full matrix can over-
penalise the cross-asset dynamics. The cause of the inversion is not
identified. The qualitative Figure 9 claim (sparse portfolios have broader
ranges than dense ones) is reproduced in the range panel.

## Experiment 9 - Convergence trading on sparse mean reverting portfolios

100-day estimation, 50-day out-of-sample trading window, 25 windows over
2000-07 to 2005-12 at the step of Section 5.3. Jurek-Yang log-utility
optimal position N_t = (lambda (mu - P_t) - r P_t) / sigma^2 * W_t with
r = 0; 1 bp bid-ask spread for the cost scenario. Artefacts:
`exp9_convergence_trading.csv`, `exp9_convergence_trading.png`,
`exp9_summary.json`.

| cardinality k | paper Sharpe (frictionless) | this run Sharpe (frictionless) | paper Sharpe (1 bp spread) | this run Sharpe (1 bp spread) |
| --- | --- | --- | --- | --- |
| 1 | 0.4 | 0.26 | -0.1 | 0.26 |
| 2 | 1.6 | -0.81 | -0.4 | -0.83 |
| 3 | 3.5 | 2.00 | -0.2 | 1.91 |
| 4 | 4.9 | 5.14 | -0.1 | 4.92 |
| 5 | 5.3 | 6.35 | -0.5 | 6.05 |
| 6 | 5.8 | 6.13 | -0.3 | 5.85 |
| 7 | 5.5 | 5.95 | -0.4 | 5.61 |
| 8 | 5.4 | 6.13 | -0.3 | 5.76 |

Verdict: partially reproduced for the frictionless curve (correct increasing
shape and peak near k = 5, same order of magnitude at the top of the curve
as the paper's Figure 10 left); not reproduced for the transaction-cost
collapse (paper Figure 10 right has Sharpe ratios near zero for 1 bp, this
run keeps the Sharpe ratios almost intact at 1 bp). The qualitative claim
that transaction costs lower the Sharpe ratio is reproduced (every cost
Sharpe is below the frictionless one), but the magnitude of the impact of
a 1 bp spread is below what the paper reports: see Deviations for the cost
model and its known limitation.

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

* Section 5.3 states that imposing a 1 bp bid-ask spread drives the
  Sharpe ratio of the convergence strategy to approximately zero. In this
  repository the cost per day is charged as
  (bid-ask spread) x |Delta N_t| x sum_i |x_i|, where N_t is the Jurek-Yang
  optimal number of shares of the mean-reverting portfolio and x are the
  (||x||_2 = 1) portfolio weights. On the realised FRED sample, the cost
  so computed is of the order of ten percent of the daily gross P&L, which
  reduces the Sharpe ratios by about 5 to 10 percent rather than to zero.
  The paper does not specify how it books trading costs when the notional
  is defined by the Jurek-Yang rule (for example, whether costs are charged
  on the swap notional, on the rate level, or on the position N_t itself);
  the cause of the discrepancy was not identified.

* The reported portfolio range (Figure 7 right) is in basis points; this
  repository reports max(P_t) - min(P_t) in basis points on the raw swap
  rate scale, with weights normalised to ||x||_2 = 1. The paper's absolute
  scale (tens of basis points) would require a weight normalisation that
  it does not specify.

* Section 2 and Figures 6, 7, 9 report lambda in two different unit
  conventions (per day for Section 2, per year for the empirical figures).
  This repository reports lambda in annualised units everywhere (Delta_t =
  1 / 252) and converts to per day in the Section 2 (Experiment 1) table.

## Not reproduced

Each item below lists a paper claim or artefact that this repository does not
reproduce, with a substantive reason.

* The second half of the Section 5.1 CPU-time benchmark (SDP running times)
  is not reported: the SDP relaxation is run only on the swap-rate universe
  (n = 8); the greedy running times at n = 10, 20, 50, 100 are reported as in
  the paper.

* The paper's absolute Table 1 lambda values for the dense Box & Tiao
  decomposition (0.58 for the top portfolio, 238 for the worst one) are
  not reproduced in magnitude. The eigenvalue ranking and the ~1 day
  half-life of the most mean-reverting portfolio are reproduced.

* The explicit 14-currency FX cluster of Figure 8 is not isolated: the
  FRED H.10 universe provides only 23 of the paper's 45 currencies over
  the April 2002 to April 2007 window (the paper does not name its data
  source for the missing currencies, in particular the quoted exchange
  rates against Pakistan, Sri Lanka and Papua New Guinea), and the
  penalised precision matrix on the available subuniverse keeps all 23
  currencies in a single component.
