"""Configuration for the replication of the sparse mean reverting portfolios paper.

All experiment parameters live here as module-level UPPER_CASE variables.
Each variable states its unit and the paper reference for its value.
"""

from __future__ import annotations

import os

# --- Shared constants ------------------------------------------------------

DELTA_T: float = 1.0 / 252.0  # year fraction of one trading day; standard annualization
TRADING_DAYS: int = 252  # trading days per year

# --- U.S. swap rate universe (paper Section 2 / Table 1, Figure 1) ---------
# Maturities used by the paper's illustrative canonical decomposition.
SWAP_MATURITIES: tuple[str, ...] = ("1Y", "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "30Y")
# FRED H.15 series ids (constant-maturity swap rates, daily, in percent).
SWAP_FRED_SERIES: dict[str, str] = {
    "1Y": "DSWP1",
    "2Y": "DSWP2",
    "3Y": "DSWP3",
    "4Y": "DSWP4",
    "5Y": "DSWP5",
    "7Y": "DSWP7",
    "10Y": "DSWP10",
    "30Y": "DSWP30",
}

# Full empirical sample window (paper page 14 and page 16: "1998 until 2005").
SWAP_START_FULL: str = "1998-01-01"
SWAP_END_FULL: str = "2005-12-31"

# --- Illustrative decomposition window (Figure 1, Figure 6, Figure 5) ------
# Paper only says "100 days of U.S. swap rate data" (page 5 and page 14); the
# exact window is not specified, so we take the final 100 trading days of the
# full sample so the results are reproducible.
ILLUSTRATIVE_N_DAYS: int = 100

# --- Experiment 1 (Box & Tiao decomposition, Table 1) ----------------------
EXP1_N_PORTFOLIOS: int = 8

# --- Experiment 2 (greedy search, Figures 5 and 6) -------------------------
EXP2_FORWARD_BACKWARD: bool = False  # paper mentions refinement; does not report using it

# --- Experiment 3 (SDP relaxation, Figure 5) -------------------------------
EXP3_SOLVER: str = "SCS"  # cvxpy solver; SCS handles the L1-constrained SDP
EXP3_RANK_TOL: float = 1e-4  # relative tolerance when recovering x from Y

# --- Experiment 4 (numerical benchmark, Figure 5 right) --------------------
EXP4_UNIVERSE_SIZES: tuple[int, ...] = (10, 20, 50, 100)
EXP4_SYNTHETIC_N_OBS: int = 100  # observations per synthetic sample
EXP4_SYNTHETIC_SEED: int = 7  # not specified by the paper

# --- Experiment 5 (covariance selection, Figures 2 and 8) ------------------
COV_SELECT_RHO_SWAP: float = 0.1  # paper page 10
COV_SELECT_RHO_FX: float = 0.01  # paper page 16

# U.S. dollar exchange rate universe (paper page 16). FRED is the daily
# source for the sample dates; the paper lists 45 currencies but historical
# FRED coverage is limited to the subset below.
FX_UNIVERSE_FRED: dict[str, str] = {
    "Australia": "DEXUSAL",
    "Brazil": "DEXBZUS",
    "Canada": "DEXCAUS",
    "China": "DEXCHUS",
    "Denmark": "DEXDNUS",
    "Eurozone": "DEXUSEU",
    "Hong Kong": "DEXHKUS",
    "India": "DEXINUS",
    "Japan": "DEXJPUS",
    "Malaysia": "DEXMAUS",
    "Mexico": "DEXMXUS",
    "New Zealand": "DEXUSNZ",
    "Norway": "DEXNOUS",
    "Singapore": "DEXSIUS",
    "South Africa": "DEXSFUS",
    "South Korea": "DEXKOUS",
    "Sri Lanka": "DEXSLUS",
    "Sweden": "DEXSDUS",
    "Switzerland": "DEXSZUS",
    "Taiwan": "DEXTAUS",
    "Thailand": "DEXTHUS",
    "United Kingdom": "DEXUSUK",
    "Venezuela": "DEXVZUS",
}
# FRED quotes some rates as USD per FX; True means invert to FX per USD.
FX_INVERT_FRED: dict[str, bool] = {
    "Australia": True,
    "Eurozone": True,
    "New Zealand": True,
    "United Kingdom": True,
}

FX_START: str = "2002-04-01"  # paper page 16: "from April 2002 until April 2007"
FX_END: str = "2007-04-30"

# The 14 currencies the paper's covariance-selection cluster isolates (Figure 8).
FX_CLUSTER_PAPER: tuple[str, ...] = (
    "Russia", "Thailand", "Pakistan", "Sri Lanka", "Czech Republic", "Taiwan",
    "India", "Hungary", "Morocco", "Japan", "Philippines", "South Africa",
    "Finland", "Norway",
)

# --- Experiment 6 (LASSO VAR, Section 4.2) ---------------------------------
LASSO_VAR_ZERO_FRAC: float = 0.20  # paper page 16

# --- Experiment 7 (swap rates: mean reversion vs. sparsity, Figure 7) ------
EXP7_EST_WINDOW: int = 200  # paper page 16
EXP7_OOS_WINDOW: int = 200  # paper page 16
EXP7_STEP: int = 50  # paper page 16

# --- Experiment 8 (FX rates with penalized estimation, Figure 9) -----------
# The paper does not state the FX rolling-window parameters; we reuse the
# swap-rate scheme of Section 5.2 so both experiments share one rolling path.
EXP8_EST_WINDOW: int = 200
EXP8_OOS_WINDOW: int = 200
EXP8_STEP: int = 50

# --- Experiment 9 (convergence trading, Figure 10) -------------------------
EXP9_EST_WINDOW: int = 100  # paper page 17
EXP9_OOS_WINDOW: int = 50  # paper page 17
EXP9_STEP: int = 50  # not specified
EXP9_RISK_FREE_RATE: float = 0.0  # frictionless case; paper does not fix r
EXP9_BID_ASK_BP: float = 1.0  # paper page 17: "bid-ask spread of 1bp"

# --- Output directory ------------------------------------------------------
RESULTS_DIR: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)
