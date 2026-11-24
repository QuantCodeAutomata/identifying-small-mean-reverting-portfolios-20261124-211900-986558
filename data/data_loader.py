"""Data access for the mean-reverting-portfolios paper replication.

The paper studies two markets:
  * U.S. constant-maturity swap rates (eight maturities, 1998-2005, daily),
  * U.S. dollar exchange rates against up to 45 currencies (April 2002 to
    April 2007, daily).

Both are macro daily series that live on the Federal Reserve Economic Data
(FRED) service, and are fetched via :mod:`pandas_datareader`.
Downloaded series are cached in ``data/cache/`` (gitignored).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from pandas_datareader import data as pdr

from src.config import (
    FX_END,
    FX_INVERT_FRED,
    FX_START,
    FX_UNIVERSE_FRED,
    SWAP_END_FULL,
    SWAP_FRED_SERIES,
    SWAP_MATURITIES,
    SWAP_START_FULL,
)

CACHE_DIR = Path(__file__).resolve().parent / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class DataLoaderError(RuntimeError):
    """Raised when a data source cannot deliver the requested series."""


def _fetch_fred(series: str, start: str, end: str) -> pd.Series:
    """Return a FRED series as a pandas Series indexed by business day."""
    cache = CACHE_DIR / f"fred_{series}.csv"
    if cache.exists():
        data = pd.read_csv(cache, index_col=0, parse_dates=True).iloc[:, 0]
        data.name = series
        return data.loc[(data.index >= pd.Timestamp(start))
                        & (data.index <= pd.Timestamp(end))]
    try:
        data = pdr.DataReader(series, "fred", start, end)
    except Exception as exc:  # pragma: no cover - network path
        raise DataLoaderError(f"FRED fetch failed for {series}: {exc}") from exc
    if data.empty:
        raise DataLoaderError(f"FRED returned no data for {series}.")
    data = data.iloc[:, 0]
    data.name = series
    data.to_csv(cache)
    return data


def load_swap_rates(
    start: str = SWAP_START_FULL,
    end: str = SWAP_END_FULL,
    maturities: tuple = SWAP_MATURITIES,
) -> pd.DataFrame:
    """Load U.S. constant-maturity swap rates from FRED (daily, in percent).

    The FRED constant-maturity swap rate series start on 2000-07-03, so the
    paper's "1998 to 2005" window is truncated to the maximum available range
    beginning on that date. This truncation is reported in RESULTS.md.
    """
    columns: dict[str, pd.Series] = {}
    for maturity in maturities:
        if maturity not in SWAP_FRED_SERIES:
            raise DataLoaderError(f"No FRED code configured for maturity {maturity}.")
        columns[maturity] = _fetch_fred(SWAP_FRED_SERIES[maturity], start, end)
    frame = pd.DataFrame(columns)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="any")
    if frame.empty:
        raise DataLoaderError("No overlapping observations for the swap maturities.")
    frame.index.name = "date"
    return frame


def load_fx_rates(
    start: str = FX_START,
    end: str = FX_END,
    universe: dict | None = None,
) -> pd.DataFrame:
    """Load U.S. dollar exchange rates (daily, FX per USD) from FRED.

    FRED quotes some rates as USD per foreign currency (euro, sterling, aussie,
    kiwi); these are inverted so every column is "foreign currency per USD".
    Currencies whose FRED series return no overlap in the requested window are
    silently dropped and reported in :attr:`DataFrame.attrs['dropped']`.
    """
    universe = universe or FX_UNIVERSE_FRED
    columns: dict[str, pd.Series] = {}
    dropped: list[str] = []
    for country, series in universe.items():
        try:
            data = _fetch_fred(series, start, end)
        except DataLoaderError:
            dropped.append(country)
            continue
        if FX_INVERT_FRED.get(country, False):
            data = 1.0 / data
        columns[country] = data
    if not columns:
        raise DataLoaderError("None of the requested FX series could be loaded.")
    frame = pd.DataFrame(columns)
    frame = frame.apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(how="any")
    if frame.empty:
        raise DataLoaderError("No overlapping observations across FX series.")
    frame.index.name = "date"
    frame.attrs["dropped"] = dropped
    return frame


def to_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return the dataframe values as a float ndarray, preserving column order."""
    return np.asarray(frame.values, dtype=float)
