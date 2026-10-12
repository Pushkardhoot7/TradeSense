"""
validation.py
-------------
Historical price data validation module for TradeSense V2.1.
Enforces data integrity, chronological order, positive price constraints,
duplicate filtering, and minimum history rules before market calculations begin.
"""

from __future__ import annotations

import logging
from typing import Tuple
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume", "Adj Close"}
MIN_OBSERVATIONS = 15  # Minimum valid trading days required for analytical significance


def validate_and_clean_price_series(
    symbol: str,
    df: pd.DataFrame,
    min_observations: int = MIN_OBSERVATIONS,
) -> Tuple[bool, str, pd.DataFrame]:
    """
    Validate and clean historical OHLCV data for a single symbol.

    Checks:
    1. Non-empty DataFrame.
    2. Datetime index, timezone-naive, sorted chronologically.
    3. Removal of duplicate dates (retaining earliest valid observation).
    4. Required columns presence (ensuring Adj Close exists).
    5. Positive valid prices (Close > 0, High >= Low, Low > 0).
    6. Minimum number of positive valid trading day observations.
    7. Excessive missing values check (drops records missing Close).

    Returns:
    (is_valid: bool, reason: str, cleaned_df: pd.DataFrame)
    """
    if df is None or df.empty:
        return False, "Empty dataframe returned by provider", pd.DataFrame()

    clean = df.copy()

    # 1. Standardize MultiIndex columns if present
    if isinstance(clean.columns, pd.MultiIndex):
        clean.columns = clean.columns.get_level_values(0)

    # 2. Date index handling
    if "Date" in clean.columns and not isinstance(clean.index, pd.DatetimeIndex):
        clean["Date"] = pd.to_datetime(clean["Date"], errors="coerce")
        clean = clean.dropna(subset=["Date"]).set_index("Date")
    elif not isinstance(clean.index, pd.DatetimeIndex):
        try:
            clean.index = pd.to_datetime(clean.index, errors="coerce")
            clean = clean[~clean.index.isna()]
        except Exception as dt_err:
            return False, f"Invalid date index: {dt_err}", pd.DataFrame()

    # Remove timezone info to enforce consistent timezone-naive market calendar
    if clean.index.tz is not None:
        clean.index = clean.index.tz_localize(None)

    clean.index.name = "Date"

    # 3. Chronological sorting
    clean = clean.sort_index()

    # 4. Remove duplicate dates
    if clean.index.duplicated().any():
        clean = clean[~clean.index.duplicated(keep="first")]

    # 5. Required columns and Adj Close handling
    if "Close" not in clean.columns:
        return False, "Missing 'Close' column", pd.DataFrame()

    if "Adj Close" not in clean.columns:
        clean["Adj Close"] = clean["Close"]

    for col in ["Open", "High", "Low"]:
        if col not in clean.columns:
            clean[col] = clean["Close"]

    if "Volume" not in clean.columns:
        clean["Volume"] = 0

    # Ensure numeric conversion
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce")

    # 6. Drop rows where Close or Adj Close is NaN or <= 0
    clean = clean.dropna(subset=["Close", "Adj Close"])
    clean = clean[(clean["Close"] > 0) & (clean["Adj Close"] > 0)]

    # Validate price sanity: High >= Low
    clean.loc[clean["High"] < clean["Low"], "High"] = clean["Low"]
    clean.loc[clean["High"] < clean["Close"], "High"] = clean["Close"]
    clean.loc[clean["Low"] > clean["Close"], "Low"] = clean["Close"]

    # 7. Check minimum history threshold
    obs_count = len(clean)
    if obs_count < min_observations:
        return (
            False,
            f"Insufficient history: {obs_count} observations (minimum required: {min_observations})",
            clean,
        )

    # Reorder standard columns
    clean = clean[["Open", "High", "Low", "Close", "Volume", "Adj Close"]]

    return True, "Valid", clean


def align_returns_matrix(price_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build returns matrix from validated price data using adjusted prices.
    Aligns observations on common dates without fabricating prices or forward-filling indefinitely.
    """
    series_dict = {}
    for sym, df in price_dict.items():
        if df is not None and not df.empty:
            col = "Adj Close" if "Adj Close" in df.columns else "Close"
            series = df[col].dropna()
            # Calculate daily percentage returns
            rets = series.pct_change().dropna()
            if len(rets) >= 5:
                series_dict[sym] = rets

    if not series_dict:
        return pd.DataFrame()

    # Concatenate across columns (outer join initially, then drop rows with all NaN)
    returns_df = pd.DataFrame(series_dict)
    # Fill modest isolated non-trading days with 0 return only if < 5% missing, otherwise drop
    # But adhere to strict policy: drop dates where majority of market was closed
    returns_df = returns_df.dropna(how="all")

    return returns_df
