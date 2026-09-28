"""
Data Loader Module for TradeSense

Handles downloading, cleaning, caching, date alignment, and demo-mode fallback for historical stock price data.

Data Source Attribution:
"Yahoo Finance (via yfinance) — unofficial, third-party data"
"""

import os
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Tuple
import numpy as np
import pandas as pd
import yfinance as yf

# Global status tracking metadata registry
_STATUS_REGISTRY: Dict[str, Dict[str, Any]] = {}
CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "cache"
)
os.makedirs(CACHE_DIR, exist_ok=True)

DATA_SOURCE_LIVE = "Yahoo Finance (via yfinance) — unofficial, third-party data"
DATA_SOURCE_DEMO = "DEMO MODE — Sample Data"


def download_stock_data(symbol: str, period: str = "1y", interval: str = "1d") -> Optional[pd.DataFrame]:
    """
    Fetch raw historical data from yfinance for a single symbol.
    Returns None on failure (does not raise exceptions).
    """
    if not symbol:
        return None

    try:
        ticker_obj = yf.Ticker(symbol)
        df = ticker_obj.history(period=period, interval=interval, auto_adjust=False)

        if df is None or df.empty:
            return None

        return df
    except Exception as e:
        print(f"yfinance download failed for {symbol}: {e}")
        return None


def clean_stock_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply missing-value, duplicate, and invalid-row handling:
    1. Ensure DatetimeIndex and sort chronologically.
    2. Remove duplicate dates.
    3. Standardize column names (Open, High, Low, Close, Adj Close, Volume).
    4. Handle missing values via forward fill then backward fill.
    5. Derive consistent 'Price' column (Adj Close falling back to Close).
    """
    if df is None or df.empty:
        return pd.DataFrame()

    cleaned_df = df.copy()

    # Reset Index if Date is in columns
    if "Date" in cleaned_df.columns:
        cleaned_df["Date"] = pd.to_datetime(cleaned_df["Date"])
        cleaned_df.set_index("Date", inplace=True)

    if not isinstance(cleaned_df.index, pd.DatetimeIndex):
        try:
            cleaned_df.index = pd.to_datetime(cleaned_df.index)
        except Exception:
            return pd.DataFrame()

    # Remove timezone info if present
    if cleaned_df.index.tz is not None:
        cleaned_df.index = cleaned_df.index.tz_localize(None)

    # Sort index & remove duplicate dates
    cleaned_df = cleaned_df.sort_index()
    cleaned_df = cleaned_df[~cleaned_df.index.duplicated(keep="first")]

    # Normalize column names
    col_rename = {}
    for col in cleaned_df.columns:
        c_str = str(col).strip()
        if c_str.lower() == "adj close" or c_str.lower() == "adjclose":
            col_rename[col] = "Adj Close"
        elif c_str.lower() == "close":
            col_rename[col] = "Close"
        elif c_str.lower() == "open":
            col_rename[col] = "Open"
        elif c_str.lower() == "high":
            col_rename[col] = "High"
        elif c_str.lower() == "low":
            col_rename[col] = "Low"
        elif c_str.lower() == "volume":
            col_rename[col] = "Volume"

    cleaned_df.rename(columns=col_rename, inplace=True)

    # Ensure required columns exist
    if "Close" not in cleaned_df.columns:
        return pd.DataFrame()

    if "Adj Close" not in cleaned_df.columns:
        cleaned_df["Adj Close"] = cleaned_df["Close"]

    # Fill missing values
    cleaned_df = cleaned_df.ffill().bfill()

    # Define consistent price column (Adj Close falling back to Close)
    cleaned_df["Price"] = np.where(
        cleaned_df["Adj Close"].notna() & (cleaned_df["Adj Close"] > 0),
        cleaned_df["Adj Close"],
        cleaned_df["Close"],
    )

    # Filter non-positive invalid prices
    cleaned_df = cleaned_df[cleaned_df["Price"] > 0]

    return cleaned_df


def _generate_demo_data(symbol: str, period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Generate reproducible sample time-series data using a fixed seed.
    """
    seed_val = int(hashlib.md5(symbol.encode("utf-8")).hexdigest()[:8], 16) % 100000
    np.random.seed(seed_val)

    periods_map = {"3m": 63, "6m": 126, "1y": 252, "2y": 504, "5y": 1260}
    num_days = periods_map.get(period.lower(), 252)

    end_date = pd.Timestamp.now().normalize()
    date_range = pd.bdate_range(end=end_date, periods=num_days)

    returns = np.random.normal(loc=0.0005, scale=0.015, size=num_days)
    price_series = 100.0 * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "Open": price_series * (1 + np.random.uniform(-0.005, 0.005, num_days)),
            "High": price_series * (1 + np.random.uniform(0.001, 0.015, num_days)),
            "Low": price_series * (1 - np.random.uniform(0.001, 0.015, num_days)),
            "Close": price_series,
            "Adj Close": price_series,
            "Price": price_series,
            "Volume": np.random.randint(100000, 5000000, size=num_days),
        },
        index=date_range,
    )
    df.index.name = "Date"
    return df


def get_historical_data(symbol: str, period: str = "1y", interval: str = "1d", allow_demo: bool = True) -> Dict[str, Any]:
    """
    High-level entry point: returns cleaned, cached (or freshly downloaded)
    historical data for a symbol.
    """
    if not symbol:
        res = {
            "symbol": "UNKNOWN",
            "data": pd.DataFrame(),
            "status": "failed",
            "is_demo": False,
            "source": DATA_SOURCE_LIVE,
            "rows": 0,
            "start_date": "N/A",
            "end_date": "N/A",
            "error_message": "Symbol is empty",
        }
        _STATUS_REGISTRY["UNKNOWN"] = res
        return res

    cache_filename = f"stock_{symbol.replace('.', '_')}_{period}_{interval}.csv"
    cache_path = os.path.join(CACHE_DIR, cache_filename)

    # Check on-disk cache
    if os.path.exists(cache_path):
        try:
            cached_df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            cleaned_cached = clean_stock_data(cached_df)
            if not cleaned_cached.empty and len(cleaned_cached) >= 10:
                res = {
                    "symbol": symbol,
                    "data": cleaned_cached,
                    "status": "ok",
                    "is_demo": False,
                    "source": DATA_SOURCE_LIVE,
                    "rows": len(cleaned_cached),
                    "start_date": str(cleaned_cached.index[0].strftime("%Y-%m-%d")),
                    "end_date": str(cleaned_cached.index[-1].strftime("%Y-%m-%d")),
                    "error_message": None,
                }
                _STATUS_REGISTRY[symbol] = res
                return res
        except Exception:
            pass

    # Fetch live data
    raw_df = download_stock_data(symbol, period=period, interval=interval)
    cleaned_df = clean_stock_data(raw_df) if raw_df is not None else pd.DataFrame()

    if not cleaned_df.empty and len(cleaned_df) >= 10:
        try:
            cleaned_df.to_csv(cache_path)
        except Exception:
            pass

        res = {
            "symbol": symbol,
            "data": cleaned_df,
            "status": "ok",
            "is_demo": False,
            "source": DATA_SOURCE_LIVE,
            "rows": len(cleaned_df),
            "start_date": str(cleaned_df.index[0].strftime("%Y-%m-%d")),
            "end_date": str(cleaned_df.index[-1].strftime("%Y-%m-%d")),
            "error_message": None,
        }
        _STATUS_REGISTRY[symbol] = res
        return res

    # If demo mode is allowed, fall back to demo series
    if allow_demo:
        demo_df = _generate_demo_data(symbol, period=period, interval=interval)
        res = {
            "symbol": symbol,
            "data": demo_df,
            "status": "demo",
            "is_demo": True,
            "source": DATA_SOURCE_DEMO,
            "rows": len(demo_df),
            "start_date": str(demo_df.index[0].strftime("%Y-%m-%d")),
            "end_date": str(demo_df.index[-1].strftime("%Y-%m-%d")),
            "error_message": "Live market data unavailable — DEMO MODE activated",
        }
        _STATUS_REGISTRY[symbol] = res
        return res

    # Failure response when live retrieval fails and demo is disallowed
    res = {
        "symbol": symbol,
        "data": pd.DataFrame(),
        "status": "failed",
        "is_demo": False,
        "source": DATA_SOURCE_LIVE,
        "rows": 0,
        "start_date": "N/A",
        "end_date": "N/A",
        "error_message": f"Unable to retrieve market data for {symbol} from Yahoo Finance.",
    }
    _STATUS_REGISTRY[symbol] = res
    return res


def align_stock_data(data_by_symbol: Dict[str, Union[Dict[str, Any], pd.DataFrame, pd.Series]]) -> Dict[str, Any]:
    """
    Align date indices across multiple stocks for consistent cross-stock comparison.
    """
    dfs = {}
    for sym, obj in data_by_symbol.items():
        if isinstance(obj, dict) and "data" in obj:
            df = obj["data"]
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            df = obj
        else:
            continue

        if df is not None and not df.empty:
            dfs[sym] = df

    if not dfs:
        return {}

    common_idx = None
    for df in dfs.values():
        if common_idx is None:
            common_idx = df.index
        else:
            common_idx = common_idx.intersection(df.index)

    if common_idx is None or len(common_idx) == 0:
        all_indices = [df.index for df in dfs.values()]
        common_idx = pd.DatetimeIndex(sorted(list(set().union(*all_indices))))

    aligned = {}
    for sym, df in dfs.items():
        aligned_df = df.reindex(common_idx).ffill().bfill()
        aligned[sym] = aligned_df

    return aligned


def get_data_status(symbol: str) -> Dict[str, Any]:
    if symbol in _STATUS_REGISTRY:
        return _STATUS_REGISTRY[symbol]

    return {
        "symbol": symbol,
        "status": "failed",
        "is_demo": False,
        "source": DATA_SOURCE_LIVE,
        "rows": 0,
        "start_date": "N/A",
        "end_date": "N/A",
        "error_message": "Data not retrieved",
    }


def fetch_aligned_close_prices(
    tickers: List[str], period: str = "1y", interval: str = "1d", allow_demo: bool = True
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Helper for dashboard to load, clean, and align Close prices across selected tickers.
    """
    data_map = {}
    status_list = []

    for t in tickers:
        res = get_historical_data(t, period=period, interval=interval, allow_demo=allow_demo)
        status_list.append(res)
        if res["status"] in ["ok", "demo"] and not res["data"].empty:
            data_map[t] = res["data"]["Price"] if "Price" in res["data"].columns else res["data"]["Close"]

    if not data_map:
        return pd.DataFrame(), status_list

    aligned_map = align_stock_data(data_map)
    combined_df = pd.DataFrame(aligned_map)
    return combined_df, status_list
