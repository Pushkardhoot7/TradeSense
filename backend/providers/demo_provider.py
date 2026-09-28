"""
demo_provider.py
----------------
Synthetic market data provider for TradeSense V2.

Generates deterministic, reproducible OHLCV price histories using
Geometric Brownian Motion (GBM) — no network calls, no yfinance.
Ideal for offline demos, unit tests, and CI environments.

Data mode label: DEMO DATA
"""

from __future__ import annotations

import datetime
import math
from typing import Any

import numpy as np
import pandas as pd

from .base_provider import MarketDataProvider


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS: int = 252  # synthetic history length

# 12 hardcoded NSE tickers with approximate base prices (INR)
_TICKERS: dict[str, float] = {
    "TCS.NS":        3500.0,
    "INFY.NS":       1500.0,
    "HDFCBANK.NS":   1600.0,
    "RELIANCE.NS":   2800.0,
    "WIPRO.NS":       450.0,
    "HCLTECH.NS":    1200.0,
    "ICICIBANK.NS":  1000.0,
    "KOTAKBANK.NS":  1750.0,
    "LTIM.NS":       5000.0,
    "AXISBANK.NS":   1100.0,
    "TATAMOTORS.NS":  900.0,
    "BAJFINANCE.NS": 7000.0,
}

# GBM parameters per ticker — (mu_annual, sigma_annual)
_GBM_PARAMS: dict[str, tuple[float, float]] = {
    "TCS.NS":        (0.12, 0.22),
    "INFY.NS":       (0.10, 0.24),
    "HDFCBANK.NS":   (0.14, 0.20),
    "RELIANCE.NS":   (0.16, 0.25),
    "WIPRO.NS":       (0.09, 0.26),
    "HCLTECH.NS":    (0.13, 0.23),
    "ICICIBANK.NS":  (0.15, 0.21),
    "KOTAKBANK.NS":  (0.11, 0.20),
    "LTIM.NS":       (0.18, 0.28),
    "AXISBANK.NS":   (0.14, 0.22),
    "TATAMOTORS.NS": (0.20, 0.32),
    "BAJFINANCE.NS": (0.17, 0.27),
}

# Company names for symbol search
_NAMES: dict[str, str] = {
    "TCS.NS":        "Tata Consultancy Services Ltd.",
    "INFY.NS":       "Infosys Ltd.",
    "HDFCBANK.NS":   "HDFC Bank Ltd.",
    "RELIANCE.NS":   "Reliance Industries Ltd.",
    "WIPRO.NS":      "Wipro Ltd.",
    "HCLTECH.NS":    "HCL Technologies Ltd.",
    "ICICIBANK.NS":  "ICICI Bank Ltd.",
    "KOTAKBANK.NS":  "Kotak Mahindra Bank Ltd.",
    "LTIM.NS":       "LTIMindtree Ltd.",
    "AXISBANK.NS":   "Axis Bank Ltd.",
    "TATAMOTORS.NS": "Tata Motors Ltd.",
    "BAJFINANCE.NS": "Bajaj Finance Ltd.",
}


# ---------------------------------------------------------------------------
# Helper: Geometric Brownian Motion price simulation
# ---------------------------------------------------------------------------

def _simulate_gbm(
    rng: np.random.Generator,
    s0: float,
    mu: float,
    sigma: float,
    n: int,
    dt: float = 1.0 / 252,
) -> np.ndarray:
    """Simulate a GBM path of length *n*.

    Parameters
    ----------
    rng:
        A seeded ``numpy.random.Generator`` for reproducibility.
    s0:
        Initial price.
    mu:
        Annualised drift (fraction, e.g. 0.12 for 12 % p.a.).
    sigma:
        Annualised volatility (fraction).
    n:
        Number of time steps (trading days).
    dt:
        Length of each step in years (default 1/252).

    Returns
    -------
    np.ndarray of shape (n,)
        Simulated closing prices.
    """
    z: np.ndarray = rng.standard_normal(n)
    # GBM increment: S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    increments: np.ndarray = np.exp(
        (mu - 0.5 * sigma ** 2) * dt + sigma * math.sqrt(dt) * z
    )
    path: np.ndarray = np.empty(n, dtype=float)
    path[0] = s0 * increments[0]
    for i in range(1, n):
        path[i] = path[i - 1] * increments[i]
    return path


# ---------------------------------------------------------------------------
# Helper: build a full OHLCV DataFrame for one ticker
# ---------------------------------------------------------------------------

def _build_ohlcv(
    rng: np.random.Generator,
    s0: float,
    mu: float,
    sigma: float,
) -> pd.DataFrame:
    """Build a TRADING_DAYS-row OHLCV DataFrame.

    Intraday noise is generated with a secondary small GBM to produce
    realistic High / Low / Open values around the Close.
    """
    close: np.ndarray = _simulate_gbm(rng, s0, mu, sigma, TRADING_DAYS)

    # Intraday range: fraction of close price drawn from a uniform dist
    intraday_range_pct: np.ndarray = rng.uniform(0.005, 0.025, TRADING_DAYS)
    half_range: np.ndarray = close * intraday_range_pct / 2.0

    high: np.ndarray  = close + half_range
    low: np.ndarray   = close - half_range
    # Open is prior close with small gap
    open_: np.ndarray = np.empty(TRADING_DAYS, dtype=float)
    open_[0] = s0
    open_[1:] = close[:-1] * (1.0 + rng.uniform(-0.005, 0.005, TRADING_DAYS - 1))
    # Volume: mean 1 M shares, lognormal
    volume: np.ndarray = (
        rng.lognormal(mean=13.8, sigma=0.5, size=TRADING_DAYS).astype(int)
    )

    # Build trading-day date range ending today
    end_date = datetime.date.today()
    dates: list[datetime.date] = []
    current = end_date - datetime.timedelta(days=1)
    while len(dates) < TRADING_DAYS:
        if current.weekday() < 5:  # Mon-Fri
            dates.append(current)
        current -= datetime.timedelta(days=1)
    dates.reverse()
    date_index = pd.to_datetime(dates)

    df = pd.DataFrame(
        {
            "Open":      open_,
            "High":      high,
            "Low":       low,
            "Close":     close,
            "Volume":    volume,
            "Adj Close": close,
        },
        index=date_index,
    )
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# Pre-build the synthetic dataset at import time (cheap, deterministic)
# ---------------------------------------------------------------------------

def _build_all_data() -> dict[str, pd.DataFrame]:
    """Build OHLCV DataFrames for every ticker using a shared market factor.

    Each ticker's daily return = beta * market_return + idiosyncratic_noise.
    This ensures realistic non-zero pairwise correlations (like actual NSE data)
    while remaining fully reproducible (seed=42).
    """
    rng = np.random.default_rng(seed=42)
    n   = TRADING_DAYS
    dt  = 1.0 / 252

    # Shared market factor — mimics Nifty 50 daily moves
    market_mu    = 0.12   # 12% annual drift
    market_sigma = 0.15   # 15% annual volatility
    market_daily = rng.normal(
        (market_mu - 0.5 * market_sigma ** 2) * dt,
        market_sigma * math.sqrt(dt),
        size=n,
    )

    # Sector betas — IT and Banking tend to correlate with market
    _BETAS: dict[str, float] = {
        "TCS.NS": 0.85, "INFY.NS": 0.90, "HDFCBANK.NS": 0.80,
        "RELIANCE.NS": 0.95, "WIPRO.NS": 0.88, "HCLTECH.NS": 0.87,
        "ICICIBANK.NS": 0.82, "KOTAKBANK.NS": 0.78, "LTIM.NS": 0.92,
        "AXISBANK.NS": 0.83, "TATAMOTORS.NS": 1.05, "BAJFINANCE.NS": 1.10,
    }

    result: dict[str, pd.DataFrame] = {}
    for ticker, s0 in _TICKERS.items():
        mu, sigma = _GBM_PARAMS[ticker]
        beta = _BETAS.get(ticker, 0.85)

        # Idiosyncratic component (~40% of total vol)
        idio_sigma = sigma * 0.40
        idio_daily = rng.normal(
            (mu - 0.5 * sigma ** 2) * dt,
            idio_sigma * math.sqrt(dt),
            size=n,
        )

        # Combined log-returns
        log_returns = beta * market_daily + idio_daily

        # Reconstruct price path via cumulative product
        path = np.empty(n, dtype=float)
        path[0] = s0 * math.exp(log_returns[0])
        for i in range(1, n):
            path[i] = path[i - 1] * math.exp(log_returns[i])

        # Build intraday OHLCV around Close
        intraday_pct = rng.uniform(0.005, 0.025, n)
        half = path * intraday_pct / 2.0
        high_arr  = path + half
        low_arr   = path - half
        open_arr  = np.empty(n, dtype=float)
        open_arr[0] = s0
        open_arr[1:] = path[:-1] * (1.0 + rng.uniform(-0.005, 0.005, n - 1))
        volume_arr = rng.lognormal(mean=13.8, sigma=0.5, size=n).astype(int)

        end_date = datetime.date.today()
        dates: list[datetime.date] = []
        current = end_date - datetime.timedelta(days=1)
        while len(dates) < n:
            if current.weekday() < 5:
                dates.append(current)
            current -= datetime.timedelta(days=1)
        dates.reverse()
        date_index = pd.to_datetime(dates)

        result[ticker] = pd.DataFrame(
            {"Open": open_arr, "High": high_arr, "Low": low_arr,
             "Close": path, "Volume": volume_arr, "Adj Close": path},
            index=date_index,
        )
        result[ticker].index.name = "Date"

    return result


_DEMO_DATA: dict[str, pd.DataFrame] = _build_all_data()


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------

class DemoDataProvider(MarketDataProvider):
    """Deterministic synthetic market data provider.

    Generates 252 trading days of OHLCV data using Geometric Brownian
    Motion seeded with ``numpy.random.default_rng(seed=42)``.
    No network calls are ever made.

    Supported tickers
    -----------------
    TCS.NS, INFY.NS, HDFCBANK.NS, RELIANCE.NS, WIPRO.NS, HCLTECH.NS,
    ICICIBANK.NS, KOTAKBANK.NS, LTIM.NS, AXISBANK.NS, TATAMOTORS.NS,
    BAJFINANCE.NS
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def data_mode_label(self) -> str:
        return "DEMO DATA"

    @property
    def supported_tickers(self) -> list[str]:
        """Return the list of all pre-generated ticker symbols."""
        return list(_TICKERS.keys())

    # ------------------------------------------------------------------
    # Quote methods
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> dict:
        """Return a synthetic latest quote derived from the last two rows.

        Parameters
        ----------
        symbol:
            Must be one of the 12 supported NSE tickers.

        Returns
        -------
        dict
            Quote dict with keys: ``symbol``, ``price``, ``change``,
            ``change_percent``, ``volume``, ``market_status``,
            ``timestamp``.

        Raises
        ------
        KeyError
            If *symbol* is not in the demo dataset.
        """
        symbol = symbol.upper()
        if symbol not in _DEMO_DATA:
            raise KeyError(
                f"'{symbol}' is not in the demo dataset. "
                f"Supported: {list(_DEMO_DATA.keys())}"
            )
        df = _DEMO_DATA[symbol]
        latest = df.iloc[-1]
        prev = df.iloc[-2]

        price: float = float(latest["Close"])
        prev_close: float = float(prev["Close"])
        change: float = price - prev_close
        change_pct: float = (change / prev_close) * 100 if prev_close != 0 else 0.0
        volume: int = int(latest["Volume"])
        timestamp: str = str(df.index[-1].date())

        return {
            "symbol": symbol,
            "price": round(price, 2),
            "change": round(change, 2),
            "change_percent": round(change_pct, 4),
            "volume": volume,
            "market_status": "UNKNOWN",
            "timestamp": timestamp,
        }

    def get_quotes(self, symbols: list[str]) -> list[dict]:
        """Return quotes for multiple tickers.

        Parameters
        ----------
        symbols:
            List of ticker symbols.

        Returns
        -------
        list[dict]
        """
        return [self.get_quote(sym) for sym in symbols]

    # ------------------------------------------------------------------
    # Historical data
    # ------------------------------------------------------------------

    def get_historical_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return a slice of the pre-generated OHLCV history.

        Parameters
        ----------
        symbol:
            One of the 12 supported NSE tickers.
        start:
            Start date as ``"YYYY-MM-DD"`` (inclusive).
        end:
            End date as ``"YYYY-MM-DD"`` (inclusive).
        interval:
            Ignored for the demo provider (always daily resolution).

        Returns
        -------
        pd.DataFrame
            Columns: ``Open``, ``High``, ``Low``, ``Close``,
            ``Volume``, ``Adj Close``.

        Raises
        ------
        KeyError
            If *symbol* is not supported.
        ValueError
            If the date range yields no rows.
        """
        symbol = symbol.upper()
        if symbol not in _DEMO_DATA:
            raise KeyError(
                f"'{symbol}' is not in the demo dataset. "
                f"Supported: {list(_DEMO_DATA.keys())}"
            )
        df = _DEMO_DATA[symbol]
        mask = (df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))
        sliced = df.loc[mask].copy()
        if sliced.empty:
            raise ValueError(
                f"No demo data for '{symbol}' in range {start} to {end}. "
                f"Available range: {df.index[0].date()} to {df.index[-1].date()}"
            )
        return sliced

    # ------------------------------------------------------------------
    # Market status & metadata
    # ------------------------------------------------------------------

    def get_market_status(self) -> str:
        """Always returns ``'UNKNOWN'`` for the demo provider."""
        return "UNKNOWN"

    def get_last_updated(self) -> str:
        """Return the most recent date present in the demo dataset."""
        latest_dates = [df.index[-1] for df in _DEMO_DATA.values()]
        most_recent = max(latest_dates)
        return str(most_recent.date())

    # ------------------------------------------------------------------
    # Symbol search
    # ------------------------------------------------------------------

    def search_symbols(self, query: str) -> list[dict]:
        """Search the 12 supported demo tickers by symbol or company name.

        Parameters
        ----------
        query:
            Case-insensitive search string matched against ticker symbols
            and company names.

        Returns
        -------
        list[dict]
            Matching entries with keys: ``symbol``, ``name``,
            ``exchange``.
        """
        query_lower = query.lower()
        results: list[dict[str, Any]] = []
        for ticker, name in _NAMES.items():
            if query_lower in ticker.lower() or query_lower in name.lower():
                results.append(
                    {
                        "symbol": ticker,
                        "name": name,
                        "exchange": "NSE",
                    }
                )
        return results
