"""
live_provider.py
----------------
Concrete Live MarketDataProvider for TradeSense V2.1.
Connects to real-time and historical National Stock Exchange (NSE) market data via yfinance.
"""

from __future__ import annotations

import concurrent.futures
import datetime
from datetime import timezone, timedelta
import logging
from typing import Any

import pandas as pd
import yfinance as yf

from .base_provider import MarketDataProvider

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


class LiveDataProvider(MarketDataProvider):
    """Real live market data provider for NSE stocks."""

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def data_mode_label(self) -> str:
        return "● LIVE • NSE"

    # ------------------------------------------------------------------
    # Market status & timestamp
    # ------------------------------------------------------------------

    def get_market_status(self) -> str:
        now_ist = datetime.datetime.now(IST)
        weekday = now_ist.weekday()
        time_min = now_ist.hour * 60 + now_ist.minute
        if weekday >= 5:
            return "CLOSED"
        if 9 * 60 <= time_min < 9 * 60 + 15:
            return "PRE-MARKET"
        if 9 * 60 + 15 <= time_min <= 15 * 60 + 30:
            return "OPEN"
        if 15 * 60 + 30 < time_min <= 16 * 60:
            return "POST-MARKET"
        return "CLOSED"

    def get_last_updated(self) -> str:
        return datetime.datetime.now(IST).strftime("%H:%M:%S")

    # ------------------------------------------------------------------
    # Quote methods
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> dict:
        ticker = yf.Ticker(symbol)
        info: dict[str, Any] = {}
        try:
            info = ticker.fast_info if hasattr(ticker, "fast_info") else (ticker.info or {})
        except Exception:
            try:
                info = ticker.info or {}
            except Exception:
                pass

        price = (
            info.get("lastPrice")
            or info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("previousClose")
        )
        prev_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
        change = None
        change_pct = None
        if price is not None and prev_close is not None and prev_close != 0:
            change = price - prev_close
            change_pct = (change / prev_close) * 100

        volume = info.get("lastVolume") or info.get("volume") or info.get("regularMarketVolume")
        return {
            "symbol": symbol,
            "price": price,
            "change": change,
            "change_percent": change_pct,
            "volume": volume,
            "market_status": self.get_market_status(),
            "timestamp": self.get_last_updated(),
        }

    def get_quotes(self, symbols: list[str]) -> list[dict]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, max(len(symbols), 1))) as executor:
            return list(executor.map(self.get_quote, symbols))

    # ------------------------------------------------------------------
    # Historical data methods
    # ------------------------------------------------------------------

    def get_historical_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )
        if df.empty:
            raise ValueError(f"No live data returned for '{symbol}' ({start} to {end}).")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Downloaded DataFrame for '{symbol}' is missing columns: {missing}")

        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"
        df = df[["Open", "High", "Low", "Close", "Volume", "Adj Close"]].copy()
        df = df.dropna(how="all")
        return df

    def get_bulk_historical_data(
        self,
        symbols: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}

        def _fetch_one(sym: str):
            try:
                data = self.get_historical_data(sym, start, end, interval)
                return sym, data
            except Exception as e:
                logger.warning("Live data fetch error for %s: %s", sym, e)
                return sym, None

        max_workers = min(15, max(len(symbols), 1))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            fetched = executor.map(_fetch_one, symbols)
            for sym, df in fetched:
                if df is not None and not df.empty:
                    results[sym] = df

        return results

    # ------------------------------------------------------------------
    # Symbol search
    # ------------------------------------------------------------------

    def search_symbols(self, query: str) -> list[dict]:
        try:
            ticker = yf.Ticker(query)
            info = ticker.info or {}
            name = info.get("longName") or info.get("shortName") or query
            exchange = info.get("exchange") or "NSE"
            return [{"symbol": query.upper(), "name": name, "exchange": exchange}]
        except Exception:
            return []
