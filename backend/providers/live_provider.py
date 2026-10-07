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
    # Historical data methods with Batch Acceleration & TTL Cache
    # ------------------------------------------------------------------

    _cache: dict[tuple, tuple[float, dict[str, pd.DataFrame]]] = {}

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
        import time

        now = time.time()
        cache_key = (tuple(sorted(symbols)), start, end, interval)

        # 1. Check TTL Cache (10 minutes TTL)
        if cache_key in self._cache:
            ts, cached_data = self._cache[cache_key]
            if now - ts < 600 and len(cached_data) >= len(symbols) * 0.7:
                logger.info("Live bulk data served from memory cache (%d stocks)", len(cached_data))
                return cached_data

        results: dict[str, pd.DataFrame] = {}

        # 2. High-speed single batch download via yfinance
        try:
            logger.info("Initiating high-speed batch download for %d symbols...", len(symbols))
            batch_df = yf.download(
                tickers=symbols,
                start=start,
                end=end,
                interval=interval,
                group_by="ticker",
                auto_adjust=False,
                threads=True,
                progress=False,
            )

            if not batch_df.empty:
                if len(symbols) == 1:
                    sym = symbols[0]
                    clean = batch_df.dropna(how="all")
                    if not clean.empty:
                        if isinstance(clean.columns, pd.MultiIndex):
                            clean.columns = clean.columns.get_level_values(0)
                        if "Close" in clean.columns and clean["Close"].dropna().shape[0] > 5:
                            if "Adj Close" not in clean.columns:
                                clean["Adj Close"] = clean["Close"]
                            results[sym] = clean
                else:
                    for sym in symbols:
                        try:
                            # Handle both multi-index formats
                            if hasattr(batch_df.columns, "levels") and sym in batch_df.columns.levels[0]:
                                sub = batch_df[sym].dropna(how="all").copy()
                                if not sub.empty and "Close" in sub.columns and sub["Close"].dropna().shape[0] > 5:
                                    if "Adj Close" not in sub.columns:
                                        sub["Adj Close"] = sub["Close"]
                                    if sub.index.tz is not None:
                                        sub.index = sub.index.tz_localize(None)
                                    sub.index.name = "Date"
                                    results[sym] = sub
                            elif "Close" in batch_df.columns and sym in batch_df["Close"].columns:
                                close_s = batch_df["Close"][sym].dropna()
                                if len(close_s) > 5:
                                    sub = pd.DataFrame({
                                        "Open": batch_df["Open"][sym] if "Open" in batch_df else close_s,
                                        "High": batch_df["High"][sym] if "High" in batch_df else close_s,
                                        "Low": batch_df["Low"][sym] if "Low" in batch_df else close_s,
                                        "Close": close_s,
                                        "Volume": batch_df["Volume"][sym] if "Volume" in batch_df else 0,
                                        "Adj Close": close_s,
                                    }).dropna(how="all")
                                    if sub.index.tz is not None:
                                        sub.index = sub.index.tz_localize(None)
                                    sub.index.name = "Date"
                                    results[sym] = sub
                        except Exception as sym_err:
                            logger.debug("Parsing error for %s: %s", sym, sym_err)
        except Exception as batch_exc:
            logger.warning("Bulk batch download failed, falling back to parallel worker fetch: %s", batch_exc)

        # 3. Fallback for any missing stocks using ThreadPoolExecutor
        missing = [s for s in symbols if s not in results]
        if missing and len(missing) < len(symbols):
            logger.info("Fetching remaining %d tickers in parallel...", len(missing))
            def _fetch_one(sym: str):
                try:
                    data = self.get_historical_data(sym, start, end, interval)
                    return sym, data
                except Exception:
                    return sym, None

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, len(missing))) as executor:
                for sym, df in executor.map(_fetch_one, missing):
                    if df is not None and not df.empty:
                        results[sym] = df

        # 4. Update cache
        if len(results) > 0:
            self._cache[cache_key] = (now, results)
            logger.info("Successfully loaded %d / %d stocks in bulk.", len(results), len(symbols))

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
