"""
live_provider.py
----------------
Concrete Live MarketDataProvider for TradeSense V2.1.
Connects to real-time and historical National Stock Exchange (NSE) market data via yfinance,
with robust error classification, authorized backup provider failover, and strict data validation.
"""

from __future__ import annotations

import concurrent.futures
import datetime
from datetime import timezone, timedelta
import logging
import os
import time
from typing import Any, Optional

import pandas as pd
import yfinance as yf

from .base_provider import MarketDataProvider
from .symbol_registry import symbol_registry
from .backup_provider import BackupMarketDataProvider
from .errors import MarketDataError, MarketDataErrorCategory, classify_yfinance_error
from ..core.validation import validate_and_clean_price_series

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
CACHE_TTL_SECONDS = int(os.environ.get("TRADESENSE_CACHE_TTL_SECONDS", "600"))


class LiveDataProvider(MarketDataProvider):
    """
    Production-grade market data provider for NSE equities.
    Integrates yfinance with structured error handling, backup provider fallback,
    and memory caching with TTL.
    """

    # Class-level bounded in-memory cache:
    # key: (tuple(sorted(symbols)), start, end, interval)
    # val: (timestamp, data_dict, ticker_sources_dict, errors_dict)
    _cache: dict[tuple, tuple[float, dict[str, pd.DataFrame], dict[str, str], dict[str, MarketDataError]]] = {}
    _yf_auth_failed: bool = False

    def __init__(self) -> None:
        self.backup_provider = BackupMarketDataProvider()
        self.last_errors: dict[str, MarketDataError] = {}
        self.ticker_sources: dict[str, str] = {}
        self.last_cache_hit: bool = False

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def data_mode_label(self) -> str:
        status = self.get_market_status()
        if status == "OPEN":
            return "● LIVE • NSE (OPEN)"
        elif status in ("CLOSED", "POST-MARKET"):
            return "NSE CLOSED • CLOSING PRICES"
        return "PRE-MARKET • NSE"

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
        target_sym = symbol_registry.get_provider_symbol(symbol, "yfinance")
        info: dict[str, Any] = {}

        if not self._yf_auth_failed:
            try:
                ticker = yf.Ticker(target_sym)
                info = ticker.fast_info if hasattr(ticker, "fast_info") else (ticker.info or {})
            except Exception as exc:
                err = classify_yfinance_error(symbol, exc)
                self.last_errors[symbol] = err
                if err.category == MarketDataErrorCategory.PROVIDER_AUTH_ERROR:
                    logger.warning("[LiveDataProvider] yfinance auth failure (HTTP 401) on %s. Flagging auth failure.", symbol)
                    self._yf_auth_failed = True

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

        # If live quote unavailable from yfinance, attempt backup provider
        source = "NSE via yfinance"
        if price is None:
            backup_quote = self.backup_provider.get_quote(symbol)
            if backup_quote.get("price") is not None:
                return backup_quote
            source = "Unavailable"

        market_status = self.get_market_status()
        is_open = (market_status == "OPEN")
        price_type = "LIVE_QUOTE" if is_open else "OFFICIAL_CLOSE"

        # Determine session date (today if open, or most recent weekday/trading date)
        now_ist = datetime.datetime.now(IST)
        session_date = now_ist.strftime("%Y-%m-%d")
        if not is_open:
            # If weekend or pre-market, session date is the latest completed session
            weekday = now_ist.weekday()
            if weekday == 5: # Saturday -> Friday
                session_date = (now_ist - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            elif weekday == 6: # Sunday -> Friday
                session_date = (now_ist - datetime.timedelta(days=2)).strftime("%Y-%m-%d")
            elif now_ist.hour < 9 or (now_ist.hour == 9 and now_ist.minute < 15):
                # Before market open on Monday -> Friday
                days_back = 3 if weekday == 0 else 1
                session_date = (now_ist - datetime.timedelta(days=days_back)).strftime("%Y-%m-%d")

        return {
            "symbol": symbol,
            "price": price,
            "change": change,
            "change_percent": change_pct,
            "volume": volume,
            "market_status": market_status,
            "is_market_open": is_open,
            "price_type": price_type,
            "session_date": session_date,
            "data_source": source,
            "timestamp": self.get_last_updated(),
        }

    def get_quotes(self, symbols: list[str]) -> list[dict]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, max(len(symbols), 1))) as executor:
            return list(executor.map(self.get_quote, symbols))

    # ------------------------------------------------------------------
    # Historical data methods with Batch Acceleration & TTL Cache
    # ------------------------------------------------------------------

    def get_historical_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        target_sym = symbol_registry.get_provider_symbol(symbol, "yfinance")
        df = pd.DataFrame()

        # 1. Primary yfinance download (skip if permanent 401 encountered)
        if not self._yf_auth_failed:
            try:
                df = yf.download(
                    target_sym,
                    start=start,
                    end=end,
                    interval=interval,
                    auto_adjust=False,
                    progress=False,
                )
            except Exception as dl_err:
                err = classify_yfinance_error(symbol, dl_err)
                self.last_errors[symbol] = err
                if err.category == MarketDataErrorCategory.PROVIDER_AUTH_ERROR:
                    logger.warning("[LiveDataProvider] yfinance auth failure on %s. Switching to backup provider.", symbol)
                    self._yf_auth_failed = True
                else:
                    logger.debug("yf download error for %s: %s", target_sym, dl_err)

        if not df.empty:
            is_valid, reason, clean_df = validate_and_clean_price_series(symbol, df)
            if is_valid:
                self.ticker_sources[symbol] = "yfinance"
                return clean_df
            else:
                self.last_errors[symbol] = MarketDataError(
                    symbol=symbol,
                    provider="yfinance",
                    category=MarketDataErrorCategory.VALIDATION_FAILURE,
                    message=f"Downloaded data failed validation: {reason}",
                )

        # 2. Backup provider fallback
        logger.info("Attempting backup provider for %s...", symbol)
        backup_df = self.backup_provider.get_historical_data(symbol, start, end, interval)
        if not backup_df.empty:
            is_valid, reason, clean_df = validate_and_clean_price_series(symbol, backup_df)
            if is_valid:
                source = "backup_api" if self.backup_provider.has_credentials else "local_historical_cache"
                self.ticker_sources[symbol] = source
                logger.info("Loaded '%s' from backup provider (%s).", symbol, source)
                return clean_df

        raise ValueError(f"No valid market data returned for '{symbol}' ({start} to {end}).")

    def get_bulk_historical_data(
        self,
        symbols: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        now = time.time()
        cache_key = (tuple(sorted(symbols)), start, end, interval)

        # 1. Check TTL Cache
        if cache_key in self._cache:
            ts, cached_data, cached_sources, cached_errors = self._cache[cache_key]
            if (now - ts) < CACHE_TTL_SECONDS and len(cached_data) >= len(symbols) * 0.7:
                logger.info("Live bulk data served from memory cache (%d stocks, TTL: %ds)", len(cached_data), CACHE_TTL_SECONDS)
                self.ticker_sources = cached_sources.copy()
                self.last_errors = cached_errors.copy()
                self.last_cache_hit = True
                return cached_data

        self.last_cache_hit = False
        results: dict[str, pd.DataFrame] = {}
        self.last_errors = {}
        self.ticker_sources = {}

        # 2. Batch download via yfinance (if not auth blocked)
        if not self._yf_auth_failed:
            query_symbols = [symbol_registry.get_provider_symbol(s, "yfinance") for s in symbols]
            try:
                logger.info("Initiating high-speed batch download for %d symbols via yfinance...", len(query_symbols))
                batch_df = yf.download(
                    tickers=query_symbols,
                    start=start,
                    end=end,
                    interval=interval,
                    group_by="ticker",
                    auto_adjust=False,
                    threads=True,
                    progress=False,
                )

                if not batch_df.empty:
                    for orig_sym in symbols:
                        q_sym = symbol_registry.get_provider_symbol(orig_sym, "yfinance")
                        try:
                            sub_df = pd.DataFrame()
                            if hasattr(batch_df.columns, "levels") and q_sym in batch_df.columns.levels[0]:
                                sub_df = batch_df[q_sym].dropna(how="all").copy()
                            elif "Close" in batch_df.columns and q_sym in batch_df["Close"].columns:
                                close_s = batch_df["Close"][q_sym].dropna()
                                if len(close_s) > 0:
                                    sub_df = pd.DataFrame({
                                        "Open": batch_df["Open"][q_sym] if "Open" in batch_df else close_s,
                                        "High": batch_df["High"][q_sym] if "High" in batch_df else close_s,
                                        "Low": batch_df["Low"][q_sym] if "Low" in batch_df else close_s,
                                        "Close": close_s,
                                        "Volume": batch_df["Volume"][q_sym] if "Volume" in batch_df else 0,
                                        "Adj Close": close_s,
                                    }).dropna(how="all")

                            if not sub_df.empty:
                                is_valid, _, clean_df = validate_and_clean_price_series(orig_sym, sub_df)
                                if is_valid:
                                    results[orig_sym] = clean_df
                                    self.ticker_sources[orig_sym] = "yfinance"
                        except Exception as parse_err:
                            logger.debug("Parsing error for %s: %s", orig_sym, parse_err)

            except Exception as batch_exc:
                err = classify_yfinance_error("BULK_BATCH", batch_exc)
                if err.category == MarketDataErrorCategory.PROVIDER_AUTH_ERROR:
                    logger.warning("[LiveDataProvider] Batch download HTTP 401 Unauthorized: %s", batch_exc)
                    self._yf_auth_failed = True
                else:
                    logger.warning("Bulk batch download failed, falling back: %s", batch_exc)

        # 3. Fallback for any missing stocks via individual worker / backup provider
        missing = [s for s in symbols if s not in results]
        if missing:
            logger.info("Checking backup provider for %d missing/unauthorized tickers...", len(missing))

            def _fetch_fallback(sym: str) -> tuple[str, Optional[pd.DataFrame], Optional[str], Optional[MarketDataError]]:
                try:
                    df = self.get_historical_data(sym, start, end, interval)
                    src = self.ticker_sources.get(sym, "unknown")
                    return sym, df, src, None
                except Exception as ex:
                    cat = MarketDataErrorCategory.EMPTY_RESULT
                    if "insufficient" in str(ex).lower():
                        cat = MarketDataErrorCategory.INSUFFICIENT_HISTORY
                    err = MarketDataError(
                        symbol=sym,
                        provider="all_providers",
                        category=cat,
                        message=str(ex),
                    )
                    return sym, None, None, err

            with concurrent.futures.ThreadPoolExecutor(max_workers=min(12, len(missing))) as executor:
                for sym, df, src, err in executor.map(_fetch_fallback, missing):
                    if df is not None and not df.empty:
                        results[sym] = df
                        if src:
                            self.ticker_sources[sym] = src
                    elif err:
                        self.last_errors[sym] = err

        # 4. Update memory cache if at least some results returned
        if len(results) > 0:
            self._cache[cache_key] = (now, results, self.ticker_sources, self.last_errors)
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
