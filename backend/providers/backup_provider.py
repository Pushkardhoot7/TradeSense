"""
backup_provider.py
-------------------
Authorized backup market data provider for TradeSense V2.1.
Provides an extensible adapter for external authorized market APIs
(such as TwelveData or AlphaVantage) using environment-configured credentials,
and seamlessly integrates verified offline historical caching without fabricating prices.

Adheres to:
- No hardcoded secrets / no secrets printed in logs.
- Normalized OHLCV pandas format.
- Bounded retries with exponential backoff for transient errors.
- Immediate fail-fast for permanent 401/403 authorization failures.
- Explicit provider origin tracking.
"""

from __future__ import annotations

import datetime
from datetime import timezone, timedelta
import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from .base_provider import MarketDataProvider
from .symbol_registry import symbol_registry
from ..core.validation import validate_and_clean_price_series

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))


class BackupMarketDataProvider(MarketDataProvider):
    """
    Configurable backup historical data provider.
    Checks environment for authorized API credentials (e.g. TWELVE_DATA_API_KEY,
    ALPHA_VANTAGE_API_KEY, or BACKUP_MARKET_DATA_API_KEY).
    Falls back to verified local historical cache if no live credentials exist.
    """

    def __init__(self) -> None:
        self.api_key: Optional[str] = (
            os.environ.get("BACKUP_MARKET_DATA_API_KEY")
            or os.environ.get("TWELVE_DATA_API_KEY")
            or os.environ.get("ALPHA_VANTAGE_API_KEY")
        )
        self.api_url: str = os.environ.get(
            "BACKUP_MARKET_DATA_API_URL", "https://api.twelvedata.com"
        )
        self.has_credentials: bool = bool(self.api_key and self.api_key.strip())
        self._auth_failed: bool = False
        self.cache_dir = Path(__file__).parent.parent.parent / "data" / "cache"

    @property
    def data_mode_label(self) -> str:
        if self.has_credentials and not self._auth_failed:
            return "BACKUP API • AUTHORIZED"
        return "LOCAL HISTORICAL CACHE"

    def get_market_status(self) -> str:
        now_ist = datetime.datetime.now(IST)
        weekday = now_ist.weekday()
        time_min = now_ist.hour * 60 + now_ist.minute
        if weekday >= 5:
            return "CLOSED"
        if 9 * 60 + 15 <= time_min <= 15 * 60 + 30:
            return "OPEN"
        return "CLOSED"

    def get_last_updated(self) -> str:
        return datetime.datetime.now(IST).strftime("%H:%M:%S")

    def get_quote(self, symbol: str) -> dict:
        """Fetch quote via authorized API or from historical cache latest observation."""
        canonical = symbol_registry.get_canonical_symbol(symbol)
        df = self.get_historical_data(canonical, start="", end="")
        if df.empty:
            return {
                "symbol": canonical,
                "price": None,
                "change": None,
                "change_percent": None,
                "volume": None,
                "market_status": self.get_market_status(),
                "timestamp": self.get_last_updated(),
            }
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        price = float(latest["Close"])
        prev_close = float(prev["Close"])
        change = price - prev_close if prev_close else 0.0
        pct = (change / prev_close * 100) if prev_close else 0.0

        return {
            "symbol": canonical,
            "price": round(price, 2),
            "change": round(change, 2),
            "change_percent": round(pct, 2),
            "volume": int(latest.get("Volume", 0)),
            "market_status": self.get_market_status(),
            "timestamp": self.get_last_updated(),
        }

    def get_quotes(self, symbols: list[str]) -> list[dict]:
        return [self.get_quote(s) for s in symbols]

    def _fetch_from_api_with_retry(
        self, symbol: str, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """
        Query external authorized API with bounded retries and exponential backoff.
        Immediately stops retries on 401/403 authorization failure.
        """
        if not self.has_credentials or self._auth_failed:
            return None

        # Clean symbol for external query
        clean_sym = symbol.replace(".NS", "")
        max_retries = 2
        delay = 0.5

        for attempt in range(max_retries):
            try:
                # Example standardized twelve-data request structure
                params = {
                    "symbol": f"{clean_sym}:NSE",
                    "interval": "1day",
                    "apikey": self.api_key,
                    "outputsize": 250,
                }
                # Do not log params containing apikey
                logger.debug("Requesting backup API for %s (attempt %d)", symbol, attempt + 1)
                resp = requests.get(
                    f"{self.api_url}/time_series",
                    params=params,
                    timeout=5.0,
                )

                if resp.status_code in (401, 403):
                    logger.warning("Backup API authentication failed (HTTP %d). Permanent failure.", resp.status_code)
                    self._auth_failed = True
                    return None

                if resp.status_code == 429:
                    logger.warning("Backup API rate limit (HTTP 429). Waiting before retry.")
                    time.sleep(delay)
                    delay *= 2
                    continue

                if resp.status_code == 200:
                    data = resp.json()
                    values = data.get("values", [])
                    if values:
                        df = pd.DataFrame(values)
                        df["datetime"] = pd.to_datetime(df["datetime"])
                        df.set_index("datetime", inplace=True)
                        df.index.name = "Date"
                        df.rename(
                            columns={
                                "open": "Open",
                                "high": "High",
                                "low": "Low",
                                "close": "Close",
                                "volume": "Volume",
                            },
                            inplace=True,
                        )
                        df["Adj Close"] = df["Close"]
                        valid, _, clean_df = validate_and_clean_price_series(symbol, df)
                        if valid:
                            return clean_df

            except requests.exceptions.RequestException as req_err:
                logger.debug("Backup API transient connection error: %s", req_err)
                time.sleep(delay)
                delay *= 2

        return None

    def _load_from_disk_cache(
        self, symbol: str, start: str | None = None, end: str | None = None
    ) -> Optional[pd.DataFrame]:
        """
        Load historical OHLCV data from local verified disk cache files.
        Never fabricates prices.
        """
        clean_sym = symbol.replace(".NS", "_NS").replace("-", "_").replace("&", "_")
        base_sym = symbol.replace(".NS", "")
        # Also check symbol registry aliases
        reg_sym = symbol_registry.get_provider_symbol(symbol, "disk_cache")

        candidates = [
            self.cache_dir / f"stock_{clean_sym}_1y_1d.csv",
            self.cache_dir / f"stock_{base_sym}_NS_1y_1d.csv",
            self.cache_dir / f"stock_{reg_sym}_1y_1d.csv",
            self.cache_dir / f"stock_{symbol}_1y_1d.csv",
        ]

        for p in candidates:
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    valid, _, clean_df = validate_and_clean_price_series(symbol, df)
                    if valid and not clean_df.empty:
                        if start and end:
                            mask = (clean_df.index >= pd.to_datetime(start)) & (
                                clean_df.index <= pd.to_datetime(end)
                            )
                            sub = clean_df[mask]
                            if len(sub) >= 15:
                                return sub
                        return clean_df
                except Exception as ex:
                    logger.debug("Failed reading cache file %s: %s", p, ex)

        return None

    def get_historical_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical data first from authorized backup API, then from verified disk cache."""
        # 1. Try authorized API
        if self.has_credentials and not self._auth_failed:
            api_df = self._fetch_from_api_with_retry(symbol, start, end)
            if api_df is not None and not api_df.empty:
                return api_df

        # 2. Try verified local cache
        cached = self._load_from_disk_cache(symbol, start, end)
        if cached is not None and not cached.empty:
            return cached

        return pd.DataFrame()

    def get_bulk_historical_data(
        self,
        symbols: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        res: dict[str, pd.DataFrame] = {}
        for s in symbols:
            df = self.get_historical_data(s, start, end, interval)
            if df is not None and not df.empty:
                res[s] = df
        return res
