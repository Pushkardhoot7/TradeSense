"""
base_provider.py
----------------
Abstract base class for all market data providers in TradeSense V2.

Every concrete provider (yfinance, live, demo) must inherit from
MarketDataProvider and implement all abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------

class ProviderNotConfiguredError(Exception):
    """Raised when a provider has not been properly configured (missing API
    keys, unsupported operation, etc.)."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class MarketDataProvider(ABC):
    """Abstract interface that every market data provider must implement.

    The interface is intentionally minimal so that different back-ends
    (yfinance, live REST APIs, synthetic demo data) can be swapped in
    transparently by the rest of the application.
    """

    # ------------------------------------------------------------------
    # Metadata helpers (concrete — may be overridden)
    # ------------------------------------------------------------------

    @property
    def data_mode_label(self) -> str:
        """Human-readable label for the data mode (e.g. 'DEMO DATA')."""
        return "UNKNOWN"

    # ------------------------------------------------------------------
    # Abstract methods — every provider must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def get_quote(self, symbol: str) -> dict:
        """Fetch the latest quote for a single ticker.

        Parameters
        ----------
        symbol:
            The ticker symbol (e.g. ``"TCS.NS"``).

        Returns
        -------
        dict with keys:
            * ``symbol``        - str
            * ``price``         - float | None
            * ``change``        - float | None  (absolute day change)
            * ``change_percent``- float | None  (percentage day change)
            * ``volume``        - int   | None
            * ``market_status`` - str   ('OPEN' | 'CLOSED' | 'PRE-MARKET' | 'UNKNOWN')
            * ``timestamp``     - str   (ISO-8601 or human-readable)
        """

    @abstractmethod
    def get_quotes(self, symbols: list[str]) -> list[dict]:
        """Fetch the latest quote for multiple tickers.

        Parameters
        ----------
        symbols:
            List of ticker symbols.

        Returns
        -------
        list[dict]
            Each element has the same structure as :meth:`get_quote`.
        """

    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame:
        """Fetch OHLCV history for a symbol over the given date range.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"RELIANCE.NS"``).
        start:
            Start date as ``"YYYY-MM-DD"`` string (inclusive).
        end:
            End date as ``"YYYY-MM-DD"`` string (inclusive).
        interval:
            Bar size - typically ``"1d"``, ``"1wk"``, ``"1mo"``.

        Returns
        -------
        pd.DataFrame
            Indexed by ``Date`` with columns:
            ``Open``, ``High``, ``Low``, ``Close``, ``Volume``,
            ``Adj Close``.
            The index is a ``DatetimeIndex`` (timezone-naive).
        """

    @abstractmethod
    def get_market_status(self) -> str:
        """Return the current market status.

        Returns
        -------
        str
            One of ``'OPEN'``, ``'CLOSED'``, ``'PRE-MARKET'``,
            or ``'UNKNOWN'``.
        """

    @abstractmethod
    def get_last_updated(self) -> str:
        """Return a human-readable timestamp of the last data update.

        Returns
        -------
        str
            e.g. ``"2024-01-15 15:30:00 IST"``
        """

    def get_bulk_historical_data(
        self,
        symbols: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> dict[str, pd.DataFrame]:
        """Fetch historical data for multiple symbols, returning {symbol: DataFrame}."""
        res: dict[str, pd.DataFrame] = {}
        for s in symbols:
            try:
                df = self.get_historical_data(s, start, end, interval)
                if not df.empty:
                    res[s] = df
            except Exception:
                pass
        return res

