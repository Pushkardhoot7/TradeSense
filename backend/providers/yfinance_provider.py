"""
yfinance_provider.py
--------------------
Concrete MarketDataProvider implementation backed by the yfinance library.

This provider is suitable for research / back-testing workflows where
historical OHLCV data is required but real-time streaming quotes are not
needed.  Market status is always reported as 'UNKNOWN' because yfinance
does not expose a reliable live market-state endpoint.
"""

from __future__ import annotations

import datetime
from typing import Any

import pandas as pd
import yfinance as yf

from .base_provider import MarketDataProvider


class YFinanceProvider(MarketDataProvider):
    """Market data provider powered by the ``yfinance`` library.

    Data mode label: **HISTORICAL DATA**

    Notes
    -----
    * ``get_market_status`` always returns ``'UNKNOWN'`` because yfinance
      does not expose a reliable live market-state endpoint.
    * ``search_symbols`` performs a best-effort search via
      ``yf.Ticker(query).info`` — results may be limited.
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def data_mode_label(self) -> str:
        return "HISTORICAL DATA"

    # ------------------------------------------------------------------
    # Quote methods
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> dict:
        """Fetch a snapshot quote for *symbol* from yfinance Ticker info.

        Parameters
        ----------
        symbol:
            Ticker symbol accepted by yfinance (e.g. ``"TCS.NS"``).

        Returns
        -------
        dict
            Quote dict with keys: ``symbol``, ``price``, ``change``,
            ``change_percent``, ``volume``, ``market_status``,
            ``timestamp``.
        """
        ticker: yf.Ticker = yf.Ticker(symbol)
        info: dict[str, Any] = {}
        try:
            info = ticker.info or {}
        except Exception:
            pass

        price: float | None = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        prev_close: float | None = info.get("previousClose") or info.get(
            "regularMarketPreviousClose"
        )
        change: float | None = None
        change_pct: float | None = None
        if price is not None and prev_close is not None and prev_close != 0:
            change = price - prev_close
            change_pct = (change / prev_close) * 100

        volume: int | None = info.get("volume") or info.get("regularMarketVolume")

        timestamp: str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "symbol": symbol,
            "price": price,
            "change": change,
            "change_percent": change_pct,
            "volume": volume,
            "market_status": "UNKNOWN",
            "timestamp": timestamp,
        }

    def get_quotes(self, symbols: list[str]) -> list[dict]:
        """Fetch quotes for multiple tickers sequentially.

        Parameters
        ----------
        symbols:
            List of ticker symbols.

        Returns
        -------
        list[dict]
            Each element has the structure documented in :meth:`get_quote`.
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
        """Download OHLCV history via ``yfinance.download``.

        Parameters
        ----------
        symbol:
            Ticker symbol (e.g. ``"RELIANCE.NS"``).
        start:
            Start date as ``"YYYY-MM-DD"`` (inclusive).
        end:
            End date as ``"YYYY-MM-DD"`` (inclusive).
        interval:
            Bar interval — ``"1d"``, ``"1wk"``, or ``"1mo"``.

        Returns
        -------
        pd.DataFrame
            DatetimeIndex (tz-naive) with columns:
            ``Open``, ``High``, ``Low``, ``Close``, ``Volume``,
            ``Adj Close``.

        Raises
        ------
        ValueError
            If no data is returned for the requested symbol / range.
        """
        df: pd.DataFrame = yf.download(
            symbol,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
        )

        if df.empty:
            raise ValueError(
                f"No data returned by yfinance for '{symbol}' "
                f"({start} to {end}, interval={interval})."
            )

        # --- flatten multi-level columns that yfinance sometimes returns ---
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # --- ensure expected columns exist ---
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Downloaded DataFrame for '{symbol}' is missing columns: {missing}"
            )

        # --- add Adj Close column if yfinance didn't provide it ---
        if "Adj Close" not in df.columns:
            df["Adj Close"] = df["Close"]

        # --- strip timezone info from the DatetimeIndex ---
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index.name = "Date"

        # --- reorder columns to canonical order ---
        df = df[["Open", "High", "Low", "Close", "Volume", "Adj Close"]].copy()
        df = df.dropna(how="all")
        return df

    # ------------------------------------------------------------------
    # Market status & metadata
    # ------------------------------------------------------------------

    def get_market_status(self) -> str:
        """Always returns ``'UNKNOWN'`` for yfinance.

        yfinance does not provide a reliable live market-state endpoint,
        so the status cannot be determined programmatically.
        """
        return "UNKNOWN"

    def get_last_updated(self) -> str:
        """Return the current local time as the last-updated timestamp."""
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ------------------------------------------------------------------
    # Symbol search
    # ------------------------------------------------------------------

    def search_symbols(self, query: str) -> list[dict]:
        """Best-effort symbol search using yfinance ``Ticker.info``.

        Parameters
        ----------
        query:
            Ticker symbol or company name fragment.

        Returns
        -------
        list[dict]
            List with one element (the matched ticker) or an empty list
            when the query resolves to no valid instrument.
        """
        try:
            ticker = yf.Ticker(query)
            info: dict[str, Any] = ticker.info or {}
            name = info.get("longName") or info.get("shortName") or query
            exchange = info.get("exchange") or "UNKNOWN"
            return [{"symbol": query.upper(), "name": name, "exchange": exchange}]
        except Exception:
            return []
