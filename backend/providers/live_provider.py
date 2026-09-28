"""
live_provider.py
----------------
Stub implementation of MarketDataProvider for live market data.

This provider is a placeholder — every method raises
``ProviderNotConfiguredError`` instructing the operator to configure
the required environment variables before using real live data.
"""

from __future__ import annotations

import pandas as pd

from .base_provider import MarketDataProvider, ProviderNotConfiguredError


# ---------------------------------------------------------------------------
# Error message shown to callers when this stub is invoked
# ---------------------------------------------------------------------------

_CONFIG_ERROR_MSG = (
    "Live data provider is not configured. "
    "Set MARKET_DATA_PROVIDER and MARKET_DATA_API_KEY in your .env file. "
    "See .env.example for setup instructions."
)


class LiveDataProvider(MarketDataProvider):
    """Stub live-data provider.

    All methods unconditionally raise :class:`ProviderNotConfiguredError`
    with a descriptive setup message.  Replace this class (or wire in a
    real implementation) once the required credentials are available.
    """

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @property
    def data_mode_label(self) -> str:
        return "LIVE DATA"

    # ------------------------------------------------------------------
    # Abstract method implementations — all raise configuration error
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> dict:
        """Not available — raises ProviderNotConfiguredError."""
        raise ProviderNotConfiguredError(_CONFIG_ERROR_MSG)

    def get_quotes(self, symbols: list[str]) -> list[dict]:
        """Not available — raises ProviderNotConfiguredError."""
        raise ProviderNotConfiguredError(_CONFIG_ERROR_MSG)

    def get_historical_data(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str,
    ) -> pd.DataFrame:
        """Not available — raises ProviderNotConfiguredError."""
        raise ProviderNotConfiguredError(_CONFIG_ERROR_MSG)

    def get_market_status(self) -> str:
        """Not available — raises ProviderNotConfiguredError."""
        raise ProviderNotConfiguredError(_CONFIG_ERROR_MSG)

    def get_last_updated(self) -> str:
        """Not available — raises ProviderNotConfiguredError."""
        raise ProviderNotConfiguredError(_CONFIG_ERROR_MSG)

    def search_symbols(self, query: str) -> list[dict]:
        """Not available — raises ProviderNotConfiguredError."""
        raise ProviderNotConfiguredError(_CONFIG_ERROR_MSG)
