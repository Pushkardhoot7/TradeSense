"""
errors.py
---------
Structured market data error classifications and exceptions for TradeSense V2.1.
Differentiates provider authorization failures, rate limits, timeouts, invalid symbols,
insufficient historical observations, and empty results.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class MarketDataErrorCategory(str, Enum):
    PROVIDER_AUTH_ERROR = "PROVIDER_AUTH_ERROR"      # HTTP 401 / Invalid crumb / Forbidden
    RATE_LIMIT_ERROR = "RATE_LIMIT_ERROR"            # HTTP 429 Too Many Requests
    TIMEOUT_ERROR = "TIMEOUT_ERROR"                  # Connection or request timeout
    INVALID_SYMBOL = "INVALID_SYMBOL"                # HTTP 404 / Quote not found / delisted
    INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"    # Too few trading days returned
    MISSING_OBSERVATIONS = "MISSING_OBSERVATIONS"    # Severe gaps in price data
    EMPTY_RESULT = "EMPTY_RESULT"                    # Provider returned zero rows
    VALIDATION_FAILURE = "VALIDATION_FAILURE"        # Data corrupt / negative prices / bad dates
    UNKNOWN_ERROR = "UNKNOWN_ERROR"                  # Unexpected exception


@dataclass
class MarketDataError:
    """Structured error object capturing provider failure details."""
    symbol: str
    provider: str
    category: MarketDataErrorCategory
    message: str
    status_code: Optional[int] = None
    is_transient: bool = False
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "provider": self.provider,
            "category": self.category.value,
            "message": self.message,
            "status_code": self.status_code,
            "is_transient": self.is_transient,
            "timestamp": self.timestamp,
        }


def classify_yfinance_error(symbol: str, exc: Exception) -> MarketDataError:
    """
    Classify an exception raised by yfinance into a structured MarketDataError.
    Distinguishes provider auth (401), rate limits (429), timeouts, and invalid symbols.
    """
    err_str = str(exc)
    err_lower = err_str.lower()

    # 1. HTTP 401: Authorization / Invalid crumb / Forbidden
    if "401" in err_str or "invalid crumb" in err_lower or "unable to access this feature" in err_lower:
        return MarketDataError(
            symbol=symbol,
            provider="yfinance",
            category=MarketDataErrorCategory.PROVIDER_AUTH_ERROR,
            message=f"Yahoo Finance HTTP 401 Unauthorized: {err_str}",
            status_code=401,
            is_transient=False,
        )

    # 2. HTTP 429: Rate limit
    if "429" in err_str or "too many requests" in err_lower or "rate limit" in err_lower:
        return MarketDataError(
            symbol=symbol,
            provider="yfinance",
            category=MarketDataErrorCategory.RATE_LIMIT_ERROR,
            message=f"Yahoo Finance rate limit encountered: {err_str}",
            status_code=429,
            is_transient=True,
        )

    # 3. Timeouts
    if "timeout" in err_lower or "timed out" in err_lower:
        return MarketDataError(
            symbol=symbol,
            provider="yfinance",
            category=MarketDataErrorCategory.TIMEOUT_ERROR,
            message=f"Request timed out: {err_str}",
            is_transient=True,
        )

    # 4. HTTP 404 / Delisted / Not Found
    if "404" in err_str or "quote not found" in err_lower or "delisted" in err_lower or "no data found" in err_lower:
        return MarketDataError(
            symbol=symbol,
            provider="yfinance",
            category=MarketDataErrorCategory.INVALID_SYMBOL,
            message=f"Ticker quote not found or delisted: {err_str}",
            status_code=404,
            is_transient=False,
        )

    # 5. Default unknown
    return MarketDataError(
        symbol=symbol,
        provider="yfinance",
        category=MarketDataErrorCategory.UNKNOWN_ERROR,
        message=f"yfinance query failed: {err_str}",
        is_transient=False,
    )
