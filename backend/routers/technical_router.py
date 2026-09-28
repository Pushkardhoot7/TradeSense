"""
technical_router.py - Technical & Fundamental Analysis API
"""
from __future__ import annotations
import logging
from typing import Any
from fastapi import APIRouter, HTTPException, Query
import yfinance as yf
from backend.core.technical import compute_all_technicals

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/stocks", tags=["Stock Intelligence"])

# In-memory cache for technicals: {symbol: (timestamp, data)}
_tech_cache: dict[str, tuple[float, Any]] = {}
_fund_cache: dict[str, tuple[float, Any]] = {}

@router.get("/{symbol}/technicals")
def get_stock_technicals(symbol: str, period: str = "6mo"):
    """Compute and return full technical analysis indicator suite."""
    import time
    now = time.time()
    cache_key = f"{symbol}_{period}"
    if cache_key in _tech_cache and (now - _tech_cache[cache_key][0] < 300):
        return _tech_cache[cache_key][1]

    try:
        df = yf.download(symbol, period=period, progress=False)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"No price data available for {symbol}")
        technicals = compute_all_technicals(df)
        technicals["symbol"] = symbol
        technicals["period"] = period
        _tech_cache[cache_key] = (now, technicals)
        return technicals
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Error computing technicals for %s: %s", symbol, exc)
        raise HTTPException(status_code=500, detail=f"Could not compute technical indicators: {exc}")

@router.get("/{symbol}/fundamentals")
def get_stock_fundamentals(symbol: str):
    """Retrieve fundamental ratios and financial health metrics from yfinance."""
    import time
    now = time.time()
    if symbol in _fund_cache and (now - _fund_cache[symbol][0] < 600):
        return _fund_cache[symbol][1]

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        # Valuation
        pe = info.get("trailingPE") or info.get("forwardPE")
        pb = info.get("priceToBook")
        ev_ebitda = info.get("enterpriseToEbitda")
        peg = info.get("pegRatio")
        market_cap = info.get("marketCap")

        # Profitability
        roe = info.get("returnOnEquity")
        if roe is not None: roe = round(roe * 100, 2)
        roa = info.get("returnOnAssets")
        if roa is not None: roa = round(roa * 100, 2)
        profit_margin = info.get("profitMargins")
        if profit_margin is not None: profit_margin = round(profit_margin * 100, 2)
        operating_margin = info.get("operatingMargins")
        if operating_margin is not None: operating_margin = round(operating_margin * 100, 2)

        # Financial Health
        debt_to_equity = info.get("debtToEquity")
        current_ratio = info.get("currentRatio")
        quick_ratio = info.get("quickRatio")
        dividend_yield = info.get("dividendYield")
        if dividend_yield is not None: dividend_yield = round(dividend_yield * 100, 2)

        # 52-week Range
        fifty_two_high = info.get("fiftyTwoWeekHigh")
        fifty_two_low = info.get("fiftyTwoWeekLow")

        result = {
            "symbol": symbol,
            "company_name": info.get("longName") or info.get("shortName") or symbol,
            "sector": info.get("sector", "Unknown"),
            "industry": info.get("industry", "Unknown"),
            "valuation": {
                "pe_ratio": round(pe, 2) if pe else None,
                "pb_ratio": round(pb, 2) if pb else None,
                "ev_ebitda": round(ev_ebitda, 2) if ev_ebitda else None,
                "peg_ratio": round(peg, 2) if peg else None,
                "market_cap": market_cap,
            },
            "profitability": {
                "roe_pct": roe,
                "roa_pct": roa,
                "net_margin_pct": profit_margin,
                "operating_margin_pct": operating_margin,
            },
            "financial_health": {
                "debt_to_equity": round(debt_to_equity, 2) if debt_to_equity else None,
                "current_ratio": round(current_ratio, 2) if current_ratio else None,
                "quick_ratio": round(quick_ratio, 2) if quick_ratio else None,
                "dividend_yield_pct": dividend_yield,
            },
            "trading_range": {
                "fifty_two_week_high": fifty_two_high,
                "fifty_two_week_low": fifty_two_low,
            },
            "data_source": "yfinance (Public Regulatory Filings & Market Consensus)",
        }
        _fund_cache[symbol] = (now, result)
        return result
    except Exception as exc:
        logger.error("Error fetching fundamentals for %s: %s", symbol, exc)
        return {
            "symbol": symbol,
            "company_name": symbol,
            "valuation": {"pe_ratio": None, "pb_ratio": None, "ev_ebitda": None, "market_cap": None},
            "profitability": {"roe_pct": None, "net_margin_pct": None},
            "financial_health": {"debt_to_equity": None, "current_ratio": None},
            "trading_range": {"fifty_two_week_high": None, "fifty_two_week_low": None},
            "data_source": "Unavailable",
        }
