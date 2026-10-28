"""
backtest_router.py - Backtesting & Scenario Stress-Testing Endpoints
"""
from __future__ import annotations
import logging
from typing import Any
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
from backend.core.backtest import run_portfolio_backtest, run_scenario_stress_test

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/portfolio", tags=["Portfolio Lab & Backtesting"])


class BacktestRequest(BaseModel):
    symbols: list[str] = Field(min_length=1)
    weights: dict[str, float] | None = None
    period: str = Field(default="1y")
    initial_capital: float = Field(default=100000.0)


class ScenarioRequest(BaseModel):
    symbols: list[str] = Field(min_length=1)
    weights: dict[str, float] | None = None


@router.post("/backtest")
def backtest_portfolio(req: BacktestRequest):
    """Run historical simulation backtest for a portfolio."""
    try:
        import pandas as pd
        from backend.pipeline import _period_to_dates

        symbols = req.symbols
        weights = req.weights or {s: 1.0 / len(symbols) for s in symbols}
        period = req.period or "1y"
        start_d, end_d = _period_to_dates(period)
        start_ts = pd.to_datetime(start_d)

        # Fetch historical closes:
        # 1) Pipeline cached price_data (if it has sufficient history)
        price_dfs = {}
        from backend.routers.analysis import _last_result
        cached_prices = _last_result.get("price_data", {})

        for s in symbols:
            if s in cached_prices and hasattr(cached_prices[s], "empty") and not cached_prices[s].empty:
                cdf = cached_prices[s]
                min_date = cdf.index.min()
                if cdf.index.tz is not None:
                    tz_start = start_ts.tz_localize(cdf.index.tz) if start_ts.tz is None else start_ts.tz_convert(cdf.index.tz)
                else:
                    tz_start = start_ts.tz_localize(None) if start_ts.tz is not None else start_ts

                # If cached df has data starting near start_d, slice it
                if min_date <= tz_start or (min_date - tz_start).days <= 7:
                    sliced = cdf.loc[cdf.index >= tz_start].copy()
                    if len(sliced) >= 5:
                        price_dfs[s] = sliced

        # 2) LiveDataProvider (handles alias mapping and disk cache) for missing or longer periods
        missing = [s for s in symbols if s not in price_dfs]
        if missing:
            try:
                from backend.providers.live_provider import LiveDataProvider
                live_p = LiveDataProvider()
                for s in missing:
                    df = live_p.get_historical_data(s, start_d, end_d, interval="1d", period=period)
                    if df is not None and not df.empty:
                        if df.index.tz is not None:
                            tz_start = start_ts.tz_localize(df.index.tz) if start_ts.tz is None else start_ts.tz_convert(df.index.tz)
                        else:
                            tz_start = start_ts.tz_localize(None) if start_ts.tz is not None else start_ts
                        min_date = df.index.min()
                        # If df does not have sufficient history for the requested period (e.g. 1y disk cache for 2y), skip to let demo provider handle
                        if (min_date - tz_start).days > 45 and period in ["2y", "3y", "5y"]:
                            continue

                        sliced = df.loc[df.index >= tz_start].copy()
                        price_dfs[s] = sliced if len(sliced) >= 5 else df
            except Exception as e:
                logger.warning("LiveDataProvider fallback failed: %s", e)

        # 3) DemoDataProvider fallback
        missing_demo = [s for s in symbols if s not in price_dfs]
        if missing_demo:
            try:
                from backend.providers.demo_provider import DemoDataProvider
                demo = DemoDataProvider()
                for s in missing_demo:
                    df = demo.get_historical_data(s, start_d, end_d, "1d")
                    if df is not None and not df.empty:
                        price_dfs[s] = df
            except Exception as e:
                logger.warning("DemoDataProvider fallback failed: %s", e)

        if not price_dfs:
            raise HTTPException(status_code=400, detail="Could not retrieve historical data for symbols.")

        results = run_portfolio_backtest(
            price_dfs=price_dfs,
            weights=weights,
            initial_capital=req.initial_capital
        )
        return results
    except Exception as exc:
        logger.error("Backtest failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Backtest error: {exc}")


@router.post("/scenario")
def stress_test_portfolio(req: ScenarioRequest):
    """Run what-if macro and sector shock scenarios on a portfolio."""
    try:
        symbols = req.symbols
        weights = req.weights or {s: 1.0 / len(symbols) for s in symbols}
        scenarios = run_scenario_stress_test(symbols=symbols, weights=weights)
        return {
            "portfolio": symbols,
            "weights": weights,
            "scenarios": scenarios,
            "disclaimer": "Scenario analysis uses linear factor exposures and historical beta approximations. Not a predictive guarantee."
        }
    except Exception as exc:
        logger.error("Scenario stress test failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Scenario error: {exc}")
