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
        symbols = req.symbols
        weights = req.weights or {s: 1.0 / len(symbols) for s in symbols}

        # Fetch historical closes:
        # 1) Pipeline cached price_data
        price_dfs = {}
        from backend.routers.analysis import _last_result
        cached_prices = _last_result.get("price_data", {})
        for s in symbols:
            if s in cached_prices and hasattr(cached_prices[s], "empty") and not cached_prices[s].empty:
                price_dfs[s] = cached_prices[s]

        # 2) LiveDataProvider (handles alias mapping and disk cache)
        missing = [s for s in symbols if s not in price_dfs]
        if missing:
            try:
                from backend.providers.live_provider import LiveDataProvider
                from backend.pipeline import _period_to_dates
                live_p = LiveDataProvider()
                start_d, end_d = _period_to_dates(req.period)
                for s in missing:
                    df = live_p.get_historical_data(s, start_d, end_d, "1d")
                    if df is not None and not df.empty:
                        price_dfs[s] = df
            except Exception as e:
                logger.warning("LiveDataProvider fallback failed: %s", e)

        # 3) DemoDataProvider fallback
        missing_demo = [s for s in symbols if s not in price_dfs]
        if missing_demo:
            try:
                from backend.providers.demo_provider import DemoDataProvider
                from backend.pipeline import _period_to_dates
                demo = DemoDataProvider()
                start_d, end_d = _period_to_dates(req.period)
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
