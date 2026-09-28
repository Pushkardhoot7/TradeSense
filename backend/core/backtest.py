"""
backtest.py - Portfolio Backtesting & Risk Analytics Engine
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd

def calculate_sortino(returns: pd.Series, risk_free_rate: float = 5.0) -> float:
    rf_daily = (1.0 + risk_free_rate / 100.0) ** (1.0 / 252.0) - 1.0
    excess_returns = returns - rf_daily
    downside = returns[returns < rf_daily]
    if len(downside) < 2:
        return 0.0
    downside_dev = np.sqrt(np.mean(downside ** 2)) * np.sqrt(252)
    annualized_excess = excess_returns.mean() * 252
    return round(float(annualized_excess / downside_dev), 2) if downside_dev > 0 else 0.0

def calculate_max_drawdown(equity_curve: list[float]) -> float:
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        if val > peak:
            peak = val
        dd = (peak - val) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return round(float(max_dd * 100), 2)

def calculate_var(returns: pd.Series, confidence: float = 0.95) -> float:
    if len(returns) < 10:
        return 0.0
    var_daily = np.percentile(returns, (1 - confidence) * 100)
    return round(float(abs(var_daily) * 100), 2)

def run_portfolio_backtest(
    price_dfs: dict[str, pd.DataFrame],
    weights: dict[str, float],
    initial_capital: float = 100000.0,
    risk_free_rate: float = 5.0,
) -> dict[str, Any]:
    symbols = list(weights.keys())
    close_series = {}
    for sym in symbols:
        if sym in price_dfs and "Close" in price_dfs[sym].columns:
            close = price_dfs[sym]["Close"].dropna()
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close_series[sym] = close

    if not close_series:
        return {"error": "No price data available for backtest constituents."}

    prices_df = pd.DataFrame(close_series).dropna()
    if len(prices_df) < 5:
        return {"error": "Insufficient overlapping trading days."}

    daily_rets = prices_df.pct_change().dropna()
    total_w = sum(weights.values())
    norm_weights = {s: weights[s] / total_w for s in symbols if s in daily_rets.columns}
    w_series = pd.Series(norm_weights)

    portfolio_daily_rets = (daily_rets * w_series).sum(axis=1)
    cum_rets = (1 + portfolio_daily_rets).cumprod()
    equity_curve = [round(float(initial_capital * r), 2) for r in cum_rets]
    dates = [str(d.date()) if hasattr(d, "date") else str(d) for d in cum_rets.index]

    final_val = equity_curve[-1] if equity_curve else initial_capital
    total_return_pct = ((final_val - initial_capital) / initial_capital) * 100.0
    n_days = len(portfolio_daily_rets)
    cagr = (((final_val / initial_capital) ** (252.0 / n_days)) - 1.0) * 100.0 if n_days > 0 else 0.0
    ann_vol = float(portfolio_daily_rets.std() * np.sqrt(252) * 100.0)
    sharpe = round((cagr - risk_free_rate) / ann_vol, 2) if ann_vol > 0 else 0.0
    sortino = calculate_sortino(portfolio_daily_rets, risk_free_rate)
    max_dd = calculate_max_drawdown(equity_curve)
    var95 = calculate_var(portfolio_daily_rets, 0.95)

    return {
        "initial_capital": initial_capital,
        "final_capital": final_val,
        "total_return_pct": round(total_return_pct, 2),
        "cagr_pct": round(cagr, 2),
        "annualized_volatility_pct": round(ann_vol, 2),
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown_pct": max_dd,
        "var_95_daily_pct": var95,
        "trading_days": n_days,
        "equity_curve": equity_curve,
        "dates": dates,
        "disclaimer": "Past backtested performance is historical simulation and does not guarantee future results."
    }

def run_scenario_stress_test(
    symbols: list[str],
    weights: dict[str, float],
    scenarios: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if not scenarios:
        scenarios = [
            {"id": "crash_10", "name": "Broad Market Correction", "shock_pct": -10.0, "scope": "market"},
            {"id": "it_shock", "name": "Tech Sector Pullback", "shock_pct": -15.0, "scope": "sector_it"},
            {"id": "bank_shock", "name": "Credit & Banking Squeeze", "shock_pct": -12.0, "scope": "sector_bank"},
            {"id": "vol_spike", "name": "Volatility Spike (VIX > 25)", "shock_pct": -6.5, "scope": "market"},
            {"id": "bull_run", "name": "Macro Liquidity Expansion", "shock_pct": 8.0, "scope": "market"},
        ]

    total_w = sum(weights.values()) if weights else 1.0
    norm_w = {s: weights.get(s, 1.0 / len(symbols)) / total_w for s in symbols}

    results = []
    for sc in scenarios:
        shock = sc["shock_pct"]
        contribs = []
        for s in symbols:
            w = norm_w.get(s, 0.0)
            st_shock = shock
            if sc["scope"] == "sector_it" and not any(t in s for t in ["TCS", "INFY", "WIPRO", "HCL", "TECHM", "LTIM"]):
                st_shock = shock * 0.2
            elif sc["scope"] == "sector_bank" and not any(t in s for t in ["BANK", "HDFC", "ICICI", "KOTAK", "AXIS", "SBIN"]):
                st_shock = shock * 0.2
            c = w * st_shock
            contribs.append({"symbol": s, "weight_pct": round(w * 100, 1), "shock_pct": round(st_shock, 1), "contribution_pct": round(c, 2)})

        total_impact = sum(item["contribution_pct"] for item in contribs)
        results.append({
            "scenario": sc["name"],
            "scope": sc["scope"],
            "base_shock_pct": shock,
            "estimated_portfolio_impact_pct": round(total_impact, 2),
            "asset_breakdown": contribs,
        })

    return results
