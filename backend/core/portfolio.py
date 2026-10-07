"""
portfolio.py - Portfolio Evaluation for TradeSense V2
=======================================================
Computes portfolio-level financial metrics using proper covariance-based
risk (σ_p = √(wᵀΣw)), not averaged individual risks.

Formula reference
-----------------
Portfolio return  : R_p = (1/k) Σ R_i
Portfolio risk    : σ_p = √(wᵀ Σ w)   where w = [1/k,...], Σ = annualised covariance
Avg pairwise corr : mean of all C(k,2) off-diagonal pairs
"""

from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Any


ANNUAL_TRADING_DAYS = 252
_RISK_FREE_RATE = 5.0  # %


def covariance_portfolio_volatility(
    portfolio: tuple[str, ...] | list[str],
    returns_df: pd.DataFrame | None = None,
    cov_df: pd.DataFrame | None = None,
) -> float:
    """
    Compute annualised portfolio volatility via covariance matrix.

    σ_p = √(wᵀ Σ w)   where Σ is daily covariance × 252

    Returns annualised percentage (e.g. 18.4 for 18.4%).
    """
    k = len(portfolio)
    if k == 0:
        return 0.0

    if cov_df is not None:
        valid = [t for t in portfolio if t in cov_df.columns]
        if not valid:
            return 0.0
        w = np.full(len(valid), 1.0 / len(valid))
        sub_cov = cov_df.loc[valid, valid].values
        port_var = float(w @ sub_cov @ w)
        return round(float(math.sqrt(max(port_var, 0.0))) * 100.0, 4)

    if returns_df is None or returns_df.empty:
        return 0.0

    valid = [t for t in portfolio if t in returns_df.columns]
    if not valid:
        return 0.0

    w = np.full(len(valid), 1.0 / len(valid))
    sub = returns_df[valid].dropna()
    if sub.shape[0] < 2:
        return 0.0

    cov = sub.cov().values * ANNUAL_TRADING_DAYS  # annualise
    port_var = float(w @ cov @ w)
    return round(float(math.sqrt(max(port_var, 0.0))) * 100.0, 4)


def portfolio_return(
    portfolio: tuple[str, ...] | list[str],
    metrics: list[dict],
) -> float:
    """Equal-weighted average return across portfolio constituents (%)."""
    sym_set = set(portfolio)
    rets = [float(m["return_pct"]) for m in metrics if m["symbol"] in sym_set and m.get("return_pct") is not None]
    if not rets:
        return 0.0
    return round(sum(rets) / len(rets), 4)


def avg_pairwise_correlation(
    portfolio: tuple[str, ...] | list[str],
    corr_df: pd.DataFrame,
) -> float:
    """Mean of all C(k,2) pairwise Pearson correlations in the portfolio."""
    symbols = list(portfolio)
    k = len(symbols)
    if k < 2 or corr_df is None or corr_df.empty:
        return 1.0

    pairs: list[float] = []
    for i in range(k):
        for j in range(i + 1, k):
            a, b = symbols[i], symbols[j]
            if a in corr_df.columns and b in corr_df.columns:
                pairs.append(float(corr_df.loc[a, b]))

    return round(sum(pairs) / len(pairs), 4) if pairs else 1.0


def evaluate_portfolio(
    portfolio: tuple[str, ...] | list[str],
    metrics: list[dict],
    returns_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    cov_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """
    Compute all portfolio-level metrics for one candidate.

    Returns
    -------
    dict: portfolio (list[str]), k, return_pct, risk_pct,
          avg_correlation, valid
    """
    symbols = list(portfolio)
    sym_set = set(symbols)
    valid_m = [m for m in metrics if m["symbol"] in sym_set]

    if len(valid_m) < len(symbols):
        return {"portfolio": symbols, "k": len(symbols), "return_pct": 0.0,
                "risk_pct": 0.0, "avg_correlation": 1.0, "valid": False}

    ret  = portfolio_return(portfolio, metrics)
    risk = covariance_portfolio_volatility(portfolio, returns_df, cov_df=cov_df)
    corr = avg_pairwise_correlation(portfolio, corr_df)

    return {
        "portfolio":       symbols,
        "k":               len(symbols),
        "return_pct":      ret,
        "risk_pct":        risk,
        "avg_correlation": corr,
        "valid":           True,
    }
