"""
DM Portfolio Scoring Module for TradeSense

Ranks generated portfolios using a Discrete Mathematics composite score combining return, covariance risk, correlation diversification, graph group independence, and poset dominance.
Integrates modules.dm_scoring engine.
"""

from typing import Dict, List, Any, Tuple, Optional
import pandas as pd
from modules.dm_scoring import run_dm_portfolio_scoring, DM_SCORE_WEIGHTS, DISCLAIMER


def score_portfolio(
    portfolio_tickers: List[str],
    metrics_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    coloring_dict: Dict[str, int]
) -> Dict[str, Any]:
    """
    Calculate DM Portfolio Score based on composite 5-component DM scoring engine.

    Args:
        portfolio_tickers: List of stock tickers in candidate portfolio.
        metrics_df: Financial metrics (CAGR, Volatility).
        corr_df: Correlation matrix.
        coloring_dict: Graph coloring map.

    Returns:
        Dictionary containing composite DM Portfolio Score and mathematical metrics breakdown.
    """
    if not portfolio_tickers or metrics_df.empty:
        return {"score": 0.0, "cagr": 0.0, "volatility": 0.0, "sharpe": 0.0, "avg_corr": 0.0}

    # Delegate to DM scoring engine for single portfolio candidate
    res = run_dm_portfolio_scoring(
        candidate_portfolios=[tuple(portfolio_tickers)],
        stock_metrics=metrics_df,
        returns_matrix=pd.DataFrame(),
        corr_df=corr_df,
        coloring_dict=coloring_dict,
        top_n=1,
    )

    top_p = res["top_portfolios"][0] if res["top_portfolios"] else None
    if top_p:
        return {
            "score": top_p["dm_score"],
            "cagr": top_p["raw_metrics"]["raw_return"],
            "volatility": top_p["raw_metrics"]["raw_risk"],
            "sharpe": top_p["component_scores"]["return_score"],
            "avg_corr": top_p["raw_metrics"]["raw_avg_corr"],
            "independence_score": top_p["component_scores"]["group_diversity_score"],
            "component_scores": top_p["component_scores"],
            "disclaimer": DISCLAIMER,
        }

    return {"score": 0.0, "cagr": 0.0, "volatility": 0.0, "sharpe": 0.0, "avg_corr": 0.0}


def rank_top_portfolios(
    portfolios: List[List[str]],
    metrics_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    coloring_dict: Dict[str, int],
    top_n: int = 3,
    returns_matrix: Optional[pd.DataFrame] = None
) -> List[Dict[str, Any]]:
    """
    Rank candidate portfolios using deterministic DM Portfolio Scoring Engine and return Top N insights.

    Args:
        portfolios: List of candidate portfolio ticker lists.
        metrics_df: Financial metrics dataframe.
        corr_df: Correlation matrix.
        coloring_dict: Graph coloring dict.
        top_n: Number of top portfolios to return (default 3).
        returns_matrix: Historical daily returns DataFrame for covariance calculation.

    Returns:
        List of dictionaries with portfolio tickers, score, and mathematical justification.
    """
    if not portfolios:
        return []

    cand_tuples = [tuple(p) for p in portfolios]
    if returns_matrix is None:
        returns_matrix = pd.DataFrame()

    res = run_dm_portfolio_scoring(
        candidate_portfolios=cand_tuples,
        stock_metrics=metrics_df,
        returns_matrix=returns_matrix,
        corr_df=corr_df,
        coloring_dict=coloring_dict,
        top_n=top_n,
    )

    results = []
    for item in res["top_portfolios"]:
        results.append({
            "rank": item["rank"],
            "portfolio": list(item["portfolio"]),
            "score": item["dm_score"],
            "cagr": item["raw_metrics"]["raw_return"] / 100.0 if item["raw_metrics"]["raw_return"] > 1.0 else item["raw_metrics"]["raw_return"],
            "volatility": item["raw_metrics"]["raw_risk"] / 100.0 if item["raw_metrics"]["raw_risk"] > 1.0 else item["raw_metrics"]["raw_risk"],
            "sharpe": item["component_scores"]["return_score"],
            "avg_corr": item["raw_metrics"]["raw_avg_corr"],
            "independence_score": item["component_scores"]["group_diversity_score"],
            "rationale": item["rationale"],
            "component_scores": item["component_scores"],
            "raw_metrics": item["raw_metrics"],
            "disclaimer": DISCLAIMER,
        })

    return results
