"""
Portfolio Combinations & Risk Evaluation Module for TradeSense

Generates candidate portfolios using combinatorics C(n, k), bounds candidate pools to prevent combinatorial explosion, and computes portfolio-level Expected Return and Correlation-Based Portfolio Risk.

Discrete Mathematics & Financial Engineering Concept:
- Combinatorics: Number of size-k portfolios chosen from n candidate stocks without replacement:
  C(n, k) = n! / (k! * (n - k)!)
- Candidate Pool Bounding: Restricts combinatorial generation to a bounded pool size (N <= 25) to prevent combinatorial explosion.
- Equal-Weighted Portfolio Return:
  E(R_p) = (1 / k) * sum(R_i)
- Correlation-Based Portfolio Risk:
  sigma_p^2 = sum_i sum_j (w_i * w_j * sigma_i * sigma_j * A[i, j])
  sigma_p = sqrt(max(sigma_p^2, 0))
  where w_i = 1 / k, sigma_i is stock i's annualized risk, and A[i, j] is Pearson correlation.
"""

import math
import itertools
from typing import List, Tuple, Dict, Any, Optional, Union
import numpy as np
import pandas as pd


def calculate_combination_count(n: int, k: int) -> int:
    """
    Return combination count C(n, k) via math.comb.

    Args:
        n: Number of available items.
        k: Size of combination.

    Returns:
        Integer combination count C(n, k). Returns 0 if n < k or k < 0.
    """
    if n < 0 or k < 0 or n < k:
        return 0
    return math.comb(n, k)


def select_candidate_pool(stock_metrics: pd.DataFrame, pool_size: int = 15) -> List[str]:
    """
    Reduce the eligible stock set to a configurable candidate pool (e.g. top-N by Sharpe Ratio / Return)
    before combination generation to prevent combinatorial explosion.

    Args:
        stock_metrics: DataFrame containing stock metrics.
        pool_size: Target candidate pool size N (default 15).

    Returns:
        List of stock symbols selected for the candidate pool.
    """
    if stock_metrics is None or stock_metrics.empty:
        return []

    # Filter out stocks with invalid / insufficient data
    if "Data Status" in stock_metrics.columns:
        valid_df = stock_metrics[stock_metrics["Data Status"] == "OK"].copy()
    else:
        valid_df = stock_metrics.copy()

    if valid_df.empty:
        valid_df = stock_metrics.copy()

    # Sort by Sharpe Ratio if available, otherwise Return %
    if "Sharpe Ratio" in valid_df.columns:
        sorted_df = valid_df.sort_values(by="Sharpe Ratio", ascending=False)
    elif "Return %" in valid_df.columns:
        sorted_df = valid_df.sort_values(by="Return %", ascending=False)
    elif "CAGR" in valid_df.columns:
        sorted_df = valid_df.sort_values(by="CAGR", ascending=False)
    else:
        sorted_df = valid_df

    target_n = min(max(2, pool_size), len(sorted_df))
    candidate_symbols = sorted_df["Stock"].iloc[:target_n].tolist()
    return candidate_symbols


def generate_candidate_portfolios(candidate_pool: List[str], k: int = 3) -> List[Tuple[str, ...]]:
    """
    Yield all size-k combinations from the candidate pool using itertools.combinations.

    Args:
        candidate_pool: List of stock symbols in the candidate pool.
        k: Portfolio size (default 3).

    Returns:
        List of tuples, where each tuple is a size-k portfolio.
    """
    if not candidate_pool or len(candidate_pool) < k or k <= 0:
        return []

    return list(itertools.combinations(candidate_pool, k))


def validate_portfolio(portfolio: Tuple[str, ...], k: int) -> bool:
    """
    Confirm the portfolio has exactly k distinct stocks with no duplicate symbols.

    Args:
        portfolio: Tuple of stock symbols.
        k: Expected portfolio size.

    Returns:
        True if valid, False otherwise.
    """
    if not portfolio or not isinstance(portfolio, tuple):
        return False

    if len(portfolio) != k:
        return False

    # Check for duplicate stocks
    if len(set(portfolio)) != k:
        return False

    return True


def evaluate_portfolio(
    portfolio: Tuple[str, ...], stock_metrics: pd.DataFrame, correlation_matrix: pd.DataFrame
) -> Dict[str, Any]:
    """
    Compute portfolio-level metrics for a given combination using existing per-stock metrics and correlation matrix:
    - Expected Return E(R_p): Equal-weighted average of individual Returns.
    - Portfolio Risk sigma_p: Computed from individual volatilities and pairwise Pearson correlations
      using standard portfolio variance formula: sigma_p^2 = sum_i sum_j (w_i w_j sigma_i sigma_j A[i, j]).

    Args:
        portfolio: Tuple of k stock symbols.
        stock_metrics: DataFrame containing per-stock metrics ('Stock', 'Return %', 'Risk %').
        correlation_matrix: Square Pearson correlation DataFrame.

    Returns:
        Dictionary containing portfolio evaluation metrics.
    """
    k = len(portfolio)
    empty_res = {
        "portfolio": portfolio,
        "k": k,
        "expected_return": 0.0,
        "portfolio_risk": 0.0,
        "sharpe_ratio": 0.0,
        "return_risk_ratio": 0.0,
        "valid": False,
    }

    if not validate_portfolio(portfolio, k) or stock_metrics is None or stock_metrics.empty:
        return empty_res

    # Extract metrics dict per stock
    if "Stock" in stock_metrics.columns:
        metrics_map = {row["Stock"]: row for _, row in stock_metrics.iterrows()}
    else:
        metrics_map = {idx: row for idx, row in stock_metrics.iterrows()}

    returns = []
    risks = []
    for s in portfolio:
        if s not in metrics_map:
            return empty_res
        m = metrics_map[s]
        ret_v = float(m.get("Return %", m.get("CAGR", 0.0)))
        risk_v = float(m.get("Risk %", m.get("Volatility", 0.0)))
        returns.append(ret_v)
        risks.append(risk_v)

    # 1. Equal-weighted expected return
    w = 1.0 / float(k)
    expected_ret = float(sum(returns) * w)

    # 2. Portfolio Risk using Correlation Matrix
    var_sum = 0.0
    for i in range(k):
        for j in range(k):
            sym_i = portfolio[i]
            sym_j = portfolio[j]

            sigma_i = risks[i]
            sigma_j = risks[j]

            # Fetch Pearson correlation from matrix (1.0 for i == j)
            if correlation_matrix is not None and not correlation_matrix.empty and sym_i in correlation_matrix.columns and sym_j in correlation_matrix.columns:
                corr_ij = float(correlation_matrix.loc[sym_i, sym_j])
            else:
                corr_ij = 1.0 if i == j else 0.0

            var_sum += w * w * sigma_i * sigma_j * corr_ij

    portfolio_risk = float(math.sqrt(max(var_sum, 0.0)))
    ratio = float(expected_ret / (portfolio_risk + 1e-6)) if portfolio_risk > 0 else 0.0
    sharpe = float((expected_ret - 5.0) / (portfolio_risk + 1e-6)) if portfolio_risk > 0 else 0.0

    return {
        "portfolio": portfolio,
        "k": k,
        "expected_return": round(expected_ret, 2),
        "portfolio_risk": round(portfolio_risk, 2),
        "sharpe_ratio": round(sharpe, 2),
        "return_risk_ratio": round(ratio, 2),
        "valid": True,
    }


def rank_portfolios(evaluated_portfolios: List[Dict[str, Any]], sort_by: str = "sharpe") -> List[Dict[str, Any]]:
    """
    Sort evaluated portfolios by the chosen ranking criterion and assign rank positions.

    Args:
        evaluated_portfolios: List of portfolio evaluation dicts.
        sort_by: Criterion ('sharpe', 'return_risk_ratio', 'return', 'risk').

    Returns:
        List of ranked portfolio dicts with 'rank' position added.
    """
    if not evaluated_portfolios:
        return []

    valid_portfolios = [p for p in evaluated_portfolios if p.get("valid", True)]
    if not valid_portfolios:
        return []

    if sort_by == "return":
        key_fn = lambda x: x.get("expected_return", 0.0)
        reverse_flag = True
    elif sort_by == "risk":
        key_fn = lambda x: x.get("portfolio_risk", 999.0)
        reverse_flag = False
    elif sort_by == "return_risk_ratio":
        key_fn = lambda x: x.get("return_risk_ratio", 0.0)
        reverse_flag = True
    else:  # 'sharpe'
        key_fn = lambda x: x.get("sharpe_ratio", 0.0)
        reverse_flag = True

    sorted_list = sorted(valid_portfolios, key=key_fn, reverse=reverse_flag)

    ranked_result = []
    for idx, item in enumerate(sorted_list, start=1):
        item_copy = dict(item)
        item_copy["rank"] = idx
        ranked_result.append(item_copy)

    return ranked_result


# Legacy helper compatibility function
def generate_diversified_portfolios(coloring_dict: Dict[str, int], k_stocks: int = 3) -> List[Tuple[str, ...]]:
    """Legacy helper compatibility."""
    if not coloring_dict:
        return []

    groups = {}
    for stock, color_id in coloring_dict.items():
        if color_id not in groups:
            groups[color_id] = []
        groups[color_id].append(stock)

    rep_stocks = [stocks[0] for stocks in groups.values()]
    if len(rep_stocks) >= k_stocks:
        return generate_candidate_portfolios(rep_stocks, k=k_stocks)
    
    all_stocks = list(coloring_dict.keys())
    return generate_candidate_portfolios(all_stocks[:15], k=k_stocks)
