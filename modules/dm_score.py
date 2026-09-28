"""
modules/dm_score.py - Automatic DM Portfolio Score Engine for TradeSense
========================================================================

Capstone analytical scoring system that combines outputs from multiple
discrete mathematics modules (Financial Metrics, Pearson Correlation,
Welsh-Powell Graph Coloring, Dominance Relations & Posets) into a single,
reproducible, fully-automatic ranking used to select the Top 3 candidate
portfolios with zero human intervention or manual curation.

Mathematical Concept & Formula
------------------------------
DM Score is a project-defined mathematical composite analytical score:

    DM Score = 0.30 × Return Score
             + 0.25 × Risk Score
             + 0.25 × Diversification Score
             + 0.10 × Graph Group Diversity Score
             + 0.10 × Dominance Score

All five component scores are normalized to the range [0, 100].

Component Definitions & Normalization Rules:
1. Return Score (30%):
   Expected return from portfolio evaluation, min-max normalized across
   the currently evaluated candidate pool:
   Return Score = 100 × (return − min_return) / (max_return − min_return)
   (Zero-division safeguard: if max_return == min_return, assign 50.0).

2. Risk Score (25%):
   Covariance-based portfolio volatility from portfolio evaluation (sigma_p = sqrt(w^T Sigma w)),
   inverted min-max normalized across candidate pool (lower risk -> higher score):
   Risk Score = 100 × (max_risk − risk) / (max_risk − min_risk)
   (Zero-division safeguard: if max_risk == min_risk, assign 50.0).

3. Diversification Score (25%):
   Average pairwise correlation (AvgCorr) among constituent stocks within the portfolio,
   mapped over fixed absolute Pearson range [-1, +1]:
   Diversification Score = 100 × (1 − AvgCorr) / 2
   (AvgCorr = -1 -> 100.0; AvgCorr = 0 -> 50.0; AvgCorr = +1 -> 0.0).

4. Graph Group Diversity Score (10%):
   Representation across distinct Welsh-Powell graph color groups:
   Graph Group Diversity Score = 100 × (distinct_colors − 1) / (min(k, total_colors) − 1)
   (Safeguard: if min(k, total_colors) <= 1, assign 100.0).

   *** Mathematical Caveat on Graph Group Diversity Score ***
   A proper graph coloring guarantees that stocks sharing the same color
   have correlation below the threshold (they form an independent set) —
   it does NOT guarantee that stocks in different color groups are correlated.
   This score therefore measures representation across the graph's structural
   clusters, not a direct correlation guarantee; the Diversification Score
   captures actual pairwise correlation directly, and the two are intentionally
   distinct signals.

5. Dominance Score (10%):
   Average normalized out-degree in the dominance partial order relation:
   normalized_dominance(stock) = |dominated_by(stock)| / (N − 1)
   Dominance Score = 100 × average(normalized_dominance(stock) for stock in portfolio)
   where N is the number of stocks in the current evaluated candidate pool.

Academic Language Standard:
- Call it: "Project-defined mathematical analytical score"
- Never call it: "Guaranteed best portfolio"
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from modules.portfolio import evaluate_portfolio, validate_portfolio
from modules.relations import get_dominated_stocks

# Default weights (must sum to 1.0 / 100%)
DEFAULT_DM_WEIGHTS: Dict[str, float] = {
    "return": 0.30,
    "risk": 0.25,
    "diversification": 0.25,
    "group_diversity": 0.10,
    "dominance": 0.10,
}

DISCLAIMER: str = (
    "DM Score is a project-defined mathematical analytical score. "
    "It is NOT a guaranteed best portfolio — it reflects a specific, "
    "documented weighting of return, risk, diversification, and dominance "
    "characteristics and should be interpreted accordingly."
)


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Helper to convert value to safe float without NaN/inf."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Component Score 1: Return Score
# ---------------------------------------------------------------------------

def calculate_return_score(candidate_portfolios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Attach a normalized Return Score [0, 100] to each evaluated portfolio.

    Formula:
        Return Score = 100 × (return − min_return) / (max_return − min_return)

    If max_return == min_return (e.g. single candidate or all tied),
    assigns all portfolios a score of 50.0 to prevent division by zero.

    Args:
        candidate_portfolios: List of portfolio dictionaries containing
            'expected_return', 'return', or 'return_pct'.

    Returns:
        List of portfolio dictionaries with 'return_score' attached.
    """
    if not candidate_portfolios:
        return []

    returns = [
        _safe_float(p.get("expected_return", p.get("return", p.get("return_pct", 0.0))))
        for p in candidate_portfolios
    ]

    min_ret = min(returns)
    max_ret = max(returns)

    scored: List[Dict[str, Any]] = []
    if math.isclose(max_ret, min_ret, abs_tol=1e-9):
        for p, ret in zip(candidate_portfolios, returns):
            item = dict(p)
            item["return_score"] = 50.0
            item["raw_return"] = ret
            scored.append(item)
    else:
        denom = max_ret - min_ret
        for p, ret in zip(candidate_portfolios, returns):
            item = dict(p)
            score = 100.0 * (ret - min_ret) / denom
            item["return_score"] = round(max(0.0, min(100.0, float(score))), 4)
            item["raw_return"] = ret
            scored.append(item)

    return scored


# ---------------------------------------------------------------------------
# Component Score 2: Risk Score
# ---------------------------------------------------------------------------

def calculate_risk_score(candidate_portfolios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Attach a normalized Risk Score [0, 100] to each evaluated portfolio.
    Uses covariance-based portfolio volatility from evaluate_portfolio().
    Lower risk yields a higher score (inverted min-max).

    Formula:
        Risk Score = 100 × (max_risk − risk) / (max_risk − min_risk)

    If max_risk == min_risk (e.g. single candidate or all tied),
    assigns all portfolios a score of 50.0 to prevent division by zero.

    Args:
        candidate_portfolios: List of portfolio dictionaries containing
            'portfolio_risk', 'risk', or 'risk_pct'.

    Returns:
        List of portfolio dictionaries with 'risk_score' attached.
    """
    if not candidate_portfolios:
        return []

    risks = [
        _safe_float(p.get("portfolio_risk", p.get("risk", p.get("risk_pct", 0.0))))
        for p in candidate_portfolios
    ]

    min_risk = min(risks)
    max_risk = max(risks)

    scored: List[Dict[str, Any]] = []
    if math.isclose(max_risk, min_risk, abs_tol=1e-9):
        for p, risk in zip(candidate_portfolios, risks):
            item = dict(p)
            item["risk_score"] = 50.0
            item["raw_risk"] = risk
            scored.append(item)
    else:
        denom = max_risk - min_risk
        for p, risk in zip(candidate_portfolios, risks):
            item = dict(p)
            score = 100.0 * (max_risk - risk) / denom
            item["risk_score"] = round(max(0.0, min(100.0, float(score))), 4)
            item["raw_risk"] = risk
            scored.append(item)

    return scored


# ---------------------------------------------------------------------------
# Component Score 3: Diversification Score
# ---------------------------------------------------------------------------

def calculate_diversification_score(
    portfolio: Tuple[str, ...],
    correlation_matrix: Optional[pd.DataFrame] = None,
) -> float:
    """
    Compute average pairwise correlation (AvgCorr) among stocks within this
    specific portfolio, and map onto fixed Pearson range [-1, +1].

    Formula:
        Diversification Score = 100 × (1 − AvgCorr) / 2

    Scale:
        AvgCorr = -1.0 -> 100.0 (maximum diversification)
        AvgCorr =  0.0 ->  50.0 (uncorrelated)
        AvgCorr = +1.0 ->   0.0 (perfect positive co-movement)

    Args:
        portfolio: Tuple of stock symbols.
        correlation_matrix: Symmetric N×N Pearson correlation DataFrame.

    Returns:
        Diversification Score in [0, 100].
    """
    stocks = list(portfolio)
    k = len(stocks)
    if k < 2 or correlation_matrix is None or correlation_matrix.empty:
        # Single stock or missing correlation: AvgCorr = 1.0 -> Score = 0.0
        return 0.0

    pair_corrs: List[float] = []
    cols = set(correlation_matrix.columns)

    for i in range(k):
        for j in range(i + 1, k):
            s_i, s_j = stocks[i], stocks[j]
            if s_i in cols and s_j in cols:
                c = _safe_float(correlation_matrix.loc[s_i, s_j], default=0.0)
                # Bounded Pearson sanity check
                c = max(-1.0, min(1.0, c))
                pair_corrs.append(c)

    if not pair_corrs:
        return 50.0

    avg_corr = float(sum(pair_corrs) / len(pair_corrs))
    score = 100.0 * (1.0 - avg_corr) / 2.0
    return round(max(0.0, min(100.0, float(score))), 4)


# ---------------------------------------------------------------------------
# Component Score 4: Graph Group Diversity Score
# ---------------------------------------------------------------------------

def calculate_group_diversity_score(
    portfolio: Tuple[str, ...],
    color_groups: Union[Dict[int, List[str]], Dict[str, int]],
    k: Optional[int] = None,
    total_colors: Optional[int] = None,
) -> float:
    """
    Compute the Graph Group Diversity Score based on Welsh-Powell color groups.

    Formula:
        Graph Group Diversity Score = 100 × (distinct_colors − 1) / (min(k, total_colors) − 1)
        If min(k, total_colors) <= 1, assigns 100.0.

    Mathematical Caveat:
        A proper coloring guarantees that stocks sharing the same color have
        correlation below the threshold (they form an independent set) — it
        does NOT guarantee that stocks in different color groups are correlated.
        This score measures representation across the graph's structural clusters,
        not a direct correlation guarantee; the Diversification Score captures
        actual pairwise correlation directly, and the two are distinct signals.

    Args:
        portfolio: Tuple of stock symbols.
        color_groups: Mapping of {color_id: [stocks]} OR {stock: color_id}.
        k: Portfolio size (defaults to len(portfolio)).
        total_colors: Total distinct color groups in graph.

    Returns:
        Graph Group Diversity Score in [0, 100].
    """
    stocks = list(portfolio)
    if not stocks:
        return 0.0

    if k is None or k <= 0:
        k = len(stocks)

    # Normalize color mapping to {stock: color_id}
    stock_to_color: Dict[str, int] = {}
    if color_groups:
        sample_key = next(iter(color_groups.keys()))
        sample_val = next(iter(color_groups.values()))

        if isinstance(sample_val, (list, tuple, set)):
            # color_id -> [stocks]
            for c_id, s_list in color_groups.items():
                for s in s_list:
                    stock_to_color[s] = int(c_id)
            if total_colors is None:
                total_colors = len(color_groups)
        else:
            # stock -> color_id
            for s, c_id in color_groups.items():
                stock_to_color[s] = int(c_id)
            if total_colors is None:
                total_colors = len(set(color_groups.values()))

    if total_colors is None or total_colors <= 0:
        total_colors = max(len(set(stock_to_color.values())), 1)

    # Find distinct colors present in this portfolio
    represented_colors = {stock_to_color[s] for s in stocks if s in stock_to_color}
    distinct_count = len(represented_colors) if represented_colors else 1

    max_possible_distinct = min(k, total_colors)

    if max_possible_distinct <= 1:
        return 100.0

    denom = float(max_possible_distinct - 1)
    numer = float(distinct_count - 1)
    score = 100.0 * (numer / denom)
    return round(max(0.0, min(100.0, float(score))), 4)


# ---------------------------------------------------------------------------
# Component Score 5: Dominance Score
# ---------------------------------------------------------------------------

def calculate_dominance_score(
    portfolio: Tuple[str, ...],
    dominance_relation: List[Tuple[str, str]],
    pool_size: int,
) -> float:
    """
    Compute the Dominance Score per Section 3.

    For each stock in the portfolio, count how many other stocks (in the current
    evaluated pool) it dominates, normalize by (N - 1), and average across the portfolio:

        normalized_dominance(stock) = |dominated_by(stock)| / (N − 1)
        Dominance Score = 100 × average(normalized_dominance(stock) for stock in portfolio)

    where N is pool_size.

    Args:
        portfolio: Tuple of stock symbols.
        dominance_relation: List of dominance pairs (A, B) where A dominates B.
        pool_size: Number of stocks in current evaluated pool (N).

    Returns:
        Dominance Score in [0, 100].
    """
    stocks = list(portfolio)
    if not stocks:
        return 0.0

    if pool_size <= 1:
        return 100.0

    # Build out-degree map: stock -> count of stocks it dominates
    out_degrees: Dict[str, int] = {}
    if dominance_relation:
        for dominator, dominated in dominance_relation:
            out_degrees[dominator] = out_degrees.get(dominator, 0) + 1

    denom = float(pool_size - 1)
    norm_scores: List[float] = []

    for s in stocks:
        dom_count = out_degrees.get(s, 0)
        norm_val = min(1.0, float(dom_count) / denom)
        norm_scores.append(norm_val)

    avg_norm = float(sum(norm_scores) / len(norm_scores))
    score = 100.0 * avg_norm
    return round(max(0.0, min(100.0, float(score))), 4)


# ---------------------------------------------------------------------------
# DM Score Combiner & Validation
# ---------------------------------------------------------------------------

def validate_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Validate that weights dictionary contains non-negative numbers summing to 1.0 (or 100%).
    Raises ValueError if invalid. Never silently renormalizes invalid weights.

    Returns:
        Normalized weights dictionary summing to 1.0.
    """
    if not weights or not isinstance(weights, dict):
        raise ValueError("Weights must be a non-empty dictionary.")

    keys = ["return", "risk", "diversification", "group_diversity", "dominance"]
    cleaned: Dict[str, float] = {}

    for k in keys:
        # Handle variations in naming (e.g. 'group', 'graph_group_diversity')
        val = None
        for alias in [k, f"{k}_score", k.replace("_diversity", ""), "group"]:
            if alias in weights:
                val = weights[alias]
                break
        if val is None:
            raise ValueError(f"Missing required weight key: '{k}'")

        f_val = _safe_float(val, default=-1.0)
        if f_val < 0.0:
            raise ValueError(f"Weight for '{k}' must be non-negative, got {val}")
        cleaned[k] = f_val

    total = sum(cleaned.values())

    # Check 1.0 scale
    if math.isclose(total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        return cleaned

    # Check 100.0 scale
    if math.isclose(total, 100.0, rel_tol=1e-3, abs_tol=1e-3):
        return {k: v / 100.0 for k, v in cleaned.items()}

    raise ValueError(
        f"Weights must sum to 100% (or 1.0). Current sum: {total:.4f}"
    )


def calculate_dm_score(
    component_scores: Dict[str, float],
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Combine the five component scores using the weighted formula in Section 2.

    Must validate that weights sum to 100% (or 1.0) before computing;
    raises ValueError otherwise — never silently renormalizes without error.

    Formula:
        DM Score = w_return * S_return
                 + w_risk * S_risk
                 + w_div * S_div
                 + w_group * S_group
                 + w_dom * S_dom

    Args:
        component_scores: Dict containing component scores:
            'return_score', 'risk_score', 'diversification_score',
            'group_diversity_score', 'dominance_score'.
        weights: Optional dictionary of weights. Defaults to DEFAULT_DM_WEIGHTS.

    Returns:
        Final DM Score in [0, 100] rounded to 4 decimal places.
    """
    if weights is None:
        valid_weights = DEFAULT_DM_WEIGHTS
    else:
        valid_weights = validate_weights(weights)

    s_ret = _safe_float(component_scores.get("return_score", component_scores.get("return", 0.0)))
    s_risk = _safe_float(component_scores.get("risk_score", component_scores.get("risk", 0.0)))
    s_div = _safe_float(component_scores.get("diversification_score", component_scores.get("diversification", 0.0)))
    s_group = _safe_float(component_scores.get("group_diversity_score", component_scores.get("group_score", component_scores.get("group_diversity", 0.0))))
    s_dom = _safe_float(component_scores.get("dominance_score", component_scores.get("dominance", 0.0)))

    dm_score = (
        valid_weights["return"] * s_ret
        + valid_weights["risk"] * s_risk
        + valid_weights["diversification"] * s_div
        + valid_weights["group_diversity"] * s_group
        + valid_weights["dominance"] * s_dom
    )

    return round(max(0.0, min(100.0, float(dm_score))), 4)


# ---------------------------------------------------------------------------
# Ranking and Top-N Selection
# ---------------------------------------------------------------------------

def rank_portfolios(scored_portfolios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort portfolios deterministically by DM Score descending.

    Tiebreaker order:
    1. Higher DM Score
    2. Higher Return Score
    3. Higher Risk Score (lower risk volatility)
    4. Original candidate insertion order (stable sort)

    Assigns 1-based 'rank' attribute to each portfolio dictionary.

    Args:
        scored_portfolios: List of scored portfolio dictionaries.

    Returns:
        Ranked list of portfolio dictionaries with 'rank' (1, 2, 3...) assigned.
    """
    if not scored_portfolios:
        return []

    # Prepare decorated list with original index for stable tiebreaking
    decorated = []
    for idx, p in enumerate(scored_portfolios):
        dm = _safe_float(p.get("dm_score", 0.0))
        ret = _safe_float(p.get("return_score", 0.0))
        risk = _safe_float(p.get("risk_score", 0.0))
        decorated.append((dm, ret, risk, idx, p))

    # Sort descending by dm, ret, risk, then ascending by original idx
    decorated.sort(key=lambda x: (-x[0], -x[1], -x[2], x[3]))

    ranked: List[Dict[str, Any]] = []
    for rank_idx, (_, _, _, _, p) in enumerate(decorated, start=1):
        item = dict(p)
        item["rank"] = rank_idx
        # Hard guarantee: 'portfolio' key is always list[str]
        port_val = item.get("portfolio", [])
        if isinstance(port_val, (tuple, set)):
            item["portfolio"] = list(port_val)
        elif not isinstance(port_val, list):
            item["portfolio"] = [str(port_val)]
        ranked.append(item)

    return ranked


def select_top_n_portfolios(ranked_portfolios: List[Dict[str, Any]], n: int = 3) -> List[Dict[str, Any]]:
    """
    Return the top n portfolios by DM Score.
    Fully automatic — zero hardcoded stock overrides or manual intervention.

    Args:
        ranked_portfolios: Ranked list of portfolio dictionaries.
        n: Number of top portfolios to select (default 3).

    Returns:
        List of top n portfolio dictionaries.
    """
    if not ranked_portfolios or n <= 0:
        return []
    return ranked_portfolios[:n]


# ---------------------------------------------------------------------------
# Full Automatic Pipeline Evaluator
# ---------------------------------------------------------------------------

def evaluate_and_score_all_candidates(
    candidate_tuples: List[Tuple[str, ...]],
    stock_metrics: pd.DataFrame,
    correlation_matrix: pd.DataFrame,
    color_groups: Union[Dict[int, List[str]], Dict[str, int]],
    dominance_relation: List[Tuple[str, str]],
    pool_size: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Full end-to-end Automatic DM Portfolio Scoring pipeline.
    Combines outputs from portfolio evaluation, correlation matrix, graph coloring,
    and dominance relations into a single deterministic ranking.

    Returns:
        {
            "top_portfolios": list of Top 3 portfolios,
            "all_evaluated": list of all scored and ranked portfolios,
            "total_evaluated": int count,
            "weights": validated weights dictionary,
            "disclaimer": standard language disclaimer string,
        }
    """
    if not candidate_tuples:
        return {
            "top_portfolios": [],
            "all_evaluated": [],
            "total_evaluated": 0,
            "weights": weights or DEFAULT_DM_WEIGHTS,
            "disclaimer": DISCLAIMER,
        }

    if pool_size is None or pool_size <= 0:
        unique_stocks = {s for t in candidate_tuples for s in t}
        pool_size = max(len(unique_stocks), 1)

    # 1. Base evaluation for each candidate
    base_evaluated: List[Dict[str, Any]] = []
    for t in candidate_tuples:
        ev = evaluate_portfolio(t, stock_metrics, correlation_matrix)
        ev["portfolio"] = list(t)
        base_evaluated.append(ev)

    # 2. Return & Risk normalization across candidate set
    scored_return = calculate_return_score(base_evaluated)
    scored_risk = calculate_risk_score(scored_return)

    # 3. Individual component scores & final DM Score
    k = len(candidate_tuples[0]) if candidate_tuples else 3
    final_scored: List[Dict[str, Any]] = []

    for p in scored_risk:
        t = tuple(p["portfolio"])
        div_score = calculate_diversification_score(t, correlation_matrix)
        group_score = calculate_group_diversity_score(t, color_groups, k=k)
        dom_score = calculate_dominance_score(t, dominance_relation, pool_size=pool_size)

        p["diversification_score"] = div_score
        p["group_score"] = group_score
        p["group_diversity_score"] = group_score
        p["dominance_score"] = dom_score

        dm = calculate_dm_score(p, weights)
        p["dm_score"] = dm
        final_scored.append(p)

    # 4. Rank and select Top 3
    ranked = rank_portfolios(final_scored)
    top_3 = select_top_n_portfolios(ranked, n=3)

    return {
        "top_portfolios": top_3,
        "all_evaluated": ranked,
        "total_evaluated": len(ranked),
        "weights": weights or DEFAULT_DM_WEIGHTS,
        "disclaimer": DISCLAIMER,
    }
