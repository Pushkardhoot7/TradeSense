"""
Automatic DM Portfolio Score System for TradeSense

Implements a deterministic, non-manual pipeline that evaluates candidate portfolios across five weighted metrics, ranks them, and selects the Top 3 — with zero human/manual override.

Formula & Weights:
  DM Score = 0.30 * Return Score
           + 0.25 * Risk Score
           + 0.25 * Diversification Score
           + 0.10 * Group Diversity Score
           + 0.10 * Dominance Score

Component Calculations & Normalization:
1. Return Score (0.30):
   - Historical equal-weighted portfolio return over lookback window (default 252 trading days).
   - Min-max normalized across candidate pool: 100 * (R_i - min(R)) / (max(R) - min(R)).
2. Risk Score (0.25):
   - Portfolio volatility via covariance matrix sigma_p = sqrt(w^T Sigma w) over lookback window.
   - Covariance captures diversification benefits (unlike simple averaging).
   - Min-max normalized across candidate pool and inverted: 100 * (max(sigma) - sigma_i) / (max(sigma) - min(sigma)).
   - Degenerate case (max(sigma) == min(sigma)) assigns 100 to all candidates.
3. Diversification Score (0.25):
   - Mean pairwise Pearson correlation AvgCorr over constituent stock returns.
   - Absolute mapping over [-1, +1] range: 100 * (1 - AvgCorr) / 2. Single-stock portfolio gets AvgCorr = 1.0.
4. Graph Group Diversity Score (0.10):
   - Distinct graph color groups (Welsh-Powell independent sets) represented in portfolio: distinct_groups.
   - Normalized by max achievable: 100 * distinct_groups / min(portfolio_size, total_color_groups).
5. Dominance Score (0.10):
   - Count of Pareto non-dominated stocks in portfolio over full stock universe (higher return AND lower/equal risk).
   - Normalized by portfolio size: 100 * count_non_dominated / portfolio_size.

Deterministic Tiebreaking Rule:
  Portfolios are sorted by DM Score descending. Ties are broken deterministically by:
    1. Higher Return Score
    2. Higher Risk Score (lower volatility)
    3. Original candidate list insertion order

Disclaimer String:
  "DM Score is a project-defined mathematical analytical score. It is NOT a guaranteed best portfolio — it reflects a specific, documented weighting of return, risk, diversification, and dominance characteristics and should be interpreted accordingly."
"""

import math
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import pandas as pd
from modules.relations import build_dominance_relation, get_non_dominated_stocks

# Named Constants & Configuration
DM_SCORE_WEIGHTS = {
    "return": 0.30,
    "risk": 0.25,
    "diversification": 0.25,
    "group_diversity": 0.10,
    "dominance": 0.10,
}

DEFAULT_LOOKBACK_DAYS = 252  # 252 trading days per year on NSE
DEFAULT_CORRELATION_THRESHOLD = 0.70  # High-correlation graph edge threshold

DISCLAIMER = (
    "DM Score is a project-defined mathematical analytical score. "
    "It is NOT a guaranteed best portfolio — it reflects a specific, documented weighting "
    "of return, risk, diversification, and dominance characteristics and should be interpreted accordingly."
)


def validate_weights_config(weights: Dict[str, float] = DM_SCORE_WEIGHTS) -> bool:
    """
    Assert that configured component weights sum to 1.0 (100%).
    """
    total_weight = sum(weights.values())
    if not math.isclose(total_weight, 1.0, rel_tol=1e-5):
        raise ValueError(f"Configured DM Score weights must sum to 1.0 (got {total_weight:.4f})")
    return True


def compute_covariance_portfolio_volatility(
    portfolio: Tuple[str, ...], returns_matrix: pd.DataFrame
) -> float:
    """
    Compute portfolio volatility using the covariance-matrix method:
      sigma_portfolio = sqrt( w^T Sigma w )
    where w is equal weights vector (1/k) and Sigma is covariance matrix over lookback window.

    Covariance-based risk is used because it captures the correlation/covariance diversification
    benefit between constituents, whereas naive averaging ignores co-movement and yields mathematically
    meaningless risk figures for multi-stock portfolios.
    """
    k = len(portfolio)
    if k == 0:
        return 0.0
    w = np.full(k, 1.0 / k)

    if returns_matrix is None or returns_matrix.empty:
        return 0.0

    valid_cols = [c for c in portfolio if c in returns_matrix.columns]
    if len(valid_cols) != k:
        # Fallback if any column missing: return std of available returns
        avail_cols = [c for c in portfolio if c in returns_matrix.columns]
        if not avail_cols:
            return 0.0
        sub_ret = returns_matrix[avail_cols]
        cov_m = sub_ret.cov().values * 252.0  # Annualized covariance
        w_sub = np.full(len(avail_cols), 1.0 / len(avail_cols))
        port_var = float(w_sub.T @ cov_m @ w_sub)
        return float(math.sqrt(max(port_var, 0.0)))

    sub_returns = returns_matrix[list(portfolio)]
    # Annualized covariance matrix (252 trading days)
    cov_matrix = sub_returns.cov().values * 252.0
    portfolio_var = float(w.T @ cov_matrix @ w)
    return float(math.sqrt(max(portfolio_var, 0.0)))


def run_dm_portfolio_scoring(
    candidate_portfolios: List[Tuple[str, ...]],
    stock_metrics: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    corr_df: pd.DataFrame,
    coloring_dict: Dict[str, int],
    weights: Dict[str, float] = DM_SCORE_WEIGHTS,
    top_n: int = 3,
    manual_rank_override: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Pure, deterministic DM Portfolio Scoring pipeline.

    Evaluates candidates across 5 weighted metrics, ranks by composite DM Score, and selects Top N.

    HARD CONSTRAINT:
      Selection is a pure function of inputs. If manual_rank_override is passed, this function raises
      a ValueError to prevent manual tampering.
    """
    # Guard assertion: No manual override allowed!
    if manual_rank_override is not None:
        raise ValueError("CRITICAL GUARD: Top 3 selection function cannot be called with a manual ranking override!")

    validate_weights_config(weights)

    if not candidate_portfolios:
        return {
            "top_portfolios": [],
            "all_evaluated": [],
            "weights": weights,
            "disclaimer": DISCLAIMER,
        }

    # Extract Pareto non-dominated stocks across full universe
    if stock_metrics is not None and not stock_metrics.empty:
        dominance_pairs = build_dominance_relation(stock_metrics)
        all_tickers = stock_metrics["Stock"].tolist() if "Stock" in stock_metrics.columns else list(stock_metrics.index)
        non_dominated_universe = set(get_non_dominated_stocks(dominance_pairs, all_tickers))
    else:
        non_dominated_universe = set()

    total_color_groups = len(set(coloring_dict.values())) if coloring_dict else 1
    if total_color_groups == 0:
        total_color_groups = 1

    # Step 1: Compute Raw Metrics for all candidates
    raw_evaluations = []
    for idx, p in enumerate(candidate_portfolios):
        k = len(p)
        if k == 0:
            continue

        # Metric 1: Return (equal-weighted average of constituent returns)
        if "Stock" in stock_metrics.columns:
            sub_m = stock_metrics[stock_metrics["Stock"].isin(p)]
            ret_col = "Return %" if "Return %" in sub_m.columns else "CAGR"
            raw_ret = float(sub_m[ret_col].mean()) if not sub_m.empty and ret_col in sub_m.columns else 0.0
        else:
            raw_ret = 0.0

        # Metric 2: Covariance-based Portfolio Volatility
        raw_risk = compute_covariance_portfolio_volatility(p, returns_matrix)

        # Metric 3: Pairwise Pearson Correlation AvgCorr
        if k > 1 and corr_df is not None and not corr_df.empty:
            pairwise_corrs = []
            for i_idx in range(k):
                for j_idx in range(i_idx + 1, k):
                    t1, t2 = p[i_idx], p[j_idx]
                    if t1 in corr_df.columns and t2 in corr_df.columns:
                        pairwise_corrs.append(float(corr_df.loc[t1, t2]))
            raw_avg_corr = float(np.mean(pairwise_corrs)) if pairwise_corrs else 1.0
        else:
            raw_avg_corr = 1.0  # Single stock portfolio has no pairwise diversification (AvgCorr = 1.0)

        # Metric 4: Distinct Graph Color Groups
        colors = set(coloring_dict.get(s, -1) for s in p)
        raw_distinct_groups = len(colors)

        # Metric 5: Count of Pareto Non-Dominated Stocks
        raw_non_dominated_count = sum(1 for s in p if s in non_dominated_universe)

        raw_evaluations.append({
            "candidate_index": idx,
            "portfolio": p,
            "k": k,
            "raw_return": raw_ret,
            "raw_risk": raw_risk,
            "raw_avg_corr": raw_avg_corr,
            "raw_distinct_groups": raw_distinct_groups,
            "raw_non_dominated_count": raw_non_dominated_count,
        })

    if not raw_evaluations:
        return {
            "top_portfolios": [],
            "all_evaluated": [],
            "weights": weights,
            "disclaimer": DISCLAIMER,
        }

    # Extract pool min/max for min-max normalizations
    all_raw_returns = [item["raw_return"] for item in raw_evaluations]
    all_raw_risks = [item["raw_risk"] for item in raw_evaluations]

    min_ret, max_ret = min(all_raw_returns), max(all_raw_returns)
    min_risk, max_risk = min(all_raw_risks), max(all_raw_risks)

    # Step 2: Compute Component Scores & Composite DM Score
    scored_candidates = []
    for item in raw_evaluations:
        k = item["k"]

        # Component 1: Return Score (Min-max normalized across candidate pool)
        if max_ret == min_ret:
            ret_score = 100.0
        else:
            ret_score = 100.0 * (item["raw_return"] - min_ret) / (max_ret - min_ret)

        # Component 2: Risk Score (Min-max normalized across pool & inverted)
        # Degenerate case (max_risk == min_risk) assigns RiskScore = 100.0 for all candidates
        if max_risk == min_risk:
            risk_score = 100.0
        else:
            risk_score = 100.0 * (max_risk - item["raw_risk"]) / (max_risk - min_risk)

        # Component 3: Diversification Score (Absolute mapping over [-1, +1] range)
        raw_corr_clamped = max(-1.0, min(1.0, item["raw_avg_corr"]))
        div_score = max(0.0, min(100.0, 100.0 * (1.0 - raw_corr_clamped) / 2.0))

        # Component 4: Graph Group Diversity Score
        max_possible_groups = min(k, total_color_groups)
        group_div_score = 100.0 * item["raw_distinct_groups"] / max_possible_groups if max_possible_groups > 0 else 100.0
        group_div_score = max(0.0, min(100.0, group_div_score))

        # Component 5: Dominance Score
        dom_score = 100.0 * item["raw_non_dominated_count"] / k if k > 0 else 0.0
        dom_score = max(0.0, min(100.0, dom_score))

        # Assert range 0 <= score <= 100 for all component scores
        assert 0.0 <= ret_score <= 100.0, f"Return score out of range: {ret_score}"
        assert 0.0 <= risk_score <= 100.0, f"Risk score out of range: {risk_score}"
        assert 0.0 <= div_score <= 100.0, f"Diversification score out of range: {div_score}"
        assert 0.0 <= group_div_score <= 100.0, f"Group diversity score out of range: {group_div_score}"
        assert 0.0 <= dom_score <= 100.0, f"Dominance score out of range: {dom_score}"

        # Weighted Composite DM Score Formula
        dm_score = (
            weights["return"] * ret_score
            + weights["risk"] * risk_score
            + weights["diversification"] * div_score
            + weights["group_diversity"] * group_div_score
            + weights["dominance"] * dom_score
        )

        item_scored = {
            "portfolio": item["portfolio"],
            "dm_score": round(float(dm_score), 2),
            "component_scores": {
                "return_score": round(float(ret_score), 2),
                "risk_score": round(float(risk_score), 2),
                "diversification_score": round(float(div_score), 2),
                "group_diversity_score": round(float(group_div_score), 2),
                "dominance_score": round(float(dom_score), 2),
            },
            "raw_metrics": {
                "raw_return": round(float(item["raw_return"]), 4),
                "raw_risk": round(float(item["raw_risk"]), 4),
                "raw_avg_corr": round(float(item["raw_avg_corr"]), 4),
                "raw_distinct_groups": item["raw_distinct_groups"],
                "raw_non_dominated_count": item["raw_non_dominated_count"],
            },
            "candidate_index": item["candidate_index"],
        }
        scored_candidates.append(item_scored)

    # Step 3: Deterministic Ranking & Tiebreaking
    # Sort key: (-dm_score, -return_score, -risk_score, candidate_index)
    sorted_candidates = sorted(
        scored_candidates,
        key=lambda x: (
            -x["dm_score"],
            -x["component_scores"]["return_score"],
            -x["component_scores"]["risk_score"],
            x["candidate_index"],
        ),
    )

    # Assign rank 1..N
    for rank_idx, cand in enumerate(sorted_candidates, start=1):
        cand["rank"] = rank_idx
        cand["rationale"] = (
            f"Rank #{rank_idx}: Composite DM Score of {cand['dm_score']:.1f}/100. "
            f"Component Scores: Return={cand['component_scores']['return_score']:.0f}, "
            f"Risk={cand['component_scores']['risk_score']:.0f}, "
            f"Diversification={cand['component_scores']['diversification_score']:.0f}, "
            f"Group Diversity={cand['component_scores']['group_diversity_score']:.0f}, "
            f"Dominance={cand['component_scores']['dominance_score']:.0f}."
        )

    top_portfolios = sorted_candidates[:top_n]

    return {
        "top_portfolios": top_portfolios,
        "all_evaluated": sorted_candidates,
        "weights": weights,
        "disclaimer": DISCLAIMER,
    }
