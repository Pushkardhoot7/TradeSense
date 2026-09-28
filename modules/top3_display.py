"""
Top 3 DM Score Display & Ranking Layer for TradeSense

Consumes outputs from modules.dm_scoring engine and formats both printable/string views and structured data contract objects.

No-Hardcoding Guarantee:
  Do not hardcode portfolio names or scores here — all values must come from the scoring engine output.

Medal Emojis Order:
  Rank 1: 🥇
  Rank 2: 🥈
  Rank 3: 🥉
"""

from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from modules.dm_scoring import (
    run_dm_portfolio_scoring,
    DM_SCORE_WEIGHTS,
    DISCLAIMER,
)
from modules.relations import build_dominance_relation, get_non_dominated_stocks

MEDAL_EMOJIS = ["🥇", "🥈", "🥉"]


def build_top3_structured_data(
    dm_scoring_result: Dict[str, Any],
    stock_metrics: pd.DataFrame,
    coloring_dict: Dict[str, int],
) -> List[Dict[str, Any]]:
    """
    Construct the required structured data contract array for UI / API consumption:
    [
      {
        "rank": int,
        "stocks": list[str],
        "dm_score": float,
        "return_raw": float,
        "return_score": float,
        "risk_raw": float,
        "risk_score": float,
        "avg_correlation": float,
        "diversification_score": float,
        "groups_represented": list[int],
        "group_diversity_score": float,
        "dominance_detail": {"non_dominated": list[str], "dominated": list[str]},
        "dominance_score": float,
        "weights_used": dict
      },
      ...
    ]
    """
    top_portfolios = dm_scoring_result.get("top_portfolios", [])
    weights_used = dm_scoring_result.get("weights", DM_SCORE_WEIGHTS)

    # Compute Pareto non-dominated universe across full stock metrics
    if stock_metrics is not None and not stock_metrics.empty:
        dominance_pairs = build_dominance_relation(stock_metrics)
        all_tickers = stock_metrics["Stock"].tolist() if "Stock" in stock_metrics.columns else list(stock_metrics.index)
        non_dominated_universe = set(get_non_dominated_stocks(dominance_pairs, all_tickers))
    else:
        non_dominated_universe = set()

    structured_list = []
    for item in top_portfolios:
        p_stocks = list(item["portfolio"])
        non_dom_list = [s for s in p_stocks if s in non_dominated_universe]
        dom_list = [s for s in p_stocks if s not in non_dominated_universe]

        groups_rep = sorted(list(set(coloring_dict.get(s, -1) for s in p_stocks)))

        struct_item = {
            "rank": item["rank"],
            "stocks": p_stocks,
            "dm_score": item["dm_score"],
            "return_raw": item["raw_metrics"]["raw_return"],
            "return_score": item["component_scores"]["return_score"],
            "risk_raw": item["raw_metrics"]["raw_risk"],
            "risk_score": item["component_scores"]["risk_score"],
            "avg_correlation": item["raw_metrics"]["raw_avg_corr"],
            "diversification_score": item["component_scores"]["diversification_score"],
            "groups_represented": groups_rep,
            "group_diversity_score": item["component_scores"]["group_diversity_score"],
            "dominance_detail": {
                "non_dominated": non_dom_list,
                "dominated": dom_list,
            },
            "dominance_score": item["component_scores"]["dominance_score"],
            "weights_used": weights_used,
        }
        structured_list.append(struct_item)

    return structured_list


# Do not hardcode portfolio names or scores here — all values must come from the scoring engine output.
def render_top3_portfolios(
    candidate_portfolios: List[Tuple[str, ...]],
    stock_metrics: pd.DataFrame,
    returns_matrix: pd.DataFrame,
    corr_df: pd.DataFrame,
    coloring_dict: Dict[str, int],
    top_n: int = 3,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Render Top 3 DM Score formatted text view and return structured data contract.

    No-Hardcoding Guarantee:
      Do not hardcode portfolio names or scores here — all values must come from the scoring engine output.
    """
    dm_res = run_dm_portfolio_scoring(
        candidate_portfolios=candidate_portfolios,
        stock_metrics=stock_metrics,
        returns_matrix=returns_matrix,
        corr_df=corr_df,
        coloring_dict=coloring_dict,
        top_n=top_n,
    )

    structured_data = build_top3_structured_data(dm_res, stock_metrics, coloring_dict)

    # Format text output
    lines = [
        "========================================",
        "TOP 3 PORTFOLIOS",
        "BY DM SCORE",
        "========================================",
        "",
    ]

    if not structured_data:
        lines.append("No eligible portfolios available to evaluate.")
        lines.append("")
        lines.append(f"Disclaimer: {DISCLAIMER}")
        return "\n".join(lines), []

    for idx, p in enumerate(structured_data):
        rank_pos = p["rank"]
        medal = MEDAL_EMOJIS[idx] if idx < len(MEDAL_EMOJIS) else f"#{rank_pos}"
        stocks_str = " + ".join(p["stocks"])

        non_dom_str = ", ".join(p["dominance_detail"]["non_dominated"]) if p["dominance_detail"]["non_dominated"] else "None"
        dom_str = ", ".join(p["dominance_detail"]["dominated"]) if p["dominance_detail"]["dominated"] else "None"
        groups_str = ", ".join(f"Group #{g}" for g in p["groups_represented"]) if p["groups_represented"] else "None"

        lines.append(f"Rank {rank_pos}: {medal} {stocks_str}")
        lines.append(f"  • DM Score: {p['dm_score']:.2f}")
        lines.append(f"  • Return: {p['return_raw']:+.2f}%")
        lines.append(f"  • Risk (Volatility): {p['risk_raw']:.2f}% (Annualized Covariance Volatility)")
        lines.append(f"  • Average Correlation: {p['avg_correlation'] * 100:.2f}%")
        lines.append(f"  • Groups Represented ({len(p['groups_represented'])}): {groups_str}")
        lines.append(f"  • Dominance Info: Non-Dominated = [{non_dom_str}] | Dominated = [{dom_str}]")
        lines.append("")

    lines.append("----------------------------------------")
    lines.append(f"Disclaimer: {DISCLAIMER}")
    lines.append("========================================")

    return "\n".join(lines), structured_data
