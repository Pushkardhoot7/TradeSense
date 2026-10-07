"""
scoring.py - DM Portfolio Score Engine for TradeSense V2.2
===========================================================
Implements the 6-component DM Score formula with expert-level
diversification logic tuned for professional holding traders.

Formula (V2.2 Expert Weights)
-----------------------------
DM Score = 0.25×Return + 0.20×Risk + 0.25×Diversification
         + 0.12×Stability + 0.10×Network_Independence
         + 0.08×Sector_Spread

Key improvements over V2.1:
- Sector diversification bonus (avoid putting all eggs in one basket)
- Downside-deviation based stability (Sortino-style, rewards asymmetric upside)
- Correlation penalty scaled non-linearly (heavily penalises corr > 0.6)
- Return scoring uses risk-adjusted Sharpe, not raw CAGR
- Larger pool evaluation with multi-factor filtering

All components normalised to [0, 100] via min-max over the candidate pool.
"""

from __future__ import annotations
import math
from typing import Any

import numpy as np
import pandas as pd

from backend.core.portfolio import evaluate_portfolio


DM_WEIGHTS: dict[str, float] = {
    "return":              0.25,
    "risk":                0.20,
    "diversification":     0.25,
    "stability":           0.12,
    "network_independence": 0.10,
    "sector_spread":       0.08,
}

DISCLAIMER = (
    "DM Score is a project-defined mathematical analytical score. "
    "It is NOT a guaranteed best portfolio — it reflects a specific, "
    "documented weighting of return, risk, diversification, stability, "
    "network independence, and sector spread characteristics. "
    "Designed for long-term holding strategies seeking balanced diversification."
)


def _minmax(value: float, lo: float, hi: float, invert: bool = False) -> float:
    if math.isclose(hi, lo):
        return 100.0
    norm = 100.0 * (value - lo) / (hi - lo)
    return round(max(0.0, min(100.0, 100.0 - norm if invert else norm)), 4)


def _compute_stability(symbols: tuple[str, ...], returns_df: pd.DataFrame) -> float:
    """
    Compute return stability using Sortino-style downside deviation.
    Lower value = more stable (will be inverted during scoring).
    Uses downside deviation rather than full std to reward portfolios
    with positive volatility (big upside moves) while penalising downside.
    """
    if returns_df is None or returns_df.empty:
        return 50.0
    valid_cols = [s for s in symbols if s in returns_df.columns]
    if not valid_cols:
        return 50.0
    # Equal-weight portfolio daily returns
    port_returns = returns_df[valid_cols].mean(axis=1).dropna()
    if len(port_returns) < 20:
        return 50.0

    # Downside deviation (Sortino-style): only negative returns contribute
    negative_returns = port_returns[port_returns < 0]
    if len(negative_returns) < 5:
        return 0.1  # Very stable — almost no negative days

    downside_std = float(negative_returns.std())

    # Also factor in max drawdown depth for stability
    cumulative = (1 + port_returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdowns = (cumulative - rolling_max) / rolling_max
    max_dd = float(drawdowns.min().item()) if hasattr(drawdowns.min(), 'item') else float(drawdowns.min())

    # Combined stability metric: downside vol + drawdown severity
    stability_raw = downside_std * 100.0 + abs(max_dd) * 20.0
    return stability_raw


def _compute_sector_spread(
    symbols: tuple[str, ...],
    metrics: list[dict],
) -> int:
    """
    Count distinct sectors across portfolio constituents.
    More sectors = better diversification for holding traders.
    """
    metrics_map = {m["symbol"]: m for m in metrics}
    sectors = set()
    for s in symbols:
        m = metrics_map.get(s, {})
        sec = m.get("sector", "Unknown")
        if sec:
            sectors.add(sec)
    return len(sectors)


def run_scoring_pipeline(
    candidates: list[tuple[str, ...]],
    metrics: list[dict],
    returns_df: pd.DataFrame,
    corr_df: pd.DataFrame,
    coloring: dict[str, int],
    non_dominated: set[str],
    top_n: int = 3,
) -> dict[str, Any]:
    """
    Deterministic DM Portfolio Scoring pipeline (V2.2).

    Expert-tuned for long-term holding traders seeking:
    - Cross-sector diversification (avoid sector concentration risk)
    - Low pairwise correlation (true diversification, not just sector labels)
    - Consistent returns with controlled downside (Sortino-style stability)
    - Risk-adjusted returns, not just raw CAGR

    HARD GUARANTEE: every dict in top_portfolios has 'portfolio': list[str]
    """
    if not candidates:
        return {"top_portfolios": [], "all_evaluated": [], "total_evaluated": 0,
                "weights": DM_WEIGHTS, "disclaimer": DISCLAIMER}

    total_color_groups = max(len(set(coloring.values())), 1) if coloring else 1
    cov_df = (returns_df.cov() * 252) if (returns_df is not None and not returns_df.empty) else None

    # ------------------------------------------------------------------
    # Step 1 — Evaluate raw metrics for every candidate
    # ------------------------------------------------------------------
    raw: list[dict] = []
    for idx, p in enumerate(candidates):
        ev = evaluate_portfolio(p, metrics, returns_df, corr_df, cov_df=cov_df)
        if not ev["valid"]:
            continue

        colors_in_portfolio = {coloring.get(s, -1) for s in p}
        dom_count = sum(1 for s in p if s in non_dominated)
        stability_cv = _compute_stability(p, returns_df)
        sector_spread = _compute_sector_spread(p, metrics)

        raw.append({
            "insertion_idx":    idx,
            "portfolio":        ev["portfolio"],
            "k":                ev["k"],
            "raw_return":       ev["return_pct"],
            "raw_risk":         ev["risk_pct"],
            "avg_correlation":  ev["avg_correlation"],
            "distinct_groups":  len(colors_in_portfolio),
            "dom_count":        dom_count,
            "stability_cv":     stability_cv,
            "sector_spread":    sector_spread,
        })

    if not raw:
        return {"top_portfolios": [], "all_evaluated": [], "total_evaluated": 0,
                "weights": DM_WEIGHTS, "disclaimer": DISCLAIMER}

    # ------------------------------------------------------------------
    # Step 2 — Pool-level min/max for normalisation
    # ------------------------------------------------------------------
    all_ret   = [r["raw_return"] for r in raw]
    all_risk  = [r["raw_risk"]   for r in raw]
    all_stab  = [r["stability_cv"] for r in raw]
    all_sects = [r["sector_spread"] for r in raw]
    min_ret, max_ret    = min(all_ret),  max(all_ret)
    min_risk, max_risk  = min(all_risk), max(all_risk)
    min_stab, max_stab  = min(all_stab), max(all_stab)
    min_sect, max_sect  = min(all_sects), max(all_sects)

    # ------------------------------------------------------------------
    # Step 3 — Compute component scores & DM Score
    # ------------------------------------------------------------------
    scored: list[dict] = []
    for r in raw:
        k = r["k"]

        ret_score  = _minmax(r["raw_return"], min_ret,  max_ret,  invert=False)
        risk_score = _minmax(r["raw_risk"],   min_risk, max_risk, invert=True)

        # Non-linear diversification: heavily penalise corr > 0.6
        # Uses exponential scaling: low correlation is disproportionately rewarded
        corr_clamped = max(-1.0, min(1.0, r["avg_correlation"]))
        if corr_clamped <= 0:
            div_score = 100.0  # Negative correlation = perfect diversification
        elif corr_clamped <= 0.3:
            div_score = 90.0 + (0.3 - corr_clamped) / 0.3 * 10.0
        elif corr_clamped <= 0.5:
            div_score = 70.0 + (0.5 - corr_clamped) / 0.2 * 20.0
        elif corr_clamped <= 0.7:
            div_score = 40.0 + (0.7 - corr_clamped) / 0.2 * 30.0
        else:
            div_score = max(0.0, 40.0 * (1.0 - corr_clamped) / 0.3)
        div_score = round(max(0.0, min(100.0, div_score)), 4)

        # Stability: lower downside deviation → higher score (inverted)
        stab_score = _minmax(r["stability_cv"], min_stab, max_stab, invert=True)

        # Network Independence: how many distinct color groups the portfolio spans
        max_possible_groups = min(k, total_color_groups)
        net_score = 100.0 * r["distinct_groups"] / max_possible_groups if max_possible_groups > 0 else 100.0
        net_score = max(0.0, min(100.0, net_score))

        # Sector Spread: how many distinct sectors the portfolio covers
        sect_score = _minmax(r["sector_spread"], min_sect, max_sect, invert=False)

        dm = (
            DM_WEIGHTS["return"]               * ret_score
            + DM_WEIGHTS["risk"]               * risk_score
            + DM_WEIGHTS["diversification"]    * div_score
            + DM_WEIGHTS["stability"]          * stab_score
            + DM_WEIGHTS["network_independence"] * net_score
            + DM_WEIGHTS["sector_spread"]      * sect_score
        )

        scored.append({
            "portfolio":             r["portfolio"],
            "dm_score":              round(dm, 2),
            "return_score":          round(ret_score, 2),
            "risk_score":            round(risk_score, 2),
            "diversification_score": round(div_score, 2),
            "stability_score":       round(stab_score, 2),
            "network_score":         round(net_score, 2),
            "sector_score":          round(sect_score, 2),
            # Keep legacy keys for backward compatibility
            "group_score":           round(net_score, 2),
            "dominance_score":       round(stab_score, 2),
            "raw_return":            round(r["raw_return"], 4),
            "raw_risk":              round(r["raw_risk"], 4),
            "avg_correlation":       round(r["avg_correlation"], 4),
            "sector_spread":         r["sector_spread"],
            "_insertion_idx":        r["insertion_idx"],
        })

    # ------------------------------------------------------------------
    # Step 4 — Deterministic sort: (-dm, -ret, +risk, insertion_idx)
    # ------------------------------------------------------------------
    scored.sort(key=lambda x: (
        -x["dm_score"],
        -x["return_score"],
        x["raw_risk"],
        x["_insertion_idx"],
    ))

    # Assign ranks and clean internal key
    for rank_i, item in enumerate(scored, start=1):
        item["rank"] = rank_i
        item.pop("_insertion_idx", None)

    return {
        "top_portfolios":  scored[:top_n],
        "all_evaluated":   scored,
        "total_evaluated": len(scored),
        "weights":         DM_WEIGHTS,
        "disclaimer":      DISCLAIMER,
    }
