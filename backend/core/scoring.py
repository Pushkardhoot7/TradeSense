"""
scoring.py - DM Portfolio Score Engine for TradeSense V2.1
===========================================================
Implements the 5-component DM Score formula deterministically.

Formula (Updated V2.1 Weights)
-------------------------------
DM Score = 0.30×Return + 0.25×Risk + 0.20×Diversification
         + 0.15×Stability + 0.10×Network_Independence

All components normalised to [0, 100] via min-max over the candidate pool.
"""

from __future__ import annotations
import math
from typing import Any

import numpy as np
import pandas as pd

from backend.core.portfolio import evaluate_portfolio


DM_WEIGHTS: dict[str, float] = {
    "return":              0.30,
    "risk":                0.25,
    "diversification":     0.20,
    "stability":           0.15,
    "network_independence": 0.10,
}

DISCLAIMER = (
    "DM Score is a project-defined mathematical analytical score. "
    "It is NOT a guaranteed best portfolio — it reflects a specific, "
    "documented weighting of return, risk, diversification, stability, "
    "and network independence characteristics and should be interpreted accordingly."
)


def _minmax(value: float, lo: float, hi: float, invert: bool = False) -> float:
    if math.isclose(hi, lo):
        return 100.0
    norm = 100.0 * (value - lo) / (hi - lo)
    return round(max(0.0, min(100.0, 100.0 - norm if invert else norm)), 4)


def _compute_stability(symbols: tuple[str, ...], returns_df: pd.DataFrame) -> float:
    """
    Compute return stability for a portfolio.
    Higher stability = more consistent returns (lower coefficient of variation).
    Returns a raw value (lower = more stable, will be inverted during scoring).
    """
    if returns_df is None or returns_df.empty:
        return 50.0  # neutral default
    valid_cols = [s for s in symbols if s in returns_df.columns]
    if not valid_cols:
        return 50.0
    # Equal-weight portfolio daily returns
    port_returns = returns_df[valid_cols].mean(axis=1).dropna()
    if len(port_returns) < 10:
        return 50.0
    # Coefficient of variation of rolling 20-day returns (lower = more stable)
    rolling_mean = port_returns.rolling(20).mean().dropna()
    rolling_std = port_returns.rolling(20).std().dropna()
    if len(rolling_mean) == 0 or rolling_mean.abs().mean() < 1e-10:
        return 50.0
    cv = float(rolling_std.mean() / max(rolling_mean.abs().mean(), 1e-10))
    return cv


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
    Deterministic DM Portfolio Scoring pipeline.

    Parameters
    ----------
    candidates    : list of (symbol, ...) tuples from combinatorics stage
    metrics       : list of per-stock metric dicts
    returns_df    : aligned daily-returns DataFrame
    corr_df       : Pearson correlation DataFrame
    coloring      : Welsh-Powell coloring dict {symbol: color_id}
    non_dominated : set of non-dominated stock symbols
    top_n         : number of top portfolios to return (default 3)

    Returns
    -------
    {
      top_portfolios:  list[PortfolioSchema-compatible dict],
      all_evaluated:   list[dict],
      total_evaluated: int,
      weights:         dict,
      disclaimer:      str,
    }

    HARD GUARANTEE: every dict in top_portfolios has 'portfolio': list[str]
    """
    if not candidates:
        return {"top_portfolios": [], "all_evaluated": [], "total_evaluated": 0,
                "weights": DM_WEIGHTS, "disclaimer": DISCLAIMER}

    total_color_groups = max(len(set(coloring.values())), 1) if coloring else 1

    # ------------------------------------------------------------------
    # Step 1 — Evaluate raw metrics for every candidate
    # ------------------------------------------------------------------
    raw: list[dict] = []
    for idx, p in enumerate(candidates):
        ev = evaluate_portfolio(p, metrics, returns_df, corr_df)
        if not ev["valid"]:
            continue

        colors_in_portfolio = {coloring.get(s, -1) for s in p}
        dom_count = sum(1 for s in p if s in non_dominated)
        stability_cv = _compute_stability(p, returns_df)

        raw.append({
            "insertion_idx":    idx,
            "portfolio":        ev["portfolio"],          # list[str] — always present
            "k":                ev["k"],
            "raw_return":       ev["return_pct"],
            "raw_risk":         ev["risk_pct"],
            "avg_correlation":  ev["avg_correlation"],
            "distinct_groups":  len(colors_in_portfolio),
            "dom_count":        dom_count,
            "stability_cv":     stability_cv,
        })

    if not raw:
        return {"top_portfolios": [], "all_evaluated": [], "total_evaluated": 0,
                "weights": DM_WEIGHTS, "disclaimer": DISCLAIMER}

    # ------------------------------------------------------------------
    # Step 2 — Pool-level min/max for normalisation
    # ------------------------------------------------------------------
    all_ret  = [r["raw_return"] for r in raw]
    all_risk = [r["raw_risk"]   for r in raw]
    all_stab = [r["stability_cv"] for r in raw]
    min_ret, max_ret   = min(all_ret),  max(all_ret)
    min_risk, max_risk = min(all_risk), max(all_risk)
    min_stab, max_stab = min(all_stab), max(all_stab)

    # ------------------------------------------------------------------
    # Step 3 — Compute component scores & DM Score
    # ------------------------------------------------------------------
    scored: list[dict] = []
    for r in raw:
        k = r["k"]

        ret_score  = _minmax(r["raw_return"], min_ret,  max_ret,  invert=False)
        risk_score = _minmax(r["raw_risk"],   min_risk, max_risk, invert=True)   # lower risk → higher score

        # Diversification: 100 * (1 - avg_corr) / 2 maps [-1,+1] → [0,100]
        corr_clamped = max(-1.0, min(1.0, r["avg_correlation"]))
        div_score = max(0.0, min(100.0, 100.0 * (1.0 - corr_clamped) / 2.0))

        # Stability: lower CV → higher stability score (inverted)
        stab_score = _minmax(r["stability_cv"], min_stab, max_stab, invert=True)

        # Network Independence: how many distinct color groups the portfolio spans
        max_possible_groups = min(k, total_color_groups)
        net_score = 100.0 * r["distinct_groups"] / max_possible_groups if max_possible_groups > 0 else 100.0
        net_score = max(0.0, min(100.0, net_score))

        dm = (
            DM_WEIGHTS["return"]               * ret_score
            + DM_WEIGHTS["risk"]               * risk_score
            + DM_WEIGHTS["diversification"]    * div_score
            + DM_WEIGHTS["stability"]          * stab_score
            + DM_WEIGHTS["network_independence"] * net_score
        )

        scored.append({
            # KEY FIX: portfolio is ALWAYS list[str]
            "portfolio":             r["portfolio"],
            "dm_score":              round(dm, 2),
            "return_score":          round(ret_score, 2),
            "risk_score":            round(risk_score, 2),
            "diversification_score": round(div_score, 2),
            "stability_score":       round(stab_score, 2),
            "network_score":         round(net_score, 2),
            # Keep legacy keys for backward compatibility
            "group_score":           round(net_score, 2),
            "dominance_score":       round(stab_score, 2),
            "raw_return":            round(r["raw_return"], 4),
            "raw_risk":              round(r["raw_risk"], 4),
            "avg_correlation":       round(r["avg_correlation"], 4),
            "_insertion_idx":        r["insertion_idx"],
        })

    # ------------------------------------------------------------------
    # Step 4 — Deterministic sort: (-dm, -ret, -risk, insertion_idx)
    # ------------------------------------------------------------------
    scored.sort(key=lambda x: (
        -x["dm_score"],
        -x["return_score"],
        -x["risk_score"],
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
