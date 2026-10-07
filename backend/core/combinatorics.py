"""
combinatorics.py - Portfolio Combination Generator for TradeSense V2
======================================================================
Implements C(n,k) = n! / (k!(n-k)!) and generates portfolio candidate
combinations, bounding the generated set to avoid combinatorial explosion.
"""

from __future__ import annotations
import math
import itertools
from typing import Any


def combination_count(n: int, k: int) -> int:
    """
    C(n, k) = n! / (k! * (n-k)!)
    Returns 0 for invalid inputs (n < 0, k < 0, k > n).
    """
    if n < 0 or k < 0 or k > n:
        return 0
    return math.comb(n, k)


def generate_combinations(
    pool: list[str],
    k: int,
    max_combinations: int = 2000,
) -> tuple[list[tuple[str, ...]], int]:
    """
    Generate all C(n,k) portfolio candidates from `pool`.

    If total_possible > max_combinations, only the first `max_combinations`
    are returned (deterministic — no random sampling).

    Returns
    -------
    (combinations_list, total_possible_count)
    """
    n = len(pool)
    if n < k or k <= 0:
        return [], 0

    total_possible = combination_count(n, k)
    combos: list[tuple[str, ...]] = []

    for combo in itertools.combinations(pool, k):
        combos.append(combo)
        if len(combos) >= max_combinations:
            break

    return combos, total_possible


def select_candidate_pool(
    metrics: list[dict],
    pool_size: int = 20,
) -> list[str]:
    """
    Select an expert-diversified candidate pool of up to `pool_size` stocks.
    Uses multi-sector round-robin selection: ranks stocks within each sector
    by Sharpe ratio (with Return% fallback) and picks top candidates across
    different sectors to prevent sector-concentration bias.
    """
    if not metrics:
        return []

    valid = [m for m in metrics if m.get("return_pct") is not None]
    if not valid:
        valid = metrics

    # Group by sector
    sector_groups: dict[str, list[dict]] = {}
    for m in valid:
        sec = m.get("sector", "Other") or "Other"
        sector_groups.setdefault(sec, []).append(m)

    # Sort stocks within each sector by Sharpe ratio descending
    def sort_key(m: dict) -> float:
        return float(m.get("sharpe", m.get("return_pct", 0.0)) or 0.0)

    for sec in sector_groups:
        sector_groups[sec].sort(key=sort_key, reverse=True)

    # Round-robin selection across sectors to guarantee multi-sector representation
    selected: list[str] = []
    round_idx = 0
    while len(selected) < pool_size:
        added_in_round = 0
        for sec, stocks in sector_groups.items():
            if round_idx < len(stocks):
                sym = stocks[round_idx]["symbol"]
                if sym not in selected:
                    selected.append(sym)
                    added_in_round += 1
                    if len(selected) >= pool_size:
                        break
        if added_in_round == 0:
            break
        round_idx += 1

    # If still below pool_size, fill with remaining highest Sharpe stocks
    if len(selected) < pool_size:
        all_ranked = sorted(valid, key=sort_key, reverse=True)
        for m in all_ranked:
            if m["symbol"] not in selected:
                selected.append(m["symbol"])
                if len(selected) >= pool_size:
                    break

    return selected[:pool_size]


def format_combinatorics_display(
    n: int,
    k: int,
    evaluated: int,
    total_possible: int,
) -> dict[str, Any]:
    """Build display-ready combinatorics summary dict."""
    return {
        "n":              n,
        "k":              k,
        "formula":        "C(n,k) = n! / (k!(n-k)!)",
        "formula_filled": f"C({n},{k}) = {n}! / ({k}!×{n-k}!)",
        "total_possible": total_possible,
        "evaluated":      evaluated,
        "candidates_display": f"C({n},{k}) = {total_possible:,}",
        "capped": evaluated < total_possible,
    }
