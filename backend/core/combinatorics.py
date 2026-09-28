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
    pool_size: int = 15,
) -> list[str]:
    """
    Reduce the eligible stock universe to at most `pool_size` candidates,
    ranked by Sharpe ratio descending (Return% as fallback).
    """
    if not metrics:
        return []

    valid = [m for m in metrics if m.get("return_pct") is not None]
    if not valid:
        valid = metrics

    def sort_key(m: dict) -> float:
        return float(m.get("sharpe", m.get("return_pct", 0.0)) or 0.0)

    ranked = sorted(valid, key=sort_key, reverse=True)
    return [m["symbol"] for m in ranked[:pool_size]]


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
