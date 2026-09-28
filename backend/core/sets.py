"""
sets.py - Set Theory Operations for TradeSense V2
===================================================
Defines and computes mathematical sets A (High Return), B (High Volume),
C (Low Risk) from actual stock metric data, then computes all standard
set operations using Python's built-in frozenset.

Every result is a real computation — no decorative text.
"""

from __future__ import annotations
from typing import Any


def define_high_return_set(metrics: list[dict], threshold: float = 5.0) -> frozenset[str]:
    """Return frozenset of stocks where return_pct > threshold."""
    return frozenset(
        m["symbol"] for m in metrics
        if float(m.get("return_pct", 0.0)) > threshold
    )


def define_low_risk_set(metrics: list[dict], threshold: float = 20.0) -> frozenset[str]:
    """Return frozenset of stocks where risk_pct < threshold."""
    return frozenset(
        m["symbol"] for m in metrics
        if float(m.get("risk_pct", 999.0)) < threshold
    )


def define_high_volume_set(metrics: list[dict]) -> frozenset[str]:
    """Return frozenset of stocks where avg_volume > mean(avg_volume across all stocks)."""
    volumes = [float(m.get("avg_volume", 0.0)) for m in metrics]
    if not volumes:
        return frozenset()
    mean_vol = sum(volumes) / len(volumes)
    return frozenset(
        m["symbol"] for m in metrics
        if float(m.get("avg_volume", 0.0)) > mean_vol
    )


def _fs_to_sorted_list(s: frozenset[str]) -> list[str]:
    return sorted(s)


def compute_all_set_operations(
    A: frozenset[str],
    B: frozenset[str],
    C: frozenset[str],
    labels: tuple[str, str, str] = ("High Return", "High Volume", "Low Risk"),
) -> dict[str, Any]:
    """
    Compute all standard set operations on A, B, C using Python frozensets.

    Returns a dict with all results as sorted lists plus counts.
    """
    A_union_B            = A | B
    A_inter_B            = A & B
    A_inter_C            = A & C
    B_inter_C            = B & C
    A_inter_B_inter_C    = A & B & C
    A_minus_B            = A - B
    B_minus_A            = B - A
    A_union_B_inter_C    = (A | B) & C
    A_union_B_union_C    = A | B | C
    complement_A_in_ABC  = A_union_B_union_C - A

    return {
        "label_A": labels[0],
        "label_B": labels[1],
        "label_C": labels[2],
        "A": {"members": _fs_to_sorted_list(A), "size": len(A)},
        "B": {"members": _fs_to_sorted_list(B), "size": len(B)},
        "C": {"members": _fs_to_sorted_list(C), "size": len(C)},
        "A_union_B":         {"members": _fs_to_sorted_list(A_union_B),         "size": len(A_union_B),         "formula": "A ∪ B"},
        "A_inter_B":         {"members": _fs_to_sorted_list(A_inter_B),         "size": len(A_inter_B),         "formula": "A ∩ B"},
        "A_inter_C":         {"members": _fs_to_sorted_list(A_inter_C),         "size": len(A_inter_C),         "formula": "A ∩ C"},
        "B_inter_C":         {"members": _fs_to_sorted_list(B_inter_C),         "size": len(B_inter_C),         "formula": "B ∩ C"},
        "A_inter_B_inter_C": {"members": _fs_to_sorted_list(A_inter_B_inter_C), "size": len(A_inter_B_inter_C),"formula": "A ∩ B ∩ C"},
        "A_minus_B":         {"members": _fs_to_sorted_list(A_minus_B),         "size": len(A_minus_B),         "formula": "A − B"},
        "B_minus_A":         {"members": _fs_to_sorted_list(B_minus_A),         "size": len(B_minus_A),         "formula": "B − A"},
        "A_union_B_inter_C": {"members": _fs_to_sorted_list(A_union_B_inter_C), "size": len(A_union_B_inter_C),"formula": "(A ∪ B) ∩ C"},
        "A_union_B_union_C": {"members": _fs_to_sorted_list(A_union_B_union_C), "size": len(A_union_B_union_C),"formula": "A ∪ B ∪ C"},
    }
