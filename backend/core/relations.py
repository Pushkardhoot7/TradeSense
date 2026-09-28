"""
relations.py - Dominance Relation Mathematics (from scratch)
============================================================
Implements Pareto-style dominance relation analysis over stock metrics.
No NetworkX is used anywhere in this module.

A stock A dominates stock B when:
    Return(A) >= Return(B)  AND  Risk(A) <= Risk(B)

This constitutes a preorder (reflexive + transitive). When tied stocks
(mutual dominance) are treated as equivalence classes the quotient becomes
a partial order (POSET).
"""

from __future__ import annotations

from collections import defaultdict


# ---------------------------------------------------------------------------
# Primitive Dominance Check
# ---------------------------------------------------------------------------

def dominates(a: dict, b: dict) -> bool:
    """Return True if stock a Pareto-dominates stock b.

    Parameters
    ----------
    a, b:
        Metric dicts that must contain keys 'return_pct' and 'risk_pct'.

    Returns
    -------
    bool
        True when Return(a) >= Return(b) AND Risk(a) <= Risk(b).
    """
    return a["return_pct"] >= b["return_pct"] and a["risk_pct"] <= b["risk_pct"]


# ---------------------------------------------------------------------------
# Build Dominance Relation
# ---------------------------------------------------------------------------

def build_dominance_relation(metrics: list[dict]) -> list[tuple[str, str]]:
    """Build the full dominance relation over a universe of stocks.

    Parameters
    ----------
    metrics:
        List of metric dicts, each containing at minimum:
        'symbol', 'return_pct', 'risk_pct'.

    Returns
    -------
    list[tuple[str, str]]
        Ordered pairs (A, B) where A dominates B and A != B.
    """
    relation: list[tuple[str, str]] = []

    for i, stock_a in enumerate(metrics):
        for j, stock_b in enumerate(metrics):
            if i == j:
                continue
            if stock_a["symbol"] == stock_b["symbol"]:
                continue
            if dominates(stock_a, stock_b):
                relation.append((stock_a["symbol"], stock_b["symbol"]))

    return relation


# ---------------------------------------------------------------------------
# Relation Matrix
# ---------------------------------------------------------------------------

def build_relation_matrix(
    tickers: list[str],
    relation_pairs: list[tuple[str, str]],
) -> list[list[int]]:
    """Build a binary relation matrix M where M[i][j] = 1 iff (tickers[i], tickers[j]) in relation.

    Rows represent dominators, columns represent dominated stocks.

    Parameters
    ----------
    tickers:
        Ordered list of stock symbols (defines row/column indices).
    relation_pairs:
        Output of :func:`build_dominance_relation`.

    Returns
    -------
    list[list[int]]
        n x n matrix of 0s and 1s.
    """
    n = len(tickers)
    idx: dict[str, int] = {t: i for i, t in enumerate(tickers)}

    matrix: list[list[int]] = [[0] * n for _ in range(n)]

    for a, b in relation_pairs:
        if a in idx and b in idx:
            matrix[idx[a]][idx[b]] = 1

    return matrix


# ---------------------------------------------------------------------------
# Reflexivity, Antisymmetry, Transitivity
# ---------------------------------------------------------------------------

def check_reflexive(tickers: list[str]) -> tuple[bool, str]:
    """Check reflexivity of the dominance relation.

    Dominance is always reflexive by definition:
    Return(A) >= Return(A) and Risk(A) <= Risk(A).

    Parameters
    ----------
    tickers:
        List of stock symbols (used only to confirm the universe is non-empty).

    Returns
    -------
    tuple[bool, str]
        Always (True, 'Reflexive: A >= A holds for all stocks by definition').
    """
    return (True, "Reflexive: A >= A holds for all stocks by definition")


def check_antisymmetric(
    relation_pairs: list[tuple[str, str]],
) -> dict:
    """Detect symmetric pairs (tied stocks) in the relation.

    In a strict partial order, antisymmetry requires that
    (A,B) and (B,A) both in relation implies A == B.
    For stock dominance, tied stocks violate this, making the
    relation a preorder rather than a strict partial order.

    Parameters
    ----------
    relation_pairs:
        List of (A, B) dominance pairs.

    Returns
    -------
    dict
        {is_antisymmetric: bool, tied_pairs: list, message: str}
    """
    pair_set: set[tuple[str, str]] = set(relation_pairs)
    tied_pairs: list[tuple[str, str]] = []

    seen: set[frozenset] = set()

    for a, b in relation_pairs:
        if a == b:
            continue
        edge = frozenset({a, b})
        if edge in seen:
            continue
        seen.add(edge)
        if (b, a) in pair_set:
            tied_pairs.append((a, b))

    is_antisymmetric = len(tied_pairs) == 0

    if is_antisymmetric:
        message = (
            "Antisymmetric: no two distinct stocks mutually dominate each other. "
            "The relation forms a strict partial order."
        )
    else:
        message = (
            f"Not strictly antisymmetric: {len(tied_pairs)} tied pair(s) found. "
            "These stocks have identical Return and Risk profiles (or are equivalent). "
            "The relation is a preorder; quotient by tied equivalence classes gives a POSET."
        )

    return {
        "is_antisymmetric": is_antisymmetric,
        "tied_pairs": [list(p) for p in tied_pairs],
        "message": message,
    }


def check_transitive(
    relation_pairs: list[tuple[str, str]],
) -> tuple[bool, str]:
    """Verify transitivity of the dominance relation.

    For all (A,B) and (B,C) in the relation, (A,C) must also be present.

    Parameters
    ----------
    relation_pairs:
        List of (A, B) dominance pairs.

    Returns
    -------
    tuple[bool, str]
        (is_transitive, description)
    """
    pair_set: set[tuple[str, str]] = set(relation_pairs)

    # Build adjacency for iteration
    successors: dict[str, list[str]] = defaultdict(list)
    for a, b in relation_pairs:
        successors[a].append(b)

    violations: list[str] = []

    for a, b in relation_pairs:
        for c in successors[b]:
            if (a, c) not in pair_set:
                violations.append(f"({a}>={b}) and ({b}>={c}) but NOT ({a}>={c})")
                if len(violations) >= 5:
                    break
        if len(violations) >= 5:
            break

    is_transitive = len(violations) == 0

    if is_transitive:
        description = (
            "Transitive: for all A>=B and B>=C, A>=C holds. "
            "This follows from the monotonicity of >= and <= over real numbers."
        )
    else:
        description = (
            f"NOT transitive - {len(violations)} violation(s) found: "
            + "; ".join(violations)
        )

    return (is_transitive, description)


# ---------------------------------------------------------------------------
# POSET Analysis
# ---------------------------------------------------------------------------

def analyze_poset(
    relation_pairs: list[tuple[str, str]],
    tickers: list[str],
) -> dict:
    """Comprehensive POSET analysis of the dominance relation.

    Parameters
    ----------
    relation_pairs:
        Output of :func:`build_dominance_relation`.
    tickers:
        Full list of stock symbols.

    Returns
    -------
    dict
        Keys: is_reflexive, is_antisymmetric, is_transitive,
        is_partial_order, antisymmetric_note, tied_groups, message.
    """
    is_reflexive, _ = check_reflexive(tickers)
    antisym_result = check_antisymmetric(relation_pairs)
    is_antisymmetric: bool = antisym_result["is_antisymmetric"]
    is_transitive, trans_desc = check_transitive(relation_pairs)

    is_partial_order = is_reflexive and is_antisymmetric and is_transitive

    # Build tied groups from tied_pairs via union-find
    tied_pairs: list[list[str]] = antisym_result["tied_pairs"]
    parent: dict[str, str] = {t: t for t in tickers}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: str, y: str) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[rx] = ry

    for pair in tied_pairs:
        if len(pair) == 2:
            _union(pair[0], pair[1])

    clusters: dict[str, list[str]] = defaultdict(list)
    for t in tickers:
        clusters[_find(t)].append(t)

    tied_groups = [sorted(v) for v in clusters.values() if len(v) > 1]

    if is_partial_order:
        message = (
            "The dominance relation is a Partial Order (POSET): "
            "reflexive, antisymmetric, and transitive."
        )
    else:
        parts = []
        if not is_antisymmetric:
            parts.append("antisymmetry fails (tied stocks exist -> preorder)")
        if not is_transitive:
            parts.append("transitivity fails")
        message = (
            "The dominance relation is a Preorder (not a strict POSET): "
            + "; ".join(parts)
            + ". Treat tied equivalence classes as single nodes for a POSET."
        )

    return {
        "is_reflexive": is_reflexive,
        "is_antisymmetric": is_antisymmetric,
        "is_transitive": is_transitive,
        "is_partial_order": is_partial_order,
        "antisymmetric_note": antisym_result["message"],
        "tied_groups": tied_groups,
        "message": message,
    }


# ---------------------------------------------------------------------------
# Frontier / Ranking Helpers
# ---------------------------------------------------------------------------

def get_non_dominated(
    relation_pairs: list[tuple[str, str]],
    tickers: list[str],
) -> list[str]:
    """Return stocks that are NOT dominated by any other stock.

    A stock is non-dominated if it has zero incoming dominance edges
    (no other stock dominates it).

    Parameters
    ----------
    relation_pairs:
        List of (dominator, dominated) pairs.
    tickers:
        Full universe of stock symbols.

    Returns
    -------
    list[str]
        Sorted list of Pareto-optimal stock symbols.
    """
    dominated: set[str] = {b for _, b in relation_pairs}
    non_dominated = [t for t in tickers if t not in dominated]
    return sorted(non_dominated)


def get_incomparable_pairs(
    relation_pairs: list[tuple[str, str]],
    tickers: list[str],
) -> list[tuple[str, str]]:
    """Find all pairs of stocks that are mutually incomparable.

    A pair (A, B) is incomparable when neither A>=B nor B>=A.

    Parameters
    ----------
    relation_pairs:
        List of (A, B) dominance pairs.
    tickers:
        Full universe of stock symbols.

    Returns
    -------
    list[tuple[str, str]]
        List of incomparable pairs (A, B) with A < B lexicographically.
    """
    pair_set: set[tuple[str, str]] = set(relation_pairs)
    incomparable: list[tuple[str, str]] = []

    n = len(tickers)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = tickers[i], tickers[j]
            if (a, b) not in pair_set and (b, a) not in pair_set:
                incomparable.append((a, b))

    return incomparable


def get_most_dominant(
    relation_pairs: list[tuple[str, str]],
    tickers: list[str],
) -> list[dict]:
    """Rank stocks by how many others they dominate (out-degree).

    Parameters
    ----------
    relation_pairs:
        List of (A, B) dominance pairs.
    tickers:
        Full universe of stock symbols.

    Returns
    -------
    list[dict]
        Dicts with keys 'symbol' and 'dominated_count', sorted
        descending by dominated_count.
    """
    out_degree: dict[str, int] = {t: 0 for t in tickers}

    for a, _ in relation_pairs:
        if a in out_degree:
            out_degree[a] += 1

    ranked = sorted(
        [{"symbol": t, "dominated_count": out_degree[t]} for t in tickers],
        key=lambda x: (-x["dominated_count"], x["symbol"]),
    )

    return ranked
