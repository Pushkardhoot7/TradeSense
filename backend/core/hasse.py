"""
hasse.py - Hasse Diagram via Transitive Reduction (from scratch)
================================================================
Builds the Hasse diagram of a dominance POSET by computing the
transitive reduction of the dominance relation.

All algorithms (transitive reduction, union-find, topological sort,
level assignment) are implemented from scratch without NetworkX.
"""

from __future__ import annotations

from collections import defaultdict, deque


# ---------------------------------------------------------------------------
# Transitive Reduction (Cover Relation)
# ---------------------------------------------------------------------------

def build_cover_relation(
    relation_pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Compute the cover relation (transitive reduction) of the dominance DAG.

    An edge (A, B) is a cover edge (i.e., A covers B) when there is no
    intermediate stock C such that A dominates C AND C dominates B.
    Only non-reflexive pairs are considered.

    Parameters
    ----------
    relation_pairs:
        Full transitive dominance relation (A, B) pairs with A != B.

    Returns
    -------
    list[tuple[str, str]]
        Cover edges only - the minimal set that preserves reachability.
    """
    pair_set: set[tuple[str, str]] = set(relation_pairs)

    # Build successor lists for efficient lookup
    successors: dict[str, set[str]] = defaultdict(set)
    for a, b in relation_pairs:
        successors[a].add(b)

    cover_edges: list[tuple[str, str]] = []

    for a, b in relation_pairs:
        # Check whether any intermediate C exists: A->C and C->B
        has_intermediate = False
        for c in successors[a]:
            if c == b or c == a:
                continue
            if b in successors[c]:
                has_intermediate = True
                break

        if not has_intermediate:
            cover_edges.append((a, b))

    return cover_edges


# ---------------------------------------------------------------------------
# Equivalence Classes (Union-Find from scratch)
# ---------------------------------------------------------------------------

def find_equivalence_classes(
    relation_pairs: list[tuple[str, str]],
) -> list[list[str]]:
    """Find groups of stocks that mutually dominate each other (tied stocks).

    Two stocks A and B are in the same equivalence class when both
    (A, B) and (B, A) are present in the relation.

    Uses a union-find (disjoint-set) data structure implemented from scratch.

    Parameters
    ----------
    relation_pairs:
        Full dominance relation pairs.

    Returns
    -------
    list[list[str]]
        Each inner list is a sorted group of tied/equivalent stocks.
        Groups of size 1 (isolated stocks) are included.
    """
    # Collect all vertices
    vertices: set[str] = set()
    for a, b in relation_pairs:
        vertices.add(a)
        vertices.add(b)

    # Union-Find data structures
    parent: dict[str, str] = {v: v for v in vertices}
    rank: dict[str, int] = {v: 0 for v in vertices}

    def _find(x: str) -> str:
        """Path-compressed find."""
        root = x
        while parent[root] != root:
            root = parent[root]
        # Path compression
        while parent[x] != root:
            nxt = parent[x]
            parent[x] = root
            x = nxt
        return root

    def _union(x: str, y: str) -> None:
        """Union by rank."""
        rx, ry = _find(x), _find(y)
        if rx == ry:
            return
        if rank[rx] < rank[ry]:
            rx, ry = ry, rx
        parent[ry] = rx
        if rank[rx] == rank[ry]:
            rank[rx] += 1

    # Build set of all pairs for O(1) lookup
    pair_set: set[tuple[str, str]] = set(relation_pairs)

    # Union mutually dominating stocks
    for a, b in relation_pairs:
        if a != b and (b, a) in pair_set:
            _union(a, b)

    # Cluster by root
    clusters: dict[str, list[str]] = defaultdict(list)
    for v in vertices:
        clusters[_find(v)].append(v)

    return [sorted(group) for group in clusters.values()]


# ---------------------------------------------------------------------------
# Level Assignment via Topological Sort (from scratch)
# ---------------------------------------------------------------------------

def build_hasse_levels(
    cover_relation: list[tuple[str, str]],
    tickers: list[str],
) -> dict[str, int]:
    """Assign a display level to each node in the Hasse DAG.

    Level 0 = maximal elements (nothing dominates them from above in the cover).
    Higher level numbers = more dominated (further down the diagram).

    Uses Kahn's algorithm for topological sort, then assigns levels based
    on longest path from maximal elements.

    Parameters
    ----------
    cover_relation:
        Output of :func:`build_cover_relation` (directed edges A -> B meaning A covers B).
    tickers:
        Full list of stock symbols.

    Returns
    -------
    dict[str, int]
        {ticker: level} where 0 is the top (maximal) level.
    """
    # Build in-degree and adjacency for the cover DAG
    # Edge (A, B): A covers B, so A is "above" B
    children: dict[str, list[str]] = {t: [] for t in tickers}   # A -> [B, ...]
    parents: dict[str, list[str]] = {t: [] for t in tickers}    # B -> [A, ...]

    for a, b in cover_relation:
        if a in children:
            children[a].append(b)
        if b in parents:
            parents[b].append(a)

    # Level = longest path FROM a source (maximal element)
    # Sources have no parents in the cover DAG
    levels: dict[str, int] = {t: 0 for t in tickers}

    # BFS/relaxation: process nodes in topological order
    # In-degree from parents perspective: how many nodes point TO this node
    in_degree: dict[str, int] = {t: len(parents[t]) for t in tickers}

    queue: deque[str] = deque()
    for t in tickers:
        if in_degree[t] == 0:
            queue.append(t)
            levels[t] = 0

    while queue:
        node = queue.popleft()
        for child in children[node]:
            levels[child] = max(levels[child], levels[node] + 1)
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)

    return levels


# ---------------------------------------------------------------------------
# Maximal and Minimal Elements
# ---------------------------------------------------------------------------

def find_maximal_elements(
    cover_relation: list[tuple[str, str]],
    tickers: list[str],
) -> list[str]:
    """Find nodes with no incoming edges in the cover DAG (top of diagram).

    Maximal elements are stocks not covered by any other stock.

    Parameters
    ----------
    cover_relation:
        Cover edges (A, B) meaning A is directly above B.
    tickers:
        Full list of stock symbols.

    Returns
    -------
    list[str]
        Sorted list of maximal element tickers.
    """
    has_incoming: set[str] = {b for _, b in cover_relation}
    maximal = [t for t in tickers if t not in has_incoming]
    return sorted(maximal)


def find_minimal_elements(
    cover_relation: list[tuple[str, str]],
    tickers: list[str],
) -> list[str]:
    """Find nodes with no outgoing edges in the cover DAG (bottom of diagram).

    Minimal elements are stocks that do not cover any other stock.

    Parameters
    ----------
    cover_relation:
        Cover edges (A, B) meaning A is directly above B.
    tickers:
        Full list of stock symbols.

    Returns
    -------
    list[str]
        Sorted list of minimal element tickers.
    """
    has_outgoing: set[str] = {a for a, _ in cover_relation}
    minimal = [t for t in tickers if t not in has_outgoing]
    return sorted(minimal)


# ---------------------------------------------------------------------------
# Hasse Analysis
# ---------------------------------------------------------------------------

def analyze_hasse(
    cover_relation: list[tuple[str, str]],
    tickers: list[str],
) -> dict:
    """Comprehensive analysis of the Hasse diagram structure.

    Parameters
    ----------
    cover_relation:
        Output of :func:`build_cover_relation`.
    tickers:
        Full list of stock symbols.

    Returns
    -------
    dict
        Keys: cover_relation, levels, maximal_elements, minimal_elements,
        num_cover_edges, structure_description.
    """
    levels = build_hasse_levels(cover_relation, tickers)
    maximal = find_maximal_elements(cover_relation, tickers)
    minimal = find_minimal_elements(cover_relation, tickers)

    num_levels = max(levels.values(), default=0) + 1

    if num_levels == 1:
        structure_description = (
            "Single-level diagram: all stocks are incomparable "
            "(no stock dominates another)."
        )
    elif len(maximal) == 1:
        structure_description = (
            f"Pyramid structure: single maximal element '{maximal[0]}' "
            f"at top, {num_levels} levels deep."
        )
    else:
        structure_description = (
            f"Multi-root Hasse diagram with {len(maximal)} maximal element(s), "
            f"{len(minimal)} minimal element(s), and {num_levels} level(s)."
        )

    return {
        "cover_relation": [list(edge) for edge in cover_relation],
        "levels": levels,
        "maximal_elements": maximal,
        "minimal_elements": minimal,
        "num_cover_edges": len(cover_relation),
        "structure_description": structure_description,
    }


# ---------------------------------------------------------------------------
# Layout for Plotly Visualization
# ---------------------------------------------------------------------------

def get_hasse_layout(
    cover_relation: list[tuple[str, str]],
    tickers: list[str],
) -> dict[str, tuple[float, float]]:
    """Compute (x, y) positions for each node for Plotly visualization.

    Arrangement:
    - y=1.0 for maximal elements (level 0), y=0.0 for deepest level.
    - x positions distribute nodes evenly within each level.

    Parameters
    ----------
    cover_relation:
        Output of :func:`build_cover_relation`.
    tickers:
        Full list of stock symbols.

    Returns
    -------
    dict[str, tuple[float, float]]
        {ticker: (x, y)} positions in [0,1] x [0,1] space.
    """
    levels = build_hasse_levels(cover_relation, tickers)

    max_level = max(levels.values(), default=0)

    # Group tickers by their level
    level_groups: dict[int, list[str]] = defaultdict(list)
    for ticker in tickers:
        level_groups[levels[ticker]].append(ticker)

    # Sort each level group for determinism
    for lvl in level_groups:
        level_groups[lvl].sort()

    positions: dict[str, tuple[float, float]] = {}

    for lvl, members in level_groups.items():
        n = len(members)
        # y: level 0 -> y=1.0, deeper levels -> lower y values
        if max_level == 0:
            y = 1.0
        else:
            y = 1.0 - (lvl / max_level)

        # Distribute x evenly in (0, 1)
        for i, ticker in enumerate(members):
            if n == 1:
                x = 0.5
            else:
                x = i / (n - 1)
            positions[ticker] = (round(x, 4), round(y, 4))

    return positions
