"""
coloring.py — Welsh-Powell Graph Coloring (from scratch)
=========================================================
Implements Welsh-Powell graph coloring algorithm completely from scratch.
No calls to nx.coloring.greedy_color() or any NetworkX coloring helpers.

Used in TradeSense V2 to assign non-competing color groups to correlated stocks,
so that stocks in the same color group share no strong correlation edges.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Core Algorithm
# ---------------------------------------------------------------------------

def welsh_powell(
    adj: dict[str, list[tuple[str, float]]]
) -> tuple[dict[str, int], list[str]]:
    """Welsh-Powell graph coloring algorithm implemented step by step.

    Parameters
    ----------
    adj:
        Adjacency dict mapping each vertex (stock ticker) to a list of
        (neighbour_ticker, edge_weight) tuples.  Only the neighbour name is
        used for coloring; edge weights are ignored.

    Returns
    -------
    coloring_dict : dict[str, int]
        Maps every stock ticker to a color integer (1-based).
    step_trace : list[str]
        One human-readable string per vertex assignment documenting the
        reasoning at each step.

    Algorithm
    ---------
    Step 1 - Compute degree of every vertex.
    Step 2 - Sort vertices in descending degree order (stable sort).
    Step 3 - Iterate sorted list; for each uncolored vertex find the
             lowest positive integer not used by any already-colored
             neighbor, and assign it.
    """

    # ------------------------------------------------------------------
    # Step 1: Calculate degree of every vertex
    # ------------------------------------------------------------------
    degrees: dict[str, int] = {v: len(neighbours) for v, neighbours in adj.items()}

    # ------------------------------------------------------------------
    # Step 2: Sort vertices descending by degree (stable for determinism)
    # ------------------------------------------------------------------
    sorted_vertices: list[str] = sorted(
        adj.keys(),
        key=lambda v: degrees[v],
        reverse=True,
    )

    # ------------------------------------------------------------------
    # Step 3: Greedy coloring in sorted order
    # ------------------------------------------------------------------
    coloring_dict: dict[str, int] = {}
    step_trace: list[str] = []

    step_trace.append(
        "Step 1: Degrees computed - "
        + ", ".join(f"{v}={degrees[v]}" for v in sorted_vertices)
    )
    step_trace.append(
        "Step 2: Sorted vertices (desc degree) - "
        + ", ".join(sorted_vertices)
    )

    for vertex in sorted_vertices:
        # Collect colors already used by neighbors
        neighbor_colors: set[int] = set()
        colored_neighbors: list[str] = []

        for neighbour, _weight in adj[vertex]:
            if neighbour in coloring_dict:
                neighbor_colors.add(coloring_dict[neighbour])
                colored_neighbors.append(f"{neighbour}->color{coloring_dict[neighbour]}")

        # Find lowest positive integer not in neighbor_colors
        color = 1
        while color in neighbor_colors:
            color += 1

        coloring_dict[vertex] = color

        step_trace.append(
            f"Step 3: Assign Color {color} to {vertex} "
            f"(degree={degrees[vertex]}, neighbors with color: {colored_neighbors})"
        )

    return coloring_dict, step_trace


# ---------------------------------------------------------------------------
# Validation Helpers
# ---------------------------------------------------------------------------

def validate_coloring(
    adj: dict[str, list[tuple[str, float]]],
    coloring: dict[str, int],
) -> tuple[bool, list[str]]:
    """Verify that no two adjacent vertices share the same color.

    Parameters
    ----------
    adj:
        Adjacency dict (same format as passed to :func:`welsh_powell`).
    coloring:
        Color assignment returned by :func:`welsh_powell`.

    Returns
    -------
    is_valid : bool
        True if the coloring is proper (no conflict).
    violations : list[str]
        Human-readable description of every conflicting edge found.
    """
    violations: list[str] = []

    seen_edges: set[frozenset] = set()

    for u, neighbours in adj.items():
        for v, _weight in neighbours:
            edge = frozenset({u, v})
            if edge in seen_edges:
                continue
            seen_edges.add(edge)

            cu = coloring.get(u)
            cv = coloring.get(v)

            if cu is None or cv is None:
                violations.append(
                    f"Edge ({u}, {v}): one or both vertices uncolored"
                )
            elif cu == cv:
                violations.append(
                    f"Edge ({u}, {v}): both assigned Color {cu} - CONFLICT"
                )

    is_valid = len(violations) == 0
    return is_valid, violations


def get_color_groups(coloring: dict[str, int]) -> dict[int, list[str]]:
    """Group stocks by their assigned color.

    Parameters
    ----------
    coloring:
        {stock_ticker: color_id} mapping.

    Returns
    -------
    dict[int, list[str]]
        {color_id: [sorted list of stock tickers]} ordered by color_id.
    """
    groups: dict[int, list[str]] = {}

    for stock, color in coloring.items():
        groups.setdefault(color, []).append(stock)

    # Sort tickers within each group for determinism
    for color in groups:
        groups[color].sort()

    # Return ordered by color_id
    return dict(sorted(groups.items()))


def verify_independent_sets(
    adj: dict[str, list[tuple[str, float]]],
    color_groups: dict[int, list[str]],
) -> dict[int, bool]:
    """Verify that each color group is a proper independent set.

    An independent set means no two members of the group share an edge.

    Parameters
    ----------
    adj:
        Adjacency dict.
    color_groups:
        Output of :func:`get_color_groups`.

    Returns
    -------
    dict[int, bool]
        {color_id: True} if the group is independent, False otherwise.
    """
    results: dict[int, bool] = {}

    for color_id, members in color_groups.items():
        member_set = set(members)
        is_independent = True

        for u in members:
            for v, _weight in adj.get(u, []):
                if v in member_set:
                    is_independent = False
                    break
            if not is_independent:
                break

        results[color_id] = is_independent

    return results


# ---------------------------------------------------------------------------
# Exact Chromatic Number (backtracking)
# ---------------------------------------------------------------------------

def exact_chromatic_number(
    adj: dict[str, list[tuple[str, float]]],
    max_vertices: int = 15,
) -> dict[str, Any]:
    """Compute the exact chromatic number via backtracking search.

    For graphs larger than max_vertices the problem is NP-hard and this
    function returns early without searching.

    Parameters
    ----------
    adj:
        Adjacency dict.
    max_vertices:
        Maximum graph size for which exact computation is attempted.

    Returns
    -------
    dict
        {exact: True, chromatic_number: k} on success, or
        {exact: False, reason: <str>} when the graph is too large.
    """
    vertices = list(adj.keys())
    n = len(vertices)

    if n == 0:
        return {"exact": True, "chromatic_number": 0}

    if n > max_vertices:
        return {
            "exact": False,
            "reason": f"Graph too large (NP-hard) - {n} vertices > limit {max_vertices}",
        }

    neighbour_sets: dict[str, set[str]] = {
        v: {nb for nb, _ in adj.get(v, [])} for v in vertices
    }

    # Try k from lower_bound (max clique or 1) up to upper bound
    # Order vertices by degree descending to fail early
    sorted_v_indices = sorted(range(n), key=lambda i: len(neighbour_sets[vertices[i]]), reverse=True)
    v_order = [vertices[i] for i in sorted_v_indices]
    idx_map = {v: i for i, v in enumerate(v_order)}

    # Step budget to prevent NP-hard exponential stall
    calls = 0
    max_calls = 5000

    def _is_safe_fast(vertex_idx: int, color: int, assignment: list[int]) -> bool:
        v = v_order[vertex_idx]
        for nb in neighbour_sets[v]:
            if nb in idx_map:
                nb_pos = idx_map[nb]
                if assignment[nb_pos] == color:
                    return False
        return True

    def _backtrack_fast(pos: int, assignment: list[int], k: int) -> bool:
        nonlocal calls
        calls += 1
        if calls > max_calls:
            return False
        if pos == n:
            return True
        for color in range(1, k + 1):
            if _is_safe_fast(pos, color, assignment):
                assignment[pos] = color
                if _backtrack_fast(pos + 1, assignment, k):
                    return True
                assignment[pos] = 0
        return False

    # Find Welsh-Powell upper bound first
    wp_colors, _ = welsh_powell(adj)
    wp_upper = len(set(wp_colors.values())) if wp_colors else n

    for k in range(1, wp_upper + 1):
        calls = 0
        assignment = [0] * n
        if _backtrack_fast(0, assignment, k):
            return {"exact": True, "chromatic_number": k}
        if calls >= max_calls:
            return {
                "exact": False,
                "reason": f"Exact search budget exceeded ({max_calls} states) — Welsh-Powell upper bound is {wp_upper}",
            }

    return {"exact": False, "reason": "Welsh-Powell upper bound used"}

