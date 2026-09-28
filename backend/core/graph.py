"""
graph.py
--------
Graph G = (V, E) construction and discrete-mathematics graph algorithms
for TradeSense V2.

All graph ALGORITHMS are implemented from scratch — BFS, DFS, connected
components, degree calculation, density — without delegating to NetworkX
algorithm functions.  NetworkX is imported **only** to provide the
``nx.Graph`` object used by external visualisation libraries.

Key functions
-------------
build_adjacency_dict          -- weighted adjacency list from corr matrix
build_adjacency_matrix        -- weighted adjacency matrix
bfs                           -- breadth-first search with step trace
dfs                           -- depth-first search (iterative) with step trace
connected_components_from_scratch -- union of BFS visits
degree / all_degrees          -- node degree helpers
graph_density                 -- 2E / (V(V-1))
calculate_graph_statistics    -- full stats dict
get_strong_relationships      -- top-N edges by weight
get_networkx_graph            -- convert adj dict to nx.Graph for layout
"""

from __future__ import annotations

from collections import deque
from typing import Any

import pandas as pd
import networkx as nx


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_adjacency_dict(
    tickers: list[str],
    corr_df: pd.DataFrame,
    threshold: float,
) -> dict[str, list[tuple[str, float]]]:
    """Build a weighted adjacency list from a correlation matrix.

    An edge (u, v) exists when ``|corr(u, v)| >= threshold``.
    The graph is undirected — each edge appears in both adj[u] and adj[v].

    Parameters
    ----------
    tickers:
        Ordered list of ticker symbols (must match corr_df columns/index).
    corr_df:
        Square Pearson correlation DataFrame.
    threshold:
        Minimum absolute correlation to include an edge (e.g. 0.5).

    Returns
    -------
    dict[str, list[tuple[str, float]]]
        ``{ticker: [(neighbour, correlation_weight), ...]}``
        Neighbours are sorted by descending absolute weight.
    """
    adj: dict[str, list[tuple[str, float]]] = {t: [] for t in tickers}

    n = len(tickers)
    for i in range(n):
        for j in range(i + 1, n):
            u = tickers[i]
            v = tickers[j]
            if u not in corr_df.columns or v not in corr_df.columns:
                continue
            corr_val: float = float(corr_df.loc[u, v])
            if pd.isna(corr_val):
                continue
            if abs(corr_val) >= threshold:
                weight = round(corr_val, 6)
                adj[u].append((v, weight))
                adj[v].append((u, weight))

    # Sort each neighbour list by descending absolute weight
    for node in adj:
        adj[node].sort(key=lambda t: abs(t[1]), reverse=True)

    return adj


def build_adjacency_matrix(
    tickers: list[str],
    corr_df: pd.DataFrame,
    threshold: float,
) -> tuple[list[str], list[list[float]]]:
    """Build a weighted adjacency matrix from a correlation matrix.

    Parameters
    ----------
    tickers:
        Ordered list of ticker symbols.
    corr_df:
        Square Pearson correlation DataFrame.
    threshold:
        Minimum absolute correlation to include an edge.

    Returns
    -------
    tuple[list[str], list[list[float]]]
        * ``ticker_list`` – the ordered ticker list used as row/col labels.
        * ``matrix``      – N×N 2-D list; ``matrix[i][j]`` = correlation
          weight if an edge exists, else ``0.0``.  Diagonal = ``0.0``.
    """
    n = len(tickers)
    matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(i + 1, n):
            u = tickers[i]
            v = tickers[j]
            if u not in corr_df.columns or v not in corr_df.columns:
                continue
            corr_val: float = float(corr_df.loc[u, v])
            if pd.isna(corr_val):
                continue
            if abs(corr_val) >= threshold:
                w = round(corr_val, 6)
                matrix[i][j] = w
                matrix[j][i] = w

    return tickers, matrix


# ---------------------------------------------------------------------------
# BFS — Breadth-First Search (from scratch)
# ---------------------------------------------------------------------------

def bfs(
    adj: dict[str, list[tuple[str, float]]],
    start: str,
) -> tuple[list[str], list[str]]:
    """Breadth-first search starting from *start*.

    Traverses all nodes reachable from *start* in the adjacency dict.
    Nodes are visited in FIFO queue order; neighbours are processed in
    the order they appear in the adjacency list.

    Parameters
    ----------
    adj:
        Weighted adjacency dict as returned by :func:`build_adjacency_dict`.
    start:
        Starting node label (ticker symbol).

    Returns
    -------
    tuple[list[str], list[str]]
        * ``traversal_order`` – ordered list of visited node labels.
        * ``step_trace``      – human-readable strings describing each
          step, e.g.
          ``"Visit TCS.NS → neighbours: [INFY.NS, HCLTECH.NS]"``
    """
    if start not in adj:
        return [], [f"Node '{start}' not found in graph."]

    visited: dict[str, bool] = {node: False for node in adj}
    queue: deque[str] = deque()
    traversal_order: list[str] = []
    step_trace: list[str] = []

    visited[start] = True
    queue.append(start)
    step_trace.append(f"Initialise queue: enqueue '{start}'")

    while queue:
        node = queue.popleft()
        traversal_order.append(node)
        neighbours = [nb for nb, _ in adj[node]]
        step_trace.append(
            f"Visit {node} \u2192 neighbours: [{', '.join(neighbours) if neighbours else 'none'}]"
        )

        for nb, _ in adj[node]:
            if not visited.get(nb, True):
                visited[nb] = True
                queue.append(nb)
                step_trace.append(f"  Enqueue '{nb}' (unvisited neighbour of {node})")

    return traversal_order, step_trace


# ---------------------------------------------------------------------------
# DFS — Depth-First Search iterative (from scratch)
# ---------------------------------------------------------------------------

def dfs(
    adj: dict[str, list[tuple[str, float]]],
    start: str,
) -> tuple[list[str], list[str]]:
    """Iterative depth-first search starting from *start*.

    Uses an explicit stack.  To achieve the expected DFS ordering,
    neighbours are pushed onto the stack in *reverse* order so that the
    first neighbour in the adjacency list is processed first.

    Parameters
    ----------
    adj:
        Weighted adjacency dict as returned by :func:`build_adjacency_dict`.
    start:
        Starting node label.

    Returns
    -------
    tuple[list[str], list[str]]
        * ``traversal_order`` – ordered list of visited node labels.
        * ``step_trace``      – human-readable step descriptions, e.g.
          ``"Visit TCS.NS → stack top; push neighbours: [INFY.NS]"``
    """
    if start not in adj:
        return [], [f"Node '{start}' not found in graph."]

    visited: set[str] = set()
    stack: list[str] = [start]
    traversal_order: list[str] = []
    step_trace: list[str] = []

    step_trace.append(f"Initialise stack: push '{start}'")

    while stack:
        node = stack.pop()
        if node in visited:
            step_trace.append(f"  Skip '{node}' (already visited)")
            continue

        visited.add(node)
        traversal_order.append(node)
        neighbours = [nb for nb, _ in adj[node]]
        step_trace.append(
            f"Visit {node} \u2192 stack top; "
            f"push neighbours (reversed): [{', '.join(reversed(neighbours)) if neighbours else 'none'}]"
        )

        # Push in reverse so first neighbour is processed next
        for nb, _ in reversed(adj[node]):
            if nb not in visited:
                stack.append(nb)
                step_trace.append(f"  Push '{nb}' onto stack")

    return traversal_order, step_trace


# ---------------------------------------------------------------------------
# Connected components (from scratch — no networkx)
# ---------------------------------------------------------------------------

def connected_components_from_scratch(
    adj: dict[str, list[tuple[str, float]]],
) -> list[list[str]]:
    """Find all connected components using BFS from each unvisited node.

    Parameters
    ----------
    adj:
        Weighted adjacency dict.

    Returns
    -------
    list[list[str]]
        Each inner list is one connected component (sorted alphabetically).
        The outer list is sorted by descending component size.
    """
    visited: set[str] = set()
    components: list[list[str]] = []

    for start in adj:
        if start in visited:
            continue
        # BFS to discover the full component
        component: list[str] = []
        queue: deque[str] = deque([start])
        visited.add(start)
        while queue:
            node = queue.popleft()
            component.append(node)
            for nb, _ in adj[node]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(sorted(component))

    # Sort components by descending size
    components.sort(key=len, reverse=True)
    return components


# ---------------------------------------------------------------------------
# Degree helpers (from scratch)
# ---------------------------------------------------------------------------

def degree(adj: dict[str, list[tuple[str, float]]], node: str) -> int:
    """Return the degree (number of edges) of *node*.

    Parameters
    ----------
    adj:
        Weighted adjacency dict.
    node:
        Target node label.

    Returns
    -------
    int
        Number of neighbours.  Returns 0 if the node is not in *adj*.
    """
    return len(adj.get(node, []))


def all_degrees(adj: dict[str, list[tuple[str, float]]]) -> dict[str, int]:
    """Compute the degree of every node in the graph.

    Parameters
    ----------
    adj:
        Weighted adjacency dict.

    Returns
    -------
    dict[str, int]
        ``{node: degree}`` for all nodes.
    """
    return {node: len(neighbours) for node, neighbours in adj.items()}


# ---------------------------------------------------------------------------
# Graph density (from scratch)
# ---------------------------------------------------------------------------

def graph_density(num_vertices: int, num_edges: int) -> float:
    """Compute graph density for a simple undirected graph.

    ``density = 2 * E / (V * (V - 1))``

    Parameters
    ----------
    num_vertices:
        Total number of vertices (V).
    num_edges:
        Total number of edges (E).

    Returns
    -------
    float
        Density in [0.0, 1.0].  Returns 0.0 when V < 2.
    """
    if num_vertices < 2:
        return 0.0
    max_edges: int = num_vertices * (num_vertices - 1)
    return (2 * num_edges) / max_edges


# ---------------------------------------------------------------------------
# Graph statistics
# ---------------------------------------------------------------------------

def calculate_graph_statistics(
    adj: dict[str, list[tuple[str, float]]],
    tickers: list[str],
) -> dict[str, Any]:
    """Compute a comprehensive set of graph-level statistics.

    Parameters
    ----------
    adj:
        Weighted adjacency dict (may be a sub-graph of all tickers).
    tickers:
        Complete list of ticker labels (used as V count).

    Returns
    -------
    dict with keys:
        ``num_vertices``, ``num_edges``, ``avg_degree``,
        ``max_degree``, ``min_degree``, ``density``,
        ``connected_components``, ``most_connected_stock``.
    """
    num_vertices: int = len(tickers)

    # Count unique edges (each edge appears twice in adj list)
    num_edges: int = sum(len(nbrs) for nbrs in adj.values()) // 2

    degrees: dict[str, int] = all_degrees(adj)

    if degrees:
        avg_deg: float = sum(degrees.values()) / len(degrees)
        max_deg: int = max(degrees.values())
        min_deg: int = min(degrees.values())
        most_connected: str = max(degrees, key=lambda n: degrees[n])
    else:
        avg_deg = 0.0
        max_deg = 0
        min_deg = 0
        most_connected = "N/A"

    density: float = graph_density(num_vertices, num_edges)
    components: list[list[str]] = connected_components_from_scratch(adj)

    return {
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "avg_degree": round(avg_deg, 4),
        "max_degree": max_deg,
        "min_degree": min_deg,
        "density": round(density, 6),
        "connected_components": len(components),
        "most_connected_stock": most_connected,
    }


# ---------------------------------------------------------------------------
# Strong relationships — top-N edges by weight
# ---------------------------------------------------------------------------

def get_strong_relationships(
    adj: dict[str, list[tuple[str, float]]],
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Return the top-N edges sorted by descending absolute correlation.

    Each unique edge (u, v) is returned only once (u < v alphabetically).

    Parameters
    ----------
    adj:
        Weighted adjacency dict.
    top_n:
        Number of top edges to return.

    Returns
    -------
    list[dict]
        Each element:
        ``{'stock_a': str, 'stock_b': str, 'correlation': float,
           'relationship': str}``
        where ``relationship`` is ``'Strong Positive'``,
        ``'Moderate Positive'``, ``'Strong Negative'``, or
        ``'Moderate Negative'``.
    """
    seen: set[frozenset[str]] = set()
    all_edges: list[tuple[str, str, float]] = []

    for node, neighbours in adj.items():
        for nb, weight in neighbours:
            key = frozenset({node, nb})
            if key not in seen:
                seen.add(key)
                all_edges.append((node, nb, weight))

    # Sort by descending absolute weight
    all_edges.sort(key=lambda e: abs(e[2]), reverse=True)
    top_edges = all_edges[:top_n]

    def _relationship(w: float) -> str:
        if w >= 0.7:
            return "Strong Positive"
        if w >= 0.3:
            return "Moderate Positive"
        if w <= -0.7:
            return "Strong Negative"
        return "Moderate Negative"

    result: list[dict[str, Any]] = []
    for u, v, w in top_edges:
        result.append(
            {
                "stock_a": u,
                "stock_b": v,
                "correlation": round(w, 6),
                "relationship": _relationship(w),
            }
        )
    return result


# ---------------------------------------------------------------------------
# Convert to NetworkX Graph (for visualisation layout only)
# ---------------------------------------------------------------------------

def get_networkx_graph(
    adj: dict[str, list[tuple[str, float]]],
) -> nx.Graph:
    """Convert the adjacency dict to a ``networkx.Graph`` for layout use.

    This function is the ONLY place NetworkX is used.  It is intended
    exclusively for external visualisation engines that need an ``nx.Graph``
    object (e.g. to compute spring-layout node positions).  No graph
    algorithm from NetworkX is called here or anywhere else in this module.

    Parameters
    ----------
    adj:
        Weighted adjacency dict.

    Returns
    -------
    nx.Graph
        Undirected weighted graph with edge attribute ``weight``.
    """
    G: nx.Graph = nx.Graph()
    G.add_nodes_from(adj.keys())

    seen: set[frozenset[str]] = set()
    for node, neighbours in adj.items():
        for nb, weight in neighbours:
            key = frozenset({node, nb})
            if key not in seen:
                seen.add(key)
                G.add_edge(node, nb, weight=weight)

    return G
