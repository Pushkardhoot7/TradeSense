"""
Graph Coloring Module for TradeSense

Implements graph coloring on the correlation graph G = (V, E) such that for every edge (u, v) in E, Color(u) != Color(v).

Discrete Mathematics Concept:
A proper graph coloring assigns a color to each vertex such that no adjacent vertices share the same color.
- Chromatic Number chi(G): The minimum number of colors required for a proper coloring of G.
- Greedy Coloring: Computes a valid coloring that provides an upper bound on chi(G).
- Exact Computation: Computed via backtracking for graphs with |V| <= 15.

Stock Market Framing:
Color groups represent mathematical partitions of non-correlated stocks (Independent Sets). They are structural graph properties, not investment signals or safety ratings.
"""

from typing import Dict, Any, List, Optional
import networkx as nx


def color_graph(graph: nx.Graph, strategy: str = "largest_first") -> Dict[str, int]:
    """
    Apply NetworkX greedy coloring (Welsh-Powell largest_first by default) to correlation graph.

    Args:
        graph: NetworkX Graph.
        strategy: Greedy coloring strategy ('largest_first', 'random_sequential', 'smallest_last', etc.).

    Returns:
        Mapping of {stock_symbol: color_id}.
    """
    if graph is None or graph.number_of_nodes() == 0:
        return {}

    try:
        coloring = nx.coloring.greedy_color(graph, strategy=strategy)
    except Exception:
        # Fallback to default greedy strategy if custom strategy fails
        coloring = nx.coloring.greedy_color(graph, strategy="largest_first")

    return coloring


def validate_coloring(graph: nx.Graph, coloring: Dict[str, int]) -> bool:
    """
    Verify that for every edge (u, v) in the graph, coloring[u] != coloring[v].
    Performs an explicit check over all edges in the graph.

    Args:
        graph: NetworkX Graph.
        coloring: Mapping of {stock_symbol: color_id}.

    Returns:
        True if coloring is valid (no adjacent vertices share a color), False otherwise.
    """
    if graph is None or graph.number_of_nodes() == 0:
        return True

    if not coloring:
        return False

    for u, v in graph.edges():
        if u not in coloring or v not in coloring:
            return False
        if coloring[u] == coloring[v]:
            return False

    return True


def get_color_groups(coloring: Dict[str, int]) -> Dict[int, List[str]]:
    """
    Return color groups {color_id: [stocks with that color]} for display.

    Args:
        coloring: Mapping of {stock_symbol: color_id}.

    Returns:
        Dictionary mapping color_id -> sorted list of stock symbols.
    """
    if not coloring:
        return {}

    groups: Dict[int, List[str]] = {}
    for stock, color_id in coloring.items():
        if color_id not in groups:
            groups[color_id] = []
        groups[color_id].append(stock)

    sorted_groups = {c_id: sorted(stocks) for c_id, stocks in sorted(groups.items())}
    return sorted_groups


def calculate_exact_chromatic_number_if_feasible(graph: nx.Graph, max_vertices: int = 15) -> Dict[str, Any]:
    """
    If graph.number_of_nodes() <= max_vertices, compute the exact chromatic number chi(G)
    via incremental backtracking k-colorability check.

    Args:
        graph: NetworkX Graph.
        max_vertices: Maximum vertex count allowed for exact NP-hard computation (default 15).

    Returns:
        Dict: {"exact": True, "chromatic_number": k} if computed, or
              {"exact": False, "reason": "graph too large for exact computation"} otherwise.
    """
    if graph is None or graph.number_of_nodes() == 0:
        return {"exact": True, "chromatic_number": 0}

    num_nodes = graph.number_of_nodes()
    if num_nodes > max_vertices:
        return {
            "exact": False,
            "reason": f"Graph has {num_nodes} vertices (cutoff is ≤ {max_vertices}). Exact chromatic number is NP-hard.",
        }

    nodes = list(graph.nodes())

    def is_k_colorable(k_colors: int) -> bool:
        color_assign = {}

        def backtrack(node_idx: int) -> bool:
            if node_idx == len(nodes):
                return True

            node = nodes[node_idx]
            used_colors = {color_assign[nbr] for nbr in graph.neighbors(node) if nbr in color_assign}

            for color in range(k_colors):
                if color not in used_colors:
                    color_assign[node] = color
                    if backtrack(node_idx + 1):
                        return True
                    del color_assign[node]

            return False

        return backtrack(0)

    # Search for minimum k from 1 to num_nodes
    for k in range(1, num_nodes + 1):
        if is_k_colorable(k):
            return {"exact": True, "chromatic_number": k}

    return {"exact": True, "chromatic_number": num_nodes}


# Legacy compatibility helper functions
def color_stock_graph(G: nx.Graph) -> Dict[str, int]:
    """Legacy helper compatibility."""
    return color_graph(G)


def get_independent_sets(coloring_dict: Dict[str, int]) -> Dict[int, List[str]]:
    """Legacy helper compatibility."""
    return get_color_groups(coloring_dict)
