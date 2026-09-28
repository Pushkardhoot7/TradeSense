"""
Graph Analysis Module for TradeSense

Converts Pearson correlation matrices into an undirected Graph G = (V, E) using NetworkX, computes network topology statistics, and identifies strong stock relationships.

Discrete Mathematics Concept:
G = (V, E)
- Vertices (V): Stock tickers in the selection.
- Edges (E): Pairwise stock relationships satisfying correlation >= threshold.
- Edge Weight: Pearson correlation coefficient.
- Edge Rule Constraints: No self-loops (i != j), no duplicate edges (undirected).
"""

from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
import pandas as pd


def build_correlation_graph(correlation_matrix: pd.DataFrame, threshold: float = 0.70) -> nx.Graph:
    """
    Build an undirected graph G = (V, E) from the correlation matrix per Section 2 edge rule.
    - No self-loops (never connect stock i to itself even though A[i,i] = 1).
    - No duplicate edges (undirected graph pair i-j counted once).
    - Edge created iff correlation(i, j) >= threshold.
    - Edge weight = correlation.

    Args:
        correlation_matrix: Square Pearson correlation DataFrame.
        threshold: Minimum correlation value required to create an edge (default 0.70).

    Returns:
        NetworkX Graph object.
    """
    G = nx.Graph()

    if correlation_matrix is None or correlation_matrix.empty:
        return G

    tickers = list(correlation_matrix.columns)
    G.add_nodes_from(tickers)

    num_tickers = len(tickers)
    for i in range(num_tickers):
        for j in range(i + 1, num_tickers):
            stock_i = tickers[i]
            stock_j = tickers[j]
            corr_val = float(correlation_matrix.iloc[i, j])

            if corr_val >= threshold:
                G.add_edge(stock_i, stock_j, weight=round(corr_val, 4))

    return G


def calculate_graph_statistics(graph: nx.Graph) -> Dict[str, Any]:
    """
    Compute graph statistics per Section 4:
    - Vertices (count)
    - Edges (count)
    - Average degree
    - Highest-degree stock (and its degree)
    - Connected components (count, and list of stock groups)
    - Density

    Args:
        graph: NetworkX Graph.

    Returns:
        Dictionary of graph statistics.
    """
    if graph is None or graph.number_of_nodes() == 0:
        return {
            "num_vertices": 0,
            "num_edges": 0,
            "avg_degree": 0.0,
            "highest_degree_stock": {"stock": "N/A", "degree": 0},
            "num_connected_components": 0,
            "connected_components": [],
            "density": 0.0,
        }

    num_vertices = graph.number_of_nodes()
    num_edges = graph.number_of_edges()

    degrees = dict(graph.degree())
    avg_degree = float(sum(degrees.values()) / float(num_vertices)) if num_vertices > 0 else 0.0

    if degrees:
        highest_stock = max(degrees, key=degrees.get)
        highest_degree = degrees[highest_stock]
    else:
        highest_stock = "N/A"
        highest_degree = 0

    components = [sorted(list(c)) for c in nx.connected_components(graph)]
    density = float(nx.density(graph))

    return {
        "num_vertices": num_vertices,
        "num_edges": num_edges,
        "avg_degree": round(avg_degree, 2),
        "highest_degree_stock": {"stock": highest_stock, "degree": highest_degree},
        "num_connected_components": len(components),
        "connected_components": components,
        "density": round(density, 4),
    }


def get_degree(graph: nx.Graph, stock: str) -> int:
    """
    Return the degree (number of edges) for a given stock/vertex.

    Args:
        graph: NetworkX Graph.
        stock: Stock ticker symbol.

    Returns:
        Integer degree of the stock vertex.
    """
    if graph is None or not graph.has_node(stock):
        return 0
    return int(graph.degree(stock))


def get_connected_components(graph: nx.Graph) -> List[List[str]]:
    """
    Return the connected components of the graph, as groups of stock symbols.

    Args:
        graph: NetworkX Graph.

    Returns:
        List of lists, where each sublist contains the stock symbols in a connected component.
    """
    if graph is None or graph.number_of_nodes() == 0:
        return []

    return [sorted(list(c)) for c in nx.connected_components(graph)]


def get_strong_relationships(graph: nx.Graph, top_n: int = 10) -> List[Dict[str, Any]]:
    """
    Return the top-N highest-weight edges (strongest correlated pairs) in the graph.

    Args:
        graph: NetworkX Graph.
        top_n: Maximum number of edges to return.

    Returns:
        List of dicts with keys ['stock_a', 'stock_b', 'weight'].
    """
    if graph is None or graph.number_of_edges() == 0:
        return []

    edges_data = []
    for u, v, data in graph.edges(data=True):
        edges_data.append({
            "stock_a": u,
            "stock_b": v,
            "weight": round(float(data.get("weight", 0.0)), 4),
        })

    edges_sorted = sorted(edges_data, key=lambda x: x["weight"], reverse=True)
    return edges_sorted[:top_n]


# Legacy compatibility functions
def compute_graph_metrics(G: nx.Graph) -> Dict[str, Any]:
    """Legacy helper compatibility."""
    return calculate_graph_statistics(G)
