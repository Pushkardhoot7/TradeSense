"""
Unit Tests for Welsh-Powell Graph Coloring Module
"""

import networkx as nx
import pytest
from modules.coloring import (
    color_graph,
    validate_coloring,
    get_color_groups,
    calculate_exact_chromatic_number_if_feasible,
)


def test_validate_coloring_valid_and_invalid():
    """Test validate_coloring detects valid vs invalid colorings."""
    # Graph A-B
    G = nx.Graph()
    G.add_edge("STOCK_A", "STOCK_B")

    valid_coloring = {"STOCK_A": 0, "STOCK_B": 1}
    assert validate_coloring(G, valid_coloring) is True

    # Deliberately invalid coloring: adjacent vertices assigned same color
    invalid_coloring = {"STOCK_A": 0, "STOCK_B": 0}
    assert validate_coloring(G, invalid_coloring) is False


def test_color_graph_validity():
    """Test color_graph produces valid coloring on a non-trivial graph."""
    # Complete bipartite graph K3,3
    G = nx.complete_bipartite_graph(3, 3)
    G = nx.relabel_nodes(G, {i: f"STOCK_{i}" for i in range(6)})

    coloring = color_graph(G)

    assert len(coloring) == 6
    assert validate_coloring(G, coloring) is True


def test_get_color_groups():
    """Test get_color_groups returns color mapping dictionary."""
    coloring = {"STOCK_A": 0, "STOCK_B": 1, "STOCK_C": 0}
    groups = get_color_groups(coloring)

    assert len(groups) == 2
    assert groups[0] == ["STOCK_A", "STOCK_C"]
    assert groups[1] == ["STOCK_B"]


def test_calculate_exact_chromatic_number_small_graphs():
    """
    Test exact chromatic number calculation against known hand-verifiable graphs:
    1. Triangle graph K3 -> Chromatic Number = 3
    2. Bipartite graph K2,2 -> Chromatic Number = 2
    """
    # Triangle graph K3
    G_triangle = nx.complete_graph(3)
    res_triangle = calculate_exact_chromatic_number_if_feasible(G_triangle, max_vertices=15)
    assert res_triangle["exact"] is True
    assert res_triangle["chromatic_number"] == 3

    # Bipartite graph K2,2
    G_bipartite = nx.complete_bipartite_graph(2, 2)
    res_bipartite = calculate_exact_chromatic_number_if_feasible(G_bipartite, max_vertices=15)
    assert res_bipartite["exact"] is True
    assert res_bipartite["chromatic_number"] == 2


def test_calculate_exact_chromatic_number_cutoff_enforcement():
    """Test calculate_exact_chromatic_number_if_feasible reports exact: False above size cutoff."""
    # Graph with 20 nodes (> 15)
    G_large = nx.path_graph(20)
    res_large = calculate_exact_chromatic_number_if_feasible(G_large, max_vertices=15)

    assert res_large["exact"] is False
    assert "graph too large" in res_large["reason"].lower() or "vertices" in res_large["reason"].lower()
