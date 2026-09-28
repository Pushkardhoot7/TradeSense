"""
Unit Tests for Correlation Graph Analysis Module
"""

import networkx as nx
import pandas as pd
import pytest
from modules.graph_analysis import (
    build_correlation_graph,
    calculate_graph_statistics,
    get_degree,
    get_connected_components,
    get_strong_relationships,
)


@pytest.fixture
def sample_corr_matrix():
    """Create a sample 4x4 symmetric correlation matrix for testing."""
    return pd.DataFrame({
        "STOCK_A": [1.0, 0.85, 0.40, 0.10],
        "STOCK_B": [0.85, 1.0, 0.75, 0.20],
        "STOCK_C": [0.40, 0.75, 1.0, 0.90],
        "STOCK_D": [0.10, 0.20, 0.90, 1.0],
    }, index=["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D"])


def test_build_correlation_graph_threshold_enforcement(sample_corr_matrix):
    """Test threshold filtering, no self-loops, and no duplicate edges."""
    # Threshold = 0.70: Edges expected for A-B (0.85), B-C (0.75), C-D (0.90)
    G = build_correlation_graph(sample_corr_matrix, threshold=0.70)

    assert G.number_of_nodes() == 4
    assert G.number_of_edges() == 3

    # No self loops (A-A, B-B, C-C, D-D must not exist)
    for node in G.nodes():
        assert not G.has_edge(node, node)

    # Specific edges
    assert G.has_edge("STOCK_A", "STOCK_B")
    assert G.has_edge("STOCK_B", "STOCK_C")
    assert G.has_edge("STOCK_C", "STOCK_D")
    assert not G.has_edge("STOCK_A", "STOCK_C")
    assert not G.has_edge("STOCK_A", "STOCK_D")


def test_configurable_threshold(sample_corr_matrix):
    """Test graph density and edge count vary with threshold."""
    G_high = build_correlation_graph(sample_corr_matrix, threshold=0.88)
    # Only C-D (0.90) meets threshold 0.88
    assert G_high.number_of_edges() == 1

    G_low = build_correlation_graph(sample_corr_matrix, threshold=0.15)
    # All pairs except A-D (0.10) meet threshold 0.15 -> 5 edges
    assert G_low.number_of_edges() == 5


def test_calculate_graph_statistics(sample_corr_matrix):
    """Test graph statistics computation."""
    G = build_correlation_graph(sample_corr_matrix, threshold=0.70)
    stats = calculate_graph_statistics(G)

    assert stats["num_vertices"] == 4
    assert stats["num_edges"] == 3
    # Degrees: A=1, B=2, C=2, D=1 -> sum=6 -> avg = 6/4 = 1.5
    assert stats["avg_degree"] == 1.5
    assert stats["highest_degree_stock"]["degree"] == 2
    # Single connected component containing all 4 stocks
    assert stats["num_connected_components"] == 1
    assert stats["connected_components"][0] == ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D"]


def test_get_degree(sample_corr_matrix):
    """Test get_degree returns vertex degree."""
    G = build_correlation_graph(sample_corr_matrix, threshold=0.70)

    assert get_degree(G, "STOCK_B") == 2
    assert get_degree(G, "STOCK_A") == 1
    assert get_degree(G, "NON_EXISTENT") == 0


def test_get_connected_components(sample_corr_matrix):
    """Test get_connected_components splits graph into component groups."""
    # High threshold 0.80 -> A-B (0.85), C-D (0.90). Component 1: [A, B], Component 2: [C, D]
    G = build_correlation_graph(sample_corr_matrix, threshold=0.80)
    components = get_connected_components(G)

    assert len(components) == 2
    assert ["STOCK_A", "STOCK_B"] in components
    assert ["STOCK_C", "STOCK_D"] in components


def test_get_strong_relationships(sample_corr_matrix):
    """Test get_strong_relationships returns top-N highest weight edges."""
    G = build_correlation_graph(sample_corr_matrix, threshold=0.70)
    top_rel = get_strong_relationships(G, top_n=2)

    assert len(top_rel) == 2
    # Highest edge is C-D (0.90)
    assert top_rel[0]["weight"] == 0.90
    assert {top_rel[0]["stock_a"], top_rel[0]["stock_b"]} == {"STOCK_C", "STOCK_D"}
