"""
Tests for backend/core/graph.py
Verifies BFS, DFS, connected components, adjacency, and graph statistics
are all implemented from scratch without NetworkX algorithm calls.
"""

import pytest
from backend.core.graph import (
    build_adjacency_dict,
    bfs,
    dfs,
    connected_components_from_scratch,
    all_degrees,
    graph_density,
    calculate_graph_statistics,
    get_strong_relationships,
)
import pandas as pd
import numpy as np


@pytest.fixture
def simple_adj():
    """A→B, A→C, B→D: simple path graph"""
    return {
        "A": [("B", 0.9), ("C", 0.8)],
        "B": [("A", 0.9), ("D", 0.85)],
        "C": [("A", 0.8)],
        "D": [("B", 0.85)],
    }


@pytest.fixture
def disconnected_adj():
    """Two separate components: {A,B} and {C,D}"""
    return {
        "A": [("B", 0.95)],
        "B": [("A", 0.95)],
        "C": [("D", 0.88)],
        "D": [("C", 0.88)],
    }


@pytest.fixture
def small_corr_df():
    tickers = ["A", "B", "C", "D"]
    data = np.array([
        [1.00, 0.90, 0.80, 0.20],
        [0.90, 1.00, 0.85, 0.15],
        [0.80, 0.85, 1.00, 0.10],
        [0.20, 0.15, 0.10, 1.00],
    ])
    return pd.DataFrame(data, index=tickers, columns=tickers)


class TestBuildAdjacencyDict:
    def test_edges_above_threshold(self, small_corr_df):
        adj = build_adjacency_dict(["A", "B", "C", "D"], small_corr_df, threshold=0.70)
        # A-B (0.90), A-C (0.80), B-C (0.85) should be edges; A-D, B-D, C-D should not
        assert any(t == "B" for t, _ in adj["A"])
        assert any(t == "C" for t, _ in adj["A"])
        assert not any(t == "D" for t, _ in adj["A"])

    def test_symmetric(self, small_corr_df):
        adj = build_adjacency_dict(["A", "B", "C", "D"], small_corr_df, threshold=0.70)
        for u in adj:
            for v, w in adj[u]:
                assert any(x == u for x, _ in adj.get(v, [])), f"Edge ({u},{v}) not mirrored in adj[{v}]"

    def test_no_self_loops(self, small_corr_df):
        adj = build_adjacency_dict(["A", "B", "C", "D"], small_corr_df, threshold=0.70)
        for u, neighbors in adj.items():
            assert not any(v == u for v, _ in neighbors)


class TestBFS:
    def test_bfs_visits_all_reachable(self, simple_adj):
        order, trace = bfs(simple_adj, "A")
        assert set(order) == {"A", "B", "C", "D"}

    def test_bfs_level_order(self, simple_adj):
        order, trace = bfs(simple_adj, "A")
        # A should come before B, C; B/C before D
        assert order[0] == "A"
        assert "D" in order
        assert order.index("D") > order.index("B")

    def test_bfs_returns_trace(self, simple_adj):
        _, trace = bfs(simple_adj, "A")
        assert len(trace) > 0
        assert any("A" in step for step in trace)

    def test_bfs_isolated_node(self):
        adj = {"X": [], "Y": []}
        order, _ = bfs(adj, "X")
        assert order == ["X"]

    def test_bfs_start_not_in_graph(self):
        order, _ = bfs({}, "MISSING")
        assert order == []


class TestDFS:
    def test_dfs_visits_all_reachable(self, simple_adj):
        order, trace = dfs(simple_adj, "A")
        assert set(order) == {"A", "B", "C", "D"}

    def test_dfs_returns_trace(self, simple_adj):
        _, trace = dfs(simple_adj, "A")
        assert len(trace) > 0

    def test_dfs_first_is_start(self, simple_adj):
        order, _ = dfs(simple_adj, "A")
        assert order[0] == "A"


class TestConnectedComponents:
    def test_connected_graph(self, simple_adj):
        comps = connected_components_from_scratch(simple_adj)
        assert len(comps) == 1
        assert set(comps[0]) == {"A", "B", "C", "D"}

    def test_disconnected_graph(self, disconnected_adj):
        comps = connected_components_from_scratch(disconnected_adj)
        assert len(comps) == 2
        sets = [set(c) for c in comps]
        assert {"A", "B"} in sets
        assert {"C", "D"} in sets

    def test_empty_graph(self):
        comps = connected_components_from_scratch({})
        assert comps == []


class TestDegrees:
    def test_degrees(self, simple_adj):
        degs = all_degrees(simple_adj)
        assert degs["A"] == 2
        assert degs["B"] == 2
        assert degs["C"] == 1
        assert degs["D"] == 1

    def test_isolated_node(self):
        adj = {"X": [], "Y": [("X", 0.5)], "X2": []}
        adj2 = {"X": [], "Y": []}
        degs = all_degrees(adj2)
        assert degs["X"] == 0
        assert degs["Y"] == 0


class TestGraphDensity:
    def test_complete_graph(self):
        # K4 has 4 vertices, 6 edges → density = 2*6/(4*3) = 1.0
        assert abs(graph_density(4, 6) - 1.0) < 1e-6

    def test_empty_graph(self):
        assert graph_density(5, 0) == 0.0

    def test_single_node(self):
        assert graph_density(1, 0) == 0.0


class TestGraphStatistics:
    def test_stats_structure(self, simple_adj):
        stats = calculate_graph_statistics(simple_adj, ["A", "B", "C", "D"])
        required_keys = {"num_vertices", "num_edges", "avg_degree", "max_degree", "density", "connected_components", "most_connected_stock"}
        assert required_keys.issubset(stats.keys())

    def test_edge_count(self, simple_adj):
        stats = calculate_graph_statistics(simple_adj, ["A", "B", "C", "D"])
        assert stats["num_edges"] == 3  # A-B, A-C, B-D

    def test_most_connected(self, simple_adj):
        stats = calculate_graph_statistics(simple_adj, ["A", "B", "C", "D"])
        # A and B both have degree 2
        assert stats["most_connected_stock"] in {"A", "B"}
