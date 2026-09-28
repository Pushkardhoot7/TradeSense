"""Tests for backend/core/coloring.py — Welsh-Powell from scratch"""
import pytest
from backend.core.coloring import welsh_powell, validate_coloring, get_color_groups, verify_independent_sets


@pytest.fixture
def triangle_adj():
    """Triangle graph K3 — needs 3 colors"""
    return {
        "A": [("B", 0.9), ("C", 0.8)],
        "B": [("A", 0.9), ("C", 0.85)],
        "C": [("A", 0.8), ("B", 0.85)],
    }


@pytest.fixture
def path_adj():
    """Path A-B-C-D — needs 2 colors (bipartite)"""
    return {
        "A": [("B", 0.9)],
        "B": [("A", 0.9), ("C", 0.85)],
        "C": [("B", 0.85), ("D", 0.8)],
        "D": [("C", 0.8)],
    }


@pytest.fixture
def empty_adj():
    return {"A": [], "B": [], "C": []}


class TestWelshPowell:
    def test_returns_coloring_and_trace(self, path_adj):
        coloring, trace = welsh_powell(path_adj)
        assert isinstance(coloring, dict)
        assert isinstance(trace, list)
        assert len(trace) > 0

    def test_all_vertices_colored(self, triangle_adj):
        coloring, _ = welsh_powell(triangle_adj)
        assert set(coloring.keys()) == {"A", "B", "C"}

    def test_colors_are_integers(self, triangle_adj):
        coloring, _ = welsh_powell(triangle_adj)
        assert all(isinstance(c, int) for c in coloring.values())

    def test_triangle_needs_at_least_3_colors(self, triangle_adj):
        coloring, _ = welsh_powell(triangle_adj)
        assert len(set(coloring.values())) >= 3

    def test_path_needs_at_most_2_colors(self, path_adj):
        coloring, _ = welsh_powell(path_adj)
        assert len(set(coloring.values())) <= 2

    def test_isolated_nodes_get_same_color(self, empty_adj):
        coloring, _ = welsh_powell(empty_adj)
        # All isolated — should all get color 0
        assert len(set(coloring.values())) == 1


class TestValidateColoring:
    def test_valid_coloring(self, path_adj):
        coloring, _ = welsh_powell(path_adj)
        is_valid = validate_coloring(path_adj, coloring)
        assert is_valid is True or (isinstance(is_valid, tuple) and is_valid[0] is True)

    def test_invalid_coloring(self, path_adj):
        # Force same color on adjacent nodes
        bad_coloring = {"A": 0, "B": 0, "C": 1, "D": 1}
        result = validate_coloring(path_adj, bad_coloring)
        valid = result if isinstance(result, bool) else result[0]
        assert valid is False


class TestColorGroups:
    def test_groups_are_independent_sets(self, triangle_adj):
        coloring, _ = welsh_powell(triangle_adj)
        groups = get_color_groups(coloring)
        # No two stocks in same group should be adjacent
        for gid, members in groups.items():
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    a, b = members[i], members[j]
                    assert not any(t == b for t, _ in triangle_adj.get(a, []))

    def test_covers_all_vertices(self, path_adj):
        coloring, _ = welsh_powell(path_adj)
        groups = get_color_groups(coloring)
        all_members = [s for g in groups.values() for s in g]
        assert set(all_members) == {"A", "B", "C", "D"}
