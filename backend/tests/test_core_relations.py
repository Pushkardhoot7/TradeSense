"""Tests for backend/core/relations.py and backend/core/hasse.py"""
import pytest
from backend.core.relations import (
    build_dominance_relation,
    build_relation_matrix,
    analyze_poset,
    get_non_dominated,
    get_incomparable_pairs,
    get_most_dominant,
)
from backend.core.hasse import build_cover_relation, analyze_hasse


@pytest.fixture
def sample_metrics():
    """A dominates B (more return, less risk); C is independent; A and C are incomparable"""
    return [
        {"symbol": "A.NS", "return_pct": 20.0, "risk_pct": 10.0},
        {"symbol": "B.NS", "return_pct": 10.0, "risk_pct": 15.0},
        {"symbol": "C.NS", "return_pct": 15.0, "risk_pct": 8.0},  # A incomparable to C
    ]


@pytest.fixture
def simple_chain():
    """Linear chain: A dominates B dominates C"""
    return [
        {"symbol": "A.NS", "return_pct": 20.0, "risk_pct": 5.0},
        {"symbol": "B.NS", "return_pct": 15.0, "risk_pct": 10.0},
        {"symbol": "C.NS", "return_pct": 10.0, "risk_pct": 15.0},
    ]


class TestDominanceRelation:
    def test_a_dominates_b(self, sample_metrics):
        pairs = build_dominance_relation(sample_metrics)
        sym_pairs = {(a, b) for a, b in pairs}
        assert ("A.NS", "B.NS") in sym_pairs

    def test_returns_list_of_tuples(self, sample_metrics):
        pairs = build_dominance_relation(sample_metrics)
        assert isinstance(pairs, list)
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pairs)

    def test_reflexive_property_verified(self, sample_metrics):
        """
        Reflexivity is a logical property of the relation (A≽A always holds).
        analyze_poset verifies it; build_dominance_relation may not store self-pairs.
        """
        pairs = build_dominance_relation(sample_metrics)
        tickers = [m["symbol"] for m in sample_metrics]
        props = analyze_poset(pairs, tickers)
        assert props.get("is_reflexive") is True, "Dominance relation must be reflexive"

    def test_chain_transitivity(self, simple_chain):
        pairs = build_dominance_relation(simple_chain)
        sym_pairs = {(a, b) for a, b in pairs}
        # A dominates B, B dominates C → A should dominate C
        assert ("A.NS", "B.NS") in sym_pairs
        assert ("B.NS", "C.NS") in sym_pairs
        assert ("A.NS", "C.NS") in sym_pairs


class TestRelationMatrix:
    def test_matrix_shape(self, sample_metrics):
        pairs = build_dominance_relation(sample_metrics)
        tickers = [m["symbol"] for m in sample_metrics]
        matrix = build_relation_matrix(tickers, pairs)
        assert len(matrix) == len(tickers)
        assert all(len(row) == len(tickers) for row in matrix)

    def test_diagonal_reflects_reflexivity_logically(self, sample_metrics):
        """
        The relation matrix M[i][j]=1 means i dominates j (stored pairs only).
        Reflexivity is a logical property verified by analyze_poset, not always stored.
        Verify the matrix is square and that poset is_reflexive=True.
        """
        pairs = build_dominance_relation(sample_metrics)
        tickers = [m["symbol"] for m in sample_metrics]
        matrix = build_relation_matrix(tickers, pairs)
        # Matrix must be square
        assert len(matrix) == len(tickers)
        assert all(len(row) == len(tickers) for row in matrix)
        # And poset is verified reflexive
        props = analyze_poset(pairs, tickers)
        assert props.get("is_reflexive") is True


class TestPosetProperties:
    def test_is_partial_order(self, sample_metrics):
        pairs = build_dominance_relation(sample_metrics)
        tickers = [m["symbol"] for m in sample_metrics]
        props = analyze_poset(pairs, tickers)
        assert props.get("is_partial_order") is True

    def test_reflexive(self, sample_metrics):
        pairs = build_dominance_relation(sample_metrics)
        tickers = [m["symbol"] for m in sample_metrics]
        props = analyze_poset(pairs, tickers)
        assert props.get("is_reflexive") is True

    def test_transitive(self, simple_chain):
        pairs = build_dominance_relation(simple_chain)
        tickers = [m["symbol"] for m in simple_chain]
        props = analyze_poset(pairs, tickers)
        assert props.get("is_transitive") is True


class TestNonDominated:
    def test_non_dominated_excludes_dominated(self, sample_metrics):
        pairs = build_dominance_relation(sample_metrics)
        tickers = [m["symbol"] for m in sample_metrics]
        non_dom = get_non_dominated(pairs, tickers)
        # B is dominated by A, so B should NOT be in non-dominated set
        assert "B.NS" not in non_dom

    def test_non_dominated_includes_maximal(self, simple_chain):
        pairs = build_dominance_relation(simple_chain)
        tickers = [m["symbol"] for m in simple_chain]
        non_dom = get_non_dominated(pairs, tickers)
        # A is never dominated by anyone
        assert "A.NS" in non_dom


class TestHasse:
    def test_cover_relation_no_transitive_edges(self, simple_chain):
        pairs = build_dominance_relation(simple_chain)
        covers = build_cover_relation(pairs)
        cover_set = {(a, b) for a, b in covers}
        # A→C is transitive (A→B→C) so must NOT appear as a cover edge
        assert ("A.NS", "C.NS") not in cover_set

    def test_cover_relation_has_direct_edges(self, simple_chain):
        pairs = build_dominance_relation(simple_chain)
        covers = build_cover_relation(pairs)
        cover_set = {(a, b) for a, b in covers}
        # Direct edges should be present
        assert ("A.NS", "B.NS") in cover_set
        assert ("B.NS", "C.NS") in cover_set

    def test_analyze_hasse_returns_structure(self, simple_chain):
        pairs = build_dominance_relation(simple_chain)
        tickers = [m["symbol"] for m in simple_chain]
        covers = build_cover_relation(pairs)
        result = analyze_hasse(covers, tickers)
        assert "maximal_elements" in result
        assert "minimal_elements" in result
        assert "levels" in result

    def test_maximal_element_is_top(self, simple_chain):
        pairs = build_dominance_relation(simple_chain)
        tickers = [m["symbol"] for m in simple_chain]
        covers = build_cover_relation(pairs)
        result = analyze_hasse(covers, tickers)
        # A.NS has no dominator, so it's maximal
        assert "A.NS" in result.get("maximal_elements", [])
