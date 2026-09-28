"""
Unit Tests for Hasse Diagram & Partial Order Verification Module
"""

import networkx as nx
import pytest
from modules.hasse import (
    check_reflexive,
    check_antisymmetric,
    check_transitive,
    is_partial_order,
    build_cover_relation,
    build_hasse_graph,
    find_maximal_minimal_elements,
)


def test_check_reflexive():
    """Test check_reflexive returns True over stock set."""
    stocks = ["STOCK_A", "STOCK_B", "STOCK_C"]
    relation = [("STOCK_A", "STOCK_B")]
    assert check_reflexive(relation, stocks) is True


def test_check_antisymmetric_tie_policy():
    """Test check_antisymmetric detects ties and applies Option A equivalence class policy."""
    stocks = ["STOCK_A", "STOCK_B", "STOCK_C"]
    # STOCK_A ≽ STOCK_B and STOCK_B ≽ STOCK_A (identical return/risk tie)
    relation_with_tie = [("STOCK_A", "STOCK_B"), ("STOCK_B", "STOCK_A")]

    res = check_antisymmetric(relation_with_tie, stocks)

    assert res["is_antisymmetric"] is False
    assert res["is_antisymmetric_up_to_equivalence"] is True
    assert len(res["tied_groups"]) == 1
    assert res["tied_groups"][0] == ["STOCK_A", "STOCK_B"]


def test_check_transitive_valid_and_failing():
    """Test check_transitive verifies valid chain and fails on non-transitive relation."""
    # Valid transitive chain: A ≽ B, B ≽ C, A ≽ C
    valid_transitive = [("STOCK_A", "STOCK_B"), ("STOCK_B", "STOCK_C"), ("STOCK_A", "STOCK_C")]
    assert check_transitive(valid_transitive) is True

    # Deliberately non-transitive relation: A ≽ B and B ≽ C exist, but A ≽ C is missing
    failing_transitive = [("STOCK_A", "STOCK_B"), ("STOCK_B", "STOCK_C")]
    assert check_transitive(failing_transitive) is False


def test_is_partial_order():
    """Test is_partial_order returns correct combined result."""
    stocks = ["STOCK_A", "STOCK_B", "STOCK_C"]
    valid_relation = [("STOCK_A", "STOCK_B"), ("STOCK_B", "STOCK_C"), ("STOCK_A", "STOCK_C")]

    assert is_partial_order(valid_relation, stocks) is True


def test_build_cover_relation_transitive_reduction():
    """
    Test build_cover_relation excludes redundant transitive edge:
    Given A ≽ B, B ≽ C, A ≽ C, cover relation must include A->B and B->C, but NOT A->C.
    """
    full_relation = [("STOCK_A", "STOCK_B"), ("STOCK_B", "STOCK_C"), ("STOCK_A", "STOCK_C")]
    cover = build_cover_relation(full_relation)

    assert len(cover) == 2
    assert ("STOCK_A", "STOCK_B") in cover
    assert ("STOCK_B", "STOCK_C") in cover
    # Redundant transitive edge A->C must be removed by transitive reduction
    assert ("STOCK_A", "STOCK_C") not in cover


def test_hasse_graph_no_self_loops():
    """Test build_hasse_graph contains no self-loops."""
    cover_relation = [("STOCK_A", "STOCK_B"), ("STOCK_B", "STOCK_C")]
    stocks = ["STOCK_A", "STOCK_B", "STOCK_C"]

    hasse_dag = build_hasse_graph(cover_relation, stocks=stocks)

    assert isinstance(hasse_dag, nx.DiGraph)
    assert hasse_dag.number_of_nodes() == 3
    assert hasse_dag.number_of_edges() == 2

    # Verify no self loops
    for node in hasse_dag.nodes():
        assert not hasse_dag.has_edge(node, node)


def test_find_maximal_minimal_elements():
    """Test find_maximal_minimal_elements identifies top undominated and bottom dominated stocks."""
    cover_relation = [("STOCK_A", "STOCK_B"), ("STOCK_B", "STOCK_C")]
    stocks = ["STOCK_A", "STOCK_B", "STOCK_C"]

    hasse_dag = build_hasse_graph(cover_relation, stocks=stocks)
    maximal, minimal = find_maximal_minimal_elements(hasse_dag)

    assert maximal == ["STOCK_A"]
    assert minimal == ["STOCK_C"]
