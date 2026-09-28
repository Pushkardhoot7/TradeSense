"""
Unit Tests for Return-Risk Dominance Binary Relation Module
"""

import pandas as pd
import pytest
from modules.relations import (
    dominates,
    build_dominance_relation,
    get_dominated_stocks,
    get_non_dominated_stocks,
    get_incomparable_pairs,
    get_most_dominant_stocks,
)


def test_dominates_worked_example():
    """
    Test dominates() against Section 2 worked example:
    TCS: Return = 12%, Risk = 15%
    INFY: Return = 9%, Risk = 18%
    Return: 12 >= 9 (TRUE), Risk: 15 <= 18 (TRUE) -> TCS ≽ INFY
    """
    tcs = {"Stock": "TCS.NS", "Return %": 12.0, "Risk %": 15.0}
    infy = {"Stock": "INFY.NS", "Return %": 9.0, "Risk %": 18.0}

    assert dominates(tcs, infy) is True
    assert dominates(infy, tcs) is False


def test_dominates_tie_conditions():
    """Test equal return but higher risk must NOT dominate."""
    stock_a = {"Stock": "A", "Return %": 10.0, "Risk %": 20.0}
    stock_b = {"Stock": "B", "Return %": 10.0, "Risk %": 15.0}  # Better risk!

    # Stock A has equal return but higher risk -> A does NOT dominate B
    assert dominates(stock_a, stock_b) is False
    # Stock B has equal return and lower risk -> B dominates A
    assert dominates(stock_b, stock_a) is True


def test_build_dominance_relation():
    """Test build_dominance_relation produces correct ordered pairs for known stock set."""
    metrics_df = pd.DataFrame([
        {"Stock": "TCS.NS", "Return %": 15.0, "Risk %": 10.0},  # Dominates all
        {"Stock": "INFY.NS", "Return %": 10.0, "Risk %": 15.0}, # Dominates Wipro
        {"Stock": "WIPRO.NS", "Return %": 5.0, "Risk %": 20.0}, # Dominated by all
    ])

    relation = build_dominance_relation(metrics_df)

    # TCS dominates INFY, TCS dominates WIPRO, INFY dominates WIPRO
    assert ("TCS.NS", "INFY.NS") in relation
    assert ("TCS.NS", "WIPRO.NS") in relation
    assert ("INFY.NS", "WIPRO.NS") in relation

    # Reverse pairs must not exist
    assert ("INFY.NS", "TCS.NS") not in relation
    assert ("WIPRO.NS", "TCS.NS") not in relation


def test_get_non_dominated_stocks():
    """Test get_non_dominated_stocks identifies stocks with zero incoming dominance edges."""
    relation = [
        ("TCS.NS", "INFY.NS"),
        ("TCS.NS", "WIPRO.NS"),
    ]
    all_stocks = ["TCS.NS", "INFY.NS", "WIPRO.NS"]

    non_dominated = get_non_dominated_stocks(relation, all_stocks)

    # TCS has zero incoming dominance edges -> non-dominated
    assert non_dominated == ["TCS.NS"]


def test_get_incomparable_pairs():
    """Test incomparable pairs detection where neither stock dominates the other."""
    # Stock A: High return (20%), High risk (25%)
    # Stock B: Low return (10%), Low risk (12%)
    # Neither A ≽ B (25% risk is not <= 12%) nor B ≽ A (10% return is not >= 20%) -> Incomparable
    metrics_df = pd.DataFrame([
        {"Stock": "STOCK_A", "Return %": 20.0, "Risk %": 25.0},
        {"Stock": "STOCK_B", "Return %": 10.0, "Risk %": 12.0},
    ])

    relation = build_dominance_relation(metrics_df)
    assert len(relation) == 0

    all_stocks = ["STOCK_A", "STOCK_B"]
    incomparable = get_incomparable_pairs(relation, all_stocks)

    assert len(incomparable) == 1
    assert incomparable[0] == ("STOCK_A", "STOCK_B")


def test_get_dominated_and_most_dominant_stocks():
    """Test dominated stocks and most dominant ranking."""
    relation = [
        ("TCS.NS", "INFY.NS"),
        ("TCS.NS", "WIPRO.NS"),
    ]
    all_stocks = ["TCS.NS", "INFY.NS", "WIPRO.NS"]

    dominated_by_tcs = get_dominated_stocks(relation, "TCS.NS")
    assert dominated_by_tcs == ["INFY.NS", "WIPRO.NS"]

    most_dom = get_most_dominant_stocks(relation, all_stocks)
    assert most_dom[0]["stock"] == "TCS.NS"
    assert most_dom[0]["out_degree"] == 2
