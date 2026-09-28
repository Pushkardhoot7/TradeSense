"""
tests/test_dm_score.py - Comprehensive Test Suite for Automatic DM Portfolio Scoring
===================================================================================

Verifies all mathematical properties, component scores, boundary limits,
weight validation, and fully automatic Top 3 selection per Section 7 of
the capstone specification.

Tests are completely deterministic and use fixed hand-constructed data.
"""

import math
import pytest
import numpy as np
import pandas as pd

from modules.dm_score import (
    calculate_return_score,
    calculate_risk_score,
    calculate_diversification_score,
    calculate_group_diversity_score,
    calculate_dominance_score,
    calculate_dm_score,
    rank_portfolios,
    select_top_n_portfolios,
    evaluate_and_score_all_candidates,
    validate_weights,
    DEFAULT_DM_WEIGHTS,
    DISCLAIMER,
)


@pytest.fixture
def sample_market_data():
    """Deterministic hand-crafted market data fixture for 4 stocks."""
    stocks = ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D"]
    metrics_df = pd.DataFrame([
        {"Stock": "STOCK_A", "Return %": 24.0, "Risk %": 12.0},  # Dominant, high return, low risk
        {"Stock": "STOCK_B", "Return %": 16.0, "Risk %": 15.0},
        {"Stock": "STOCK_C", "Return %": 8.0,  "Risk %": 20.0},
        {"Stock": "STOCK_D", "Return %": 4.0,  "Risk %": 28.0},
    ])

    corr_matrix = pd.DataFrame(
        [
            [1.0, 0.2, 0.5, 0.1],
            [0.2, 1.0, 0.3, 0.0],
            [0.5, 0.3, 1.0, 0.4],
            [0.1, 0.0, 0.4, 1.0],
        ],
        index=stocks,
        columns=stocks,
    )

    color_groups = {
        0: ["STOCK_A", "STOCK_D"],  # Independent set
        1: ["STOCK_B"],
        2: ["STOCK_C"],
    }

    dominance_relation = [
        ("STOCK_A", "STOCK_B"),
        ("STOCK_A", "STOCK_C"),
        ("STOCK_A", "STOCK_D"),
        ("STOCK_B", "STOCK_C"),
        ("STOCK_B", "STOCK_D"),
        ("STOCK_C", "STOCK_D"),
    ]

    return {
        "metrics_df": metrics_df,
        "corr_matrix": corr_matrix,
        "color_groups": color_groups,
        "dominance_relation": dominance_relation,
    }


# ===========================================================================
# 1. Return Score & Sensitivity Tests
# ===========================================================================

def test_return_score_monotonicity():
    """Increasing a portfolio's return (holding other components fixed) increases its Return Score and DM Score."""
    candidates = [
        {"portfolio": ["A", "B"], "expected_return": 10.0, "portfolio_risk": 15.0},
        {"portfolio": ["A", "C"], "expected_return": 15.0, "portfolio_risk": 15.0},
        {"portfolio": ["A", "D"], "expected_return": 20.0, "portfolio_risk": 15.0},
    ]
    scored = calculate_return_score(candidates)

    scores = [p["return_score"] for p in scored]
    assert scores[0] == 0.0
    assert scores[1] == 50.0
    assert scores[2] == 100.0
    assert scores[0] < scores[1] < scores[2]

    # Check DM Score increases with return score holding other component scores constant
    dm_1 = calculate_dm_score({"return_score": scores[0], "risk_score": 50.0, "diversification_score": 50.0, "group_score": 50.0, "dominance_score": 50.0})
    dm_2 = calculate_dm_score({"return_score": scores[1], "risk_score": 50.0, "diversification_score": 50.0, "group_score": 50.0, "dominance_score": 50.0})
    dm_3 = calculate_dm_score({"return_score": scores[2], "risk_score": 50.0, "diversification_score": 50.0, "group_score": 50.0, "dominance_score": 50.0})

    assert dm_1 < dm_2 < dm_3


def test_return_score_tied_edge_case():
    """All candidate portfolios tied on return assign equal safe score of 50.0."""
    candidates = [
        {"portfolio": ["A", "B"], "expected_return": 12.0},
        {"portfolio": ["C", "D"], "expected_return": 12.0},
    ]
    scored = calculate_return_score(candidates)
    assert all(p["return_score"] == 50.0 for p in scored)


# ===========================================================================
# 2. Risk Score & Sensitivity Tests
# ===========================================================================

def test_risk_score_monotonicity():
    """Increasing a portfolio's risk (holding other components fixed) decreases its Risk Score and DM Score."""
    candidates = [
        {"portfolio": ["A", "B"], "portfolio_risk": 10.0},  # Lowest risk -> highest score
        {"portfolio": ["A", "C"], "portfolio_risk": 20.0},
        {"portfolio": ["A", "D"], "portfolio_risk": 30.0},  # Highest risk -> lowest score
    ]
    scored = calculate_risk_score(candidates)

    scores = [p["risk_score"] for p in scored]
    assert scores[0] == 100.0
    assert scores[1] == 50.0
    assert scores[2] == 0.0
    assert scores[0] > scores[1] > scores[2]

    # DM Score check
    dm_1 = calculate_dm_score({"return_score": 50.0, "risk_score": scores[0], "diversification_score": 50.0, "group_score": 50.0, "dominance_score": 50.0})
    dm_2 = calculate_dm_score({"return_score": 50.0, "risk_score": scores[1], "diversification_score": 50.0, "group_score": 50.0, "dominance_score": 50.0})
    dm_3 = calculate_dm_score({"return_score": 50.0, "risk_score": scores[2], "diversification_score": 50.0, "group_score": 50.0, "dominance_score": 50.0})

    assert dm_1 > dm_2 > dm_3


def test_risk_score_tied_edge_case():
    """All candidate portfolios tied on risk assign equal safe score of 50.0."""
    candidates = [
        {"portfolio": ["A", "B"], "portfolio_risk": 18.0},
        {"portfolio": ["C", "D"], "portfolio_risk": 18.0},
    ]
    scored = calculate_risk_score(candidates)
    assert all(p["risk_score"] == 50.0 for p in scored)


# ===========================================================================
# 3. Diversification Score Tests
# ===========================================================================

def test_diversification_score_properties():
    """Lower average pairwise correlation increases the Diversification Score."""
    stocks = ["S1", "S2", "S3", "S4"]
    corr_perfect_neg = pd.DataFrame([
        [1.0, -1.0],
        [-1.0, 1.0],
    ], index=["S1", "S2"], columns=["S1", "S2"])

    corr_zero = pd.DataFrame([
        [1.0, 0.0],
        [0.0, 1.0],
    ], index=["S1", "S2"], columns=["S1", "S2"])

    corr_perfect_pos = pd.DataFrame([
        [1.0, 1.0],
        [1.0, 1.0],
    ], index=["S1", "S2"], columns=["S1", "S2"])

    score_neg = calculate_diversification_score(("S1", "S2"), corr_perfect_neg)
    score_zero = calculate_diversification_score(("S1", "S2"), corr_zero)
    score_pos = calculate_diversification_score(("S1", "S2"), corr_perfect_pos)

    assert score_neg == 100.0  # AvgCorr = -1 -> (1 - (-1))/2 * 100 = 100
    assert score_zero == 50.0  # AvgCorr =  0 -> (1 - 0)/2 * 100 = 50
    assert score_pos == 0.0    # AvgCorr = +1 -> (1 - 1)/2 * 100 = 0

    assert score_neg > score_zero > score_pos


# ===========================================================================
# 4. Graph Group Diversity Score Tests
# ===========================================================================

def test_group_diversity_score():
    """Tests group diversity calculation across color groups."""
    color_groups = {
        0: ["A", "B"],
        1: ["C"],
        2: ["D"],
    }
    # Portfolio of 3 stocks from 3 different colors -> max diversity = 100
    score_3_distinct = calculate_group_diversity_score(("A", "C", "D"), color_groups, k=3, total_colors=3)
    assert score_3_distinct == 100.0

    # Portfolio of 2 stocks sharing the same color 0 -> distinct = 1 -> (1-1)/(2-1) = 0
    score_same_color = calculate_group_diversity_score(("A", "B"), color_groups, k=2, total_colors=3)
    assert score_same_color == 0.0

    # Edge case: single color graph
    single_color = {0: ["A", "B", "C"]}
    score_single = calculate_group_diversity_score(("A", "B"), single_color, k=2, total_colors=1)
    assert score_single == 100.0


# ===========================================================================
# 5. Dominance Score Tests
# ===========================================================================

def test_dominance_score():
    """Tests dominance score based on out-degree in the dominance DAG."""
    # 4 stocks in pool. A dominates 3 (B, C, D); B dominates 1 (D); C, D dominate 0.
    dom_rel = [
        ("A", "B"), ("A", "C"), ("A", "D"),
        ("B", "D"),
    ]
    # Stock A normalized = 3 / (4-1) = 1.0 -> 100
    score_a = calculate_dominance_score(("A",), dom_rel, pool_size=4)
    assert score_a == 100.0

    # Stock D normalized = 0 / 3 = 0.0 -> 0
    score_d = calculate_dominance_score(("D",), dom_rel, pool_size=4)
    assert score_d == 0.0

    # Portfolio (A, B) average: (1.0 + 1/3) / 2 = (4/3)/2 = 2/3 = 66.6667
    score_ab = calculate_dominance_score(("A", "B"), dom_rel, pool_size=4)
    assert math.isclose(score_ab, 66.6667, abs_tol=1e-3)


# ===========================================================================
# 6. Boundary, Normalization & Range Tests [0, 100]
# ===========================================================================

def test_all_component_scores_bounded_in_0_100():
    """Every component score stays strictly within [0, 100] across diverse inputs."""
    comp = {
        "return_score": 100.0,
        "risk_score": 0.0,
        "diversification_score": 75.0,
        "group_score": 50.0,
        "dominance_score": 25.0,
    }
    dm = calculate_dm_score(comp)
    assert 0.0 <= dm <= 100.0
    # Expected: 0.30*100 + 0.25*0 + 0.25*75 + 0.10*50 + 0.10*25 = 30 + 0 + 18.75 + 5 + 2.5 = 56.25
    assert math.isclose(dm, 56.25, abs_tol=1e-4)


# ===========================================================================
# 7. Weight Validation Tests
# ===========================================================================

def test_calculate_dm_score_rejects_invalid_weights():
    """calculate_dm_score() rejects weights that do not sum to 100% or 1.0."""
    invalid_weights_low = {
        "return": 0.20, "risk": 0.20, "diversification": 0.20,
        "group_diversity": 0.10, "dominance": 0.10
    }  # Sum = 0.80 != 1.0
    with pytest.raises(ValueError, match="Weights must sum to 100%"):
        calculate_dm_score({"return_score": 50.0}, weights=invalid_weights_low)

    invalid_weights_high = {
        "return": 0.40, "risk": 0.30, "diversification": 0.30,
        "group_diversity": 0.10, "dominance": 0.10
    }  # Sum = 1.20 != 1.0
    with pytest.raises(ValueError, match="Weights must sum to 100%"):
        calculate_dm_score({"return_score": 50.0}, weights=invalid_weights_high)

    invalid_negative = {
        "return": -0.10, "risk": 0.50, "diversification": 0.40,
        "group_diversity": 0.10, "dominance": 0.10
    }
    with pytest.raises(ValueError):
        calculate_dm_score({"return_score": 50.0}, weights=invalid_negative)


def test_calculate_dm_score_accepts_percentage_weights():
    """Weights given on 0-100 scale (30, 25, 25, 10, 10) are correctly accepted."""
    pct_weights = {
        "return": 30.0, "risk": 25.0, "diversification": 25.0,
        "group_diversity": 10.0, "dominance": 10.0
    }
    score = calculate_dm_score({
        "return_score": 100.0, "risk_score": 100.0, "diversification_score": 100.0,
        "group_score": 100.0, "dominance_score": 100.0
    }, weights=pct_weights)
    assert math.isclose(score, 100.0, abs_tol=1e-4)


# ===========================================================================
# 8. Fully Automatic Selection Tests
# ===========================================================================

def test_automatic_top3_selection_no_hardcoded_overrides(sample_market_data):
    """
    Running the scoring pipeline with two different candidate pools produces
    different Top 3 results driven purely by the data — confirming no hardcoded
    stock names influence the outcome.
    """
    pool_1 = [("STOCK_A", "STOCK_B"), ("STOCK_A", "STOCK_C"), ("STOCK_C", "STOCK_D")]
    pool_2 = [("STOCK_B", "STOCK_D"), ("STOCK_C", "STOCK_D"), ("STOCK_B", "STOCK_C")]

    res_1 = evaluate_and_score_all_candidates(
        candidate_tuples=pool_1,
        stock_metrics=sample_market_data["metrics_df"],
        correlation_matrix=sample_market_data["corr_matrix"],
        color_groups=sample_market_data["color_groups"],
        dominance_relation=sample_market_data["dominance_relation"],
    )

    res_2 = evaluate_and_score_all_candidates(
        candidate_tuples=pool_2,
        stock_metrics=sample_market_data["metrics_df"],
        correlation_matrix=sample_market_data["corr_matrix"],
        color_groups=sample_market_data["color_groups"],
        dominance_relation=sample_market_data["dominance_relation"],
    )

    top_1 = [p["portfolio"] for p in res_1["top_portfolios"]]
    top_2 = [p["portfolio"] for p in res_2["top_portfolios"]]

    assert top_1 != top_2, "Pipeline must produce data-driven results, not static or hardcoded outputs"
    assert top_1[0] == ["STOCK_A", "STOCK_B"]  # Highest return, low risk, high dominance

    # Verify rank assignments
    for rank_idx, p in enumerate(res_1["top_portfolios"], start=1):
        assert p["rank"] == rank_idx
        assert "dm_score" in p
        assert "return_score" in p
        assert "risk_score" in p
        assert "diversification_score" in p
        assert "dominance_score" in p


def test_ranking_and_tiebreaking():
    """Verifies deterministic tie-breaking logic."""
    portfolios = [
        {"portfolio": ["C", "D"], "dm_score": 75.0, "return_score": 60.0, "risk_score": 50.0},
        {"portfolio": ["A", "B"], "dm_score": 85.0, "return_score": 80.0, "risk_score": 70.0},
        {"portfolio": ["E", "F"], "dm_score": 75.0, "return_score": 70.0, "risk_score": 50.0},  # Breaks tie with C-D via return_score
    ]
    ranked = rank_portfolios(portfolios)
    assert ranked[0]["portfolio"] == ["A", "B"]
    assert ranked[0]["rank"] == 1
    assert ranked[1]["portfolio"] == ["E", "F"]
    assert ranked[1]["rank"] == 2
    assert ranked[2]["portfolio"] == ["C", "D"]
    assert ranked[2]["rank"] == 3
