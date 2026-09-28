"""
Deterministic Test Suite for Automatic DM Portfolio Scoring System

Tests use fixed, hand-constructed sample arrays (zero randomness, no seeds).
"""

import math
import numpy as np
import pandas as pd
import pytest
from modules.dm_scoring import (
    run_dm_portfolio_scoring,
    compute_covariance_portfolio_volatility,
    validate_weights_config,
    DM_SCORE_WEIGHTS,
    DISCLAIMER,
)


@pytest.fixture
def fixed_market_fixture():
    """
    Fixed, hand-constructed market dataset fixture.
    """
    # 4 stocks: STOCK_A, STOCK_B, STOCK_C, STOCK_D
    # 10 trading days of returns
    returns_data = {
        "STOCK_A": [0.01, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.01, 0.02, 0.03], # High return, low risk
        "STOCK_B": [0.00, -0.01, 0.01, 0.00, 0.01, 0.00, -0.01, 0.01, 0.00, 0.01], # Med return, low risk
        "STOCK_C": [0.05, -0.04, 0.06, -0.05, 0.05, -0.04, 0.06, -0.05, 0.05, -0.04], # Volatile
        "STOCK_D": [-0.01, -0.02, -0.01, -0.03, -0.02, -0.01, -0.02, -0.01, -0.02, -0.03], # Negative return
    }
    returns_df = pd.DataFrame(returns_data)

    metrics_df = pd.DataFrame([
        {"Stock": "STOCK_A", "Return %": 20.0, "Risk %": 10.0},
        {"Stock": "STOCK_B", "Return %": 10.0, "Risk %": 10.0},
        {"Stock": "STOCK_C", "Return %": 15.0, "Risk %": 35.0},
        {"Stock": "STOCK_D", "Return %": -10.0, "Risk %": 12.0},
    ])

    corr_df = returns_df.corr()
    coloring_dict = {
        "STOCK_A": 0,
        "STOCK_B": 1,
        "STOCK_C": 0,
        "STOCK_D": 2,
    }

    return {
        "returns_df": returns_df,
        "metrics_df": metrics_df,
        "corr_df": corr_df,
        "coloring_dict": coloring_dict,
    }


def test_weight_sum_sanity_check():
    """Assert configured weights sum to 1.0 (100%)."""
    assert validate_weights_config(DM_SCORE_WEIGHTS) is True

    invalid_weights = dict(DM_SCORE_WEIGHTS)
    invalid_weights["return"] = 0.50  # Total = 1.20
    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_weights_config(invalid_weights)


def test_monotonicity_return(fixed_market_fixture):
    """
    Given two candidate portfolios identical except Portfolio A has strictly higher return than Portfolio B,
    assert ReturnScore(A) > ReturnScore(B) and DM_Score(A) > DM_Score(B).
    """
    cand_a = ("STOCK_A", "STOCK_B")  # Return = (20 + 10) / 2 = 15%
    cand_b = ("STOCK_B", "STOCK_D")  # Return = (10 - 10) / 2 = 0%

    res = run_dm_portfolio_scoring(
        candidate_portfolios=[cand_a, cand_b],
        stock_metrics=fixed_market_fixture["metrics_df"],
        returns_matrix=fixed_market_fixture["returns_df"],
        corr_df=fixed_market_fixture["corr_df"],
        coloring_dict=fixed_market_fixture["coloring_dict"],
    )

    all_evals = {item["portfolio"]: item for item in res["all_evaluated"]}
    eval_a = all_evals[cand_a]
    eval_b = all_evals[cand_b]

    assert eval_a["component_scores"]["return_score"] > eval_b["component_scores"]["return_score"]
    assert eval_a["dm_score"] > eval_b["dm_score"]


def test_monotonicity_risk(fixed_market_fixture):
    """
    Given two candidate portfolios identical except Portfolio A has strictly lower volatility than Portfolio B,
    assert RiskScore(A) > RiskScore(B).
    """
    cand_low_risk = ("STOCK_A", "STOCK_B")  # Low risk constituents
    cand_high_risk = ("STOCK_C", "STOCK_D") # STOCK_C has 35% volatility

    res = run_dm_portfolio_scoring(
        candidate_portfolios=[cand_low_risk, cand_high_risk],
        stock_metrics=fixed_market_fixture["metrics_df"],
        returns_matrix=fixed_market_fixture["returns_df"],
        corr_df=fixed_market_fixture["corr_df"],
        coloring_dict=fixed_market_fixture["coloring_dict"],
    )

    all_evals = {item["portfolio"]: item for item in res["all_evaluated"]}
    eval_low = all_evals[cand_low_risk]
    eval_high = all_evals[cand_high_risk]

    # Lower raw risk -> higher RiskScore
    assert eval_low["raw_metrics"]["raw_risk"] < eval_high["raw_metrics"]["raw_risk"]
    assert eval_low["component_scores"]["risk_score"] > eval_high["component_scores"]["risk_score"]


def test_monotonicity_diversification(fixed_market_fixture):
    """
    Given two candidate portfolios identical except Portfolio A has strictly lower average pairwise correlation,
    assert DiversificationScore(A) > DiversificationScore(B).
    """
    # Create fixed correlation matrix where pair (A, B) has low correlation, pair (A, C) has high correlation
    custom_corr = pd.DataFrame([
        [1.0, 0.1, 0.9],
        [0.1, 1.0, 0.5],
        [0.9, 0.5, 1.0]
    ], index=["STOCK_A", "STOCK_B", "STOCK_C"], columns=["STOCK_A", "STOCK_B", "STOCK_C"])

    metrics_df = pd.DataFrame([
        {"Stock": "STOCK_A", "Return %": 10.0, "Risk %": 10.0},
        {"Stock": "STOCK_B", "Return %": 10.0, "Risk %": 10.0},
        {"Stock": "STOCK_C", "Return %": 10.0, "Risk %": 10.0},
    ])

    cand_low_corr = ("STOCK_A", "STOCK_B") # corr = 0.1
    cand_high_corr = ("STOCK_A", "STOCK_C") # corr = 0.9

    res = run_dm_portfolio_scoring(
        candidate_portfolios=[cand_low_corr, cand_high_corr],
        stock_metrics=metrics_df,
        returns_matrix=pd.DataFrame(),
        corr_df=custom_corr,
        coloring_dict={"STOCK_A": 0, "STOCK_B": 1, "STOCK_C": 0},
    )

    all_evals = {item["portfolio"]: item for item in res["all_evaluated"]}
    eval_low = all_evals[cand_low_corr]
    eval_high = all_evals[cand_high_corr]

    assert eval_low["raw_metrics"]["raw_avg_corr"] < eval_high["raw_metrics"]["raw_avg_corr"]
    assert eval_low["component_scores"]["diversification_score"] > eval_high["component_scores"]["diversification_score"]


def test_range_check_and_disclaimer(fixed_market_fixture):
    """
    Assert 0 <= score <= 100 for all component scores across all evaluated candidates,
    and verify disclaimer string presence.
    """
    candidates = [("STOCK_A", "STOCK_B"), ("STOCK_B", "STOCK_C"), ("STOCK_A", "STOCK_D")]

    res = run_dm_portfolio_scoring(
        candidate_portfolios=candidates,
        stock_metrics=fixed_market_fixture["metrics_df"],
        returns_matrix=fixed_market_fixture["returns_df"],
        corr_df=fixed_market_fixture["corr_df"],
        coloring_dict=fixed_market_fixture["coloring_dict"],
    )

    assert res["disclaimer"] == DISCLAIMER

    for item in res["all_evaluated"]:
        comps = item["component_scores"]
        for comp_name, comp_val in comps.items():
            assert 0.0 <= comp_val <= 100.0, f"Component {comp_name} out of bounds: {comp_val}"
        assert 0.0 <= item["dm_score"] <= 100.0


def test_no_manual_selection_guard(fixed_market_fixture):
    """
    Assert calling the Top 3 selection function twice with same inputs produces identical, order-stable result,
    and passing manual_rank_override raises ValueError.
    """
    candidates = [("STOCK_A", "STOCK_B"), ("STOCK_B", "STOCK_C"), ("STOCK_A", "STOCK_D")]

    res1 = run_dm_portfolio_scoring(
        candidate_portfolios=candidates,
        stock_metrics=fixed_market_fixture["metrics_df"],
        returns_matrix=fixed_market_fixture["returns_df"],
        corr_df=fixed_market_fixture["corr_df"],
        coloring_dict=fixed_market_fixture["coloring_dict"],
    )

    res2 = run_dm_portfolio_scoring(
        candidate_portfolios=candidates,
        stock_metrics=fixed_market_fixture["metrics_df"],
        returns_matrix=fixed_market_fixture["returns_df"],
        corr_df=fixed_market_fixture["corr_df"],
        coloring_dict=fixed_market_fixture["coloring_dict"],
    )

    # Order-stable pure function guarantee
    assert res1["top_portfolios"] == res2["top_portfolios"]

    # Guard assertion check: Passing manual override MUST fail loudly!
    with pytest.raises(ValueError, match="CRITICAL GUARD"):
        run_dm_portfolio_scoring(
            candidate_portfolios=candidates,
            stock_metrics=fixed_market_fixture["metrics_df"],
            returns_matrix=fixed_market_fixture["returns_df"],
            corr_df=fixed_market_fixture["corr_df"],
            coloring_dict=fixed_market_fixture["coloring_dict"],
            manual_rank_override=[2, 1, 0],
        )


def test_edge_case_identical_risk():
    """
    Test degenerate case where all candidate portfolios have identical volatility.
    Verify RiskScore = 100 for all candidates with no division by zero.
    """
    metrics_df = pd.DataFrame([
        {"Stock": "STOCK_A", "Return %": 10.0, "Risk %": 15.0},
        {"Stock": "STOCK_B", "Return %": 20.0, "Risk %": 15.0},
    ])
    # Empty returns matrix to force identical fallback risk
    res = run_dm_portfolio_scoring(
        candidate_portfolios=[("STOCK_A",), ("STOCK_B",)],
        stock_metrics=metrics_df,
        returns_matrix=pd.DataFrame(),
        corr_df=pd.DataFrame(),
        coloring_dict={"STOCK_A": 0, "STOCK_B": 1},
    )

    for item in res["all_evaluated"]:
        assert item["component_scores"]["risk_score"] == 100.0


def test_edge_case_single_stock_portfolio(fixed_market_fixture):
    """Test Diversification and Dominance scores do not crash or error on 1-stock portfolios."""
    res = run_dm_portfolio_scoring(
        candidate_portfolios=[("STOCK_A",), ("STOCK_B",)],
        stock_metrics=fixed_market_fixture["metrics_df"],
        returns_matrix=fixed_market_fixture["returns_df"],
        corr_df=fixed_market_fixture["corr_df"],
        coloring_dict=fixed_market_fixture["coloring_dict"],
    )

    assert len(res["top_portfolios"]) == 2
    for item in res["all_evaluated"]:
        assert item["raw_metrics"]["raw_avg_corr"] == 1.0  # Single stock correlation treated as 1.0
        assert item["component_scores"]["diversification_score"] == 0.0  # 100 * (1 - 1) / 2 = 0.0
