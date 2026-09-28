"""
Unit Tests for Portfolio Combinations & Risk Evaluation Module
"""

import math
import numpy as np
import pandas as pd
import pytest
from modules.portfolio import (
    calculate_combination_count,
    select_candidate_pool,
    generate_candidate_portfolios,
    validate_portfolio,
    evaluate_portfolio,
    rank_portfolios,
)


def test_calculate_combination_count():
    """
    Test calculate_combination_count matches worked example:
    C(20, 3) = 1140 and C(5, 2) = 10.
    """
    assert calculate_combination_count(20, 3) == 1140
    assert calculate_combination_count(5, 2) == 10
    assert calculate_combination_count(10, 10) == 1
    assert calculate_combination_count(3, 5) == 0  # n < k


def test_select_candidate_pool():
    """Test select_candidate_pool reduces eligible stock set to candidate pool size."""
    metrics_df = pd.DataFrame([
        {"Stock": f"STOCK_{i}", "Sharpe Ratio": float(i), "Data Status": "OK"}
        for i in range(20)
    ])

    pool_10 = select_candidate_pool(metrics_df, pool_size=10)
    assert len(pool_10) == 10
    # Highest Sharpe Ratio stock (STOCK_19) must be selected in top-10 pool
    assert "STOCK_19" in pool_10


def test_generate_candidate_portfolios():
    """
    Test generate_candidate_portfolios produces exactly C(pool_size, k) unique combinations
    with no duplicates and no repeated stocks within single portfolio.
    """
    pool = ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D", "STOCK_E"] # 5 stocks
    portfolios = generate_candidate_portfolios(pool, k=3)

    # C(5, 3) = 10 combinations
    assert len(portfolios) == 10
    assert len(set(portfolios)) == 10  # Unique

    # Each portfolio must contain 3 distinct stocks
    for p in portfolios:
        assert len(p) == 3
        assert len(set(p)) == 3


def test_validate_portfolio():
    """Test validate_portfolio rejects malformed portfolios."""
    # Valid portfolio of size 3
    assert validate_portfolio(("STOCK_A", "STOCK_B", "STOCK_C"), k=3) is True

    # Malformed: wrong size (2 instead of 3)
    assert validate_portfolio(("STOCK_A", "STOCK_B"), k=3) is False

    # Malformed: duplicate stocks inside single portfolio
    assert validate_portfolio(("STOCK_A", "STOCK_A", "STOCK_B"), k=3) is False


def test_evaluate_portfolio_expected_return():
    """Test evaluate_portfolio produces correct expected return for hand-computed example."""
    metrics_df = pd.DataFrame([
        {"Stock": "STOCK_A", "Return %": 20.0, "Risk %": 15.0},
        {"Stock": "STOCK_B", "Return %": 10.0, "Risk %": 15.0},
    ])
    corr_matrix = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["STOCK_A", "STOCK_B"], columns=["STOCK_A", "STOCK_B"])

    eval_res = evaluate_portfolio(("STOCK_A", "STOCK_B"), metrics_df, corr_matrix)

    assert eval_res["valid"] is True
    # Equal-weighted return = (20.0 + 10.0) / 2 = 15.0%
    assert np.isclose(eval_res["expected_return"], 15.0)


def test_evaluate_portfolio_risk_reflects_correlation():
    """
    Test evaluate_portfolio risk correctly incorporates correlation matrix:
    Portfolio of highly correlated stocks (+1.0) must show HIGHER risk than one of low/negative correlation.
    """
    metrics_df = pd.DataFrame([
        {"Stock": "STOCK_A", "Return %": 10.0, "Risk %": 20.0},
        {"Stock": "STOCK_B", "Return %": 10.0, "Risk %": 20.0},
    ])

    # Case 1: Perfect positive correlation (+1.0)
    corr_high = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=["STOCK_A", "STOCK_B"], columns=["STOCK_A", "STOCK_B"])
    eval_high = evaluate_portfolio(("STOCK_A", "STOCK_B"), metrics_df, corr_high)

    # Case 2: Zero correlation (0.0)
    corr_zero = pd.DataFrame([[1.0, 0.0], [0.0, 1.0]], index=["STOCK_A", "STOCK_B"], columns=["STOCK_A", "STOCK_B"])
    eval_zero = evaluate_portfolio(("STOCK_A", "STOCK_B"), metrics_df, corr_zero)

    # Case 3: Perfect negative correlation (-1.0)
    corr_neg = pd.DataFrame([[1.0, -1.0], [-1.0, 1.0]], index=["STOCK_A", "STOCK_B"], columns=["STOCK_A", "STOCK_B"])
    eval_neg = evaluate_portfolio(("STOCK_A", "STOCK_B"), metrics_df, corr_neg)

    # High correlation (+1.0) -> risk = 20.0%
    assert np.isclose(eval_high["portfolio_risk"], 20.0)

    # Zero correlation (0.0) -> risk = sqrt(0.25*(400 + 400)) = sqrt(200) = 14.14%
    assert np.isclose(eval_zero["portfolio_risk"], 14.14, atol=0.01)

    # Negative correlation (-1.0) -> risk = sqrt(0.25*(400 - 400 - 400 + 400)) = 0.0%
    assert np.isclose(eval_neg["portfolio_risk"], 0.0)

    # High correlation risk MUST be greater than zero correlation risk
    assert eval_high["portfolio_risk"] > eval_zero["portfolio_risk"]
    assert eval_zero["portfolio_risk"] > eval_neg["portfolio_risk"]
