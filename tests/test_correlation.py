"""
Unit Tests for Correlation Matrix Engine and Matrix Property Verifications
"""

import numpy as np
import pandas as pd
import pytest
from modules.correlation import (
    calculate_return_matrix,
    calculate_correlation_matrix,
    validate_symmetry,
    validate_matrix_properties,
    get_correlation_pairs,
    calculate_correlation_summary_stats,
)


def test_calculate_return_matrix():
    """Test return matrix calculation builds aligned daily returns matrix."""
    idx = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    df1 = pd.DataFrame({"Price": [100.0, 105.0, 110.0]}, index=idx)
    df2 = pd.DataFrame({"Price": [200.0, 190.0, 195.0]}, index=idx)

    ret_matrix = calculate_return_matrix({"STOCK_A.NS": df1, "STOCK_B.NS": df2})

    assert not ret_matrix.empty
    assert "STOCK_A.NS" in ret_matrix.columns
    assert "STOCK_B.NS" in ret_matrix.columns
    assert len(ret_matrix) == 2  # 2 return days


def test_correlation_matrix_properties():
    """
    Test formal matrix properties per Section 6:
    1. Diagonal values equal 1 (A[i,i] = 1)
    2. Matrix is symmetric (A == A^T)
    3. All values lie within [-1, +1]
    """
    returns_df = pd.DataFrame({
        "STOCK_A": [0.01, 0.02, -0.01, 0.03, -0.02],
        "STOCK_B": [0.01, 0.02, -0.01, 0.03, -0.02],  # Perfect positive correlation
        "STOCK_C": [-0.01, -0.02, 0.01, -0.03, 0.02], # Perfect inverse correlation
        "STOCK_D": [0.005, -0.01, 0.015, -0.005, 0.01]
    })

    corr_df = calculate_correlation_matrix(returns_df)

    assert corr_df.shape == (4, 4)

    # 1. Diagonal values equal 1
    for stock in corr_df.columns:
        assert np.isclose(corr_df.loc[stock, stock], 1.0)

    # 2. Matrix is symmetric (A == A^T)
    assert validate_symmetry(corr_df) is True

    # 3. Bounded range [-1, +1]
    assert (corr_df.values >= -1.0 - 1e-6).all()
    assert (corr_df.values <= 1.0 + 1e-6).all()

    # Full properties verification
    props = validate_matrix_properties(corr_df)
    assert props["identity"] is True
    assert props["symmetry"] is True
    assert props["bounded"] is True
    assert props["valid"] is True


def test_get_correlation_pairs():
    """Test get_correlation_pairs extracts unique off-diagonal upper-triangle pairs."""
    corr_df = pd.DataFrame({
        "STOCK_A": [1.0, 0.85, 0.2],
        "STOCK_B": [0.85, 1.0, -0.4],
        "STOCK_C": [0.2, -0.4, 1.0]
    }, index=["STOCK_A", "STOCK_B", "STOCK_C"])

    pairs_df = get_correlation_pairs(corr_df)

    # N=3 stocks -> (3 * 2) / 2 = 3 unique pairs
    assert len(pairs_df) == 3
    assert set(pairs_df.columns) == {"Stock A", "Stock B", "Correlation"}
    
    # Highest pair should be STOCK_A and STOCK_B (0.85)
    assert pairs_df.iloc[0]["Stock A"] == "STOCK_A"
    assert pairs_df.iloc[0]["Stock B"] == "STOCK_B"
    assert np.isclose(pairs_df.iloc[0]["Correlation"], 0.85)


def test_calculate_correlation_summary_stats():
    """Test summary statistics calculation per Section 5."""
    corr_df = pd.DataFrame({
        "STOCK_A": [1.0, 0.8, -0.3],
        "STOCK_B": [0.8, 1.0, 0.1],
        "STOCK_C": [-0.3, 0.1, 1.0]
    }, index=["STOCK_A", "STOCK_B", "STOCK_C"])

    stats = calculate_correlation_summary_stats(corr_df)

    assert stats["num_stocks"] == 3
    assert stats["matrix_size"] == "3 × 3"
    assert stats["highest_pair"]["Correlation"] == 0.8
    assert stats["lowest_pair"]["Correlation"] == -0.3
    # Off-diagonal values: 0.8, -0.3, 0.1 -> mean = 0.6 / 3 = 0.2
    assert np.isclose(stats["avg_pairwise_corr"], 0.2)
