"""
Unit Tests for Set Theory & Boolean Logic Screening Module
"""

import pandas as pd
import pytest
from modules.set_theory import (
    define_high_return_set,
    define_low_risk_set,
    define_high_volume_set,
    set_intersection,
    set_union,
    set_difference,
    evaluate_screening_rule,
)


@pytest.fixture
def sample_metrics_df():
    """Create sample metrics dataframe for testing."""
    return pd.DataFrame([
        {"Stock": "STOCK_A", "Return %": 20.0, "Risk %": 10.0, "Average Volume": 1000.0},
        {"Stock": "STOCK_B", "Return %": 15.0, "Risk %": 15.0, "Average Volume": 2000.0},
        {"Stock": "STOCK_C", "Return %": 10.0, "Risk %": 25.0, "Average Volume": 3000.0}, # Avg Vol = 2000
    ])


def test_set_definitions_boundary_rules(sample_metrics_df):
    """
    Test set definitions enforce strict inequalities:
    Return > threshold, Risk < threshold, Volume > avg_volume.
    """
    # High Return > 15.0 -> STOCK_A (20.0 > 15.0). STOCK_B is 15.0 (strict > excluded).
    high_ret = define_high_return_set(sample_metrics_df, return_threshold=15.0)
    assert high_ret == {"STOCK_A"}

    # Low Risk < 15.0 -> STOCK_A (10.0 < 15.0). STOCK_B is 15.0 (strict < excluded).
    low_risk = define_low_risk_set(sample_metrics_df, risk_threshold=15.0)
    assert low_risk == {"STOCK_A"}

    # Average Volume = (1000 + 2000 + 3000) / 3 = 2000.0
    # High Volume > 2000.0 -> STOCK_C (3000.0 > 2000.0)
    high_vol = define_high_volume_set(sample_metrics_df)
    assert high_vol == {"STOCK_C"}


def test_set_operations():
    """Test set_intersection, set_union, and set_difference."""
    set_a = {"TCS.NS", "INFY.NS"}
    set_b = {"INFY.NS", "WIPRO.NS"}

    # Intersection: {INFY.NS}
    assert set_intersection(set_a, set_b) == {"INFY.NS"}

    # Union: {TCS.NS, INFY.NS, WIPRO.NS}
    assert set_union(set_a, set_b) == {"TCS.NS", "INFY.NS", "WIPRO.NS"}

    # Difference A \ B: {TCS.NS}
    assert set_difference(set_a, set_b) == {"TCS.NS"}


def test_evaluate_screening_rule_truth_table(sample_metrics_df):
    """
    Test evaluate_screening_rule computes each Boolean condition independently
    and requires ALL 3 to be TRUE for Final Result = TRUE.
    Verify 2 TRUE + 1 FALSE yields FALSE.
    """
    # Thresholds: return > 12.0, risk < 18.0
    # STOCK_A: Return=20>12 (T), Risk=10<18 (T), Volume=1000>2000 (F) -> 2 TRUE + 1 FALSE -> Final Result = FALSE
    # STOCK_B: Return=15>12 (T), Risk=15<18 (T), Volume=2000>2000 (F) -> Final Result = FALSE
    df_screen = evaluate_screening_rule(sample_metrics_df, return_threshold=12.0, risk_threshold=18.0)

    assert len(df_screen) == 3
    
    row_a = df_screen[df_screen["Stock"] == "STOCK_A"].iloc[0]
    assert bool(row_a["Return Condition"]) is True
    assert bool(row_a["Risk Condition"]) is True
    assert bool(row_a["Volume Condition"]) is False
    assert bool(row_a["Final Result"]) is False  # 2 TRUE + 1 FALSE -> FALSE


def test_evaluate_screening_rule_all_true(sample_metrics_df):
    """Test stock meeting all 3 conditions yields Final Result = TRUE."""
    # Custom metrics where STOCK_A meets all 3 criteria
    df_custom = pd.DataFrame([
        {"Stock": "QUALIFIED_STOCK", "Return %": 25.0, "Risk %": 12.0, "Average Volume": 5000.0},
        {"Stock": "OTHER_STOCK", "Return %": 5.0, "Risk %": 30.0, "Average Volume": 1000.0},
    ])

    df_screen = evaluate_screening_rule(df_custom, return_threshold=15.0, risk_threshold=20.0)
    row_q = df_screen[df_screen["Stock"] == "QUALIFIED_STOCK"].iloc[0]

    assert bool(row_q["Return Condition"]) is True
    assert bool(row_q["Risk Condition"]) is True
    assert bool(row_q["Volume Condition"]) is True
    assert bool(row_q["Final Result"]) is True


def test_dynamic_threshold_parameterization(sample_metrics_df):
    """Test re-running with different thresholds alters the result sets accordingly."""
    # Return threshold = 25.0 -> No stocks meet return threshold
    set_high_25 = define_high_return_set(sample_metrics_df, return_threshold=25.0)
    assert len(set_high_25) == 0

    # Return threshold = 5.0 -> All 3 stocks meet return threshold
    set_high_5 = define_high_return_set(sample_metrics_df, return_threshold=5.0)
    assert len(set_high_5) == 3
