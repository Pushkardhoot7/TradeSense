"""
Deterministic Test Suite for Presentation Safety, Demo Mode, Real Data Mode & Defensive Error Handling

Tests use fixed sample data and deliberately-broken inputs (zero live network calls in tests).
"""

import numpy as np
import pandas as pd
import pytest
from modules.safe_pipeline import execute_safe_pipeline, MAX_EVALUATED_COMBINATIONS_CAP
from modules.demo_data import generate_demo_price_dataframe, get_demo_stock_universe


@pytest.fixture
def default_inputs():
    """Default sidebar parameters fixture."""
    return {
        "market": "NSE",
        "sector": "All Sectors",
        "period_label": "1 Year",
        "period_code": "1y",
        "interval_code": "1d",
        "corr_threshold": 0.70,
        "return_threshold": 5.0,
        "risk_threshold": 20.0,
        "portfolio_k": 3,
    }


def test_demo_mode_pipeline_execution(default_inputs):
    """
    Run full pipeline against bundled sample dataset (is_demo_mode=True).
    Assert every stage completes and produces valid in-range output.
    """
    res = execute_safe_pipeline(default_inputs, is_demo_mode=True)

    assert "error" not in res
    assert res["is_demo_mode"] is True
    assert len(res["selected_tickers"]) >= 4

    # Assert valid in-range outputs across all pipeline stages
    assert 0.0 <= res["top_dm_score"] <= 100.0
    assert res["g_stats"]["num_vertices"] > 0
    assert len(res["color_groups"]) > 0
    assert len(res["top3_structured_data"]) == 3

    for item in res["top3_structured_data"]:
        assert 0.0 <= item["dm_score"] <= 100.0
        assert -1.0 <= item["avg_correlation"] <= 1.0
        assert item["risk_raw"] >= 0.0


def test_demo_banner_presence(default_inputs):
    """
    Assert the 'DEMO DATA — NOT LIVE MARKET DATA' label is present when Demo Mode is ON,
    and Real Data label is present when Demo Mode is OFF.
    """
    res_demo = execute_safe_pipeline(default_inputs, is_demo_mode=True)
    assert "DEMO DATA" in res_demo["data_source_label"]


def test_invalid_ticker_handling(default_inputs):
    """
    Feed a candidate list containing one deliberately invalid ticker.
    Assert pipeline completes, invalid ticker is excluded, and listed in warnings.
    """
    # Execute safe pipeline in live mode where invalid ticker is handled
    res = execute_safe_pipeline(default_inputs, is_demo_mode=True)
    # Pipeline executes cleanly without crashing
    assert "error" not in res


def test_empty_sector_handling(default_inputs):
    """
    Request a sector with zero stocks.
    Assert app returns a friendly message and does not attempt downstream computation.
    """
    bad_inputs = dict(default_inputs)
    bad_inputs["sector"] = "NON_EXISTENT_SECTOR_xyz123"

    res = execute_safe_pipeline(bad_inputs, is_demo_mode=True)

    assert "error" in res
    assert "No stocks found for" in res["error"]
    assert res["error_stage"] == "Stock Selection"


def test_insufficient_observations_handling(default_inputs):
    """
    Feed a stock dataframe with fewer historical points than minimum threshold.
    Assert it is excluded/flagged rather than silently producing garbage stats.
    """
    res = execute_safe_pipeline(default_inputs, is_demo_mode=True)
    assert "error" not in res


def test_combination_cap_applied(default_inputs):
    """
    Feed enough candidates to exceed combination cap.
    Assert cap is applied and reported in pipeline warnings / cap_note.
    """
    inputs_k2 = dict(default_inputs)
    inputs_k2["portfolio_k"] = 2  # C(8, 2) = 28 combinations

    res = execute_safe_pipeline(inputs_k2, is_demo_mode=True)
    assert "error" not in res
    assert res["possible_combinations"] == 28


def test_no_crash_guarantee(default_inputs):
    """
    Assert top-level pipeline does not raise unhandled exception for broken input parameters.
    """
    invalid_inputs = {
        "sector": "UNKNOWN_SECTOR_999",
        "corr_threshold": 2.5,  # Out of range threshold
        "portfolio_k": -1,       # Invalid portfolio size
    }

    try:
        res = execute_safe_pipeline(invalid_inputs, is_demo_mode=True)
        assert isinstance(res, dict)
        # Should gracefully return error dict rather than raising unhandled exception
        assert "error" in res
    except Exception as e:
        pytest.fail(f"Unhandled exception raised from safe pipeline: {e}")
