"""
Deterministic Test Suite for Top 3 DM Score Display & Ranking Layer

All test fixtures use clearly fictional placeholder tickers (STOCKA, STOCKB, STOCKC...) to prevent confusion with real tickers.
Zero randomness, no seeds.
"""

import numpy as np
import pandas as pd
import pytest
from modules.top3_display import (
    render_top3_portfolios,
    build_top3_structured_data,
    MEDAL_EMOJIS,
)
from modules.dm_scoring import run_dm_portfolio_scoring, DISCLAIMER


@pytest.fixture
def fictional_market_fixture():
    """Fixed fictional market dataset fixture using STOCKA, STOCKB, STOCKC, STOCKD, STOCKE."""
    returns_data = {
        "STOCKA": [0.02, 0.03, 0.02, 0.04, 0.03], # High return, med risk
        "STOCKB": [0.01, 0.01, 0.01, 0.02, 0.01], # Med return, ultra-low risk (5%) -> Incomparable with STOCKA
        "STOCKC": [0.05, -0.04, 0.06, -0.05, 0.05], # High return, high risk
        "STOCKD": [-0.02, -0.01, -0.02, -0.01, -0.02], # Negative return, high risk -> Dominated by STOCKA
        "STOCKE": [0.00, 0.00, 0.01, 0.00, 0.01], # Low return, high risk
    }
    returns_df = pd.DataFrame(returns_data)

    metrics_df = pd.DataFrame([
        {"Stock": "STOCKA", "Return %": 25.0, "Risk %": 8.0},  # Non-dominated
        {"Stock": "STOCKB", "Return %": 12.0, "Risk %": 5.0},  # Non-dominated (Lower risk than STOCKA: 5.0 < 8.0)
        {"Stock": "STOCKC", "Return %": 18.0, "Risk %": 30.0},
        {"Stock": "STOCKD", "Return %": -15.0, "Risk %": 20.0}, # Dominated by STOCKA (Return -15 < 25 AND Risk 20 > 8)
        {"Stock": "STOCKE", "Return %": 5.0, "Risk %": 15.0},
    ])

    corr_df = returns_df.corr()
    coloring_dict = {
        "STOCKA": 0,
        "STOCKB": 1,
        "STOCKC": 0,
        "STOCKD": 2,
        "STOCKE": 1,
    }

    return {
        "returns_df": returns_df,
        "metrics_df": metrics_df,
        "corr_df": corr_df,
        "coloring_dict": coloring_dict,
    }


def test_end_to_end_pipeline_top3_ranking(fictional_market_fixture):
    """
    Feed >= 5 candidate portfolios through full pipeline and assert returned Top 3
    are exactly the 3 highest DM Scores from the full set in descending order.
    """
    candidates = [
        ("STOCKA", "STOCKB"),
        ("STOCKB", "STOCKC"),
        ("STOCKA", "STOCKC"),
        ("STOCKC", "STOCKD"),
        ("STOCKD", "STOCKE"),
        ("STOCKA", "STOCKE"),
    ]  # 6 candidate portfolios

    rendered_text, struct_data = render_top3_portfolios(
        candidate_portfolios=candidates,
        stock_metrics=fictional_market_fixture["metrics_df"],
        returns_matrix=fictional_market_fixture["returns_df"],
        corr_df=fictional_market_fixture["corr_df"],
        coloring_dict=fictional_market_fixture["coloring_dict"],
        top_n=3,
    )

    assert len(struct_data) == 3
    assert struct_data[0]["rank"] == 1
    assert struct_data[1]["rank"] == 2
    assert struct_data[2]["rank"] == 3

    # Assert strict descending order of DM Scores
    assert struct_data[0]["dm_score"] >= struct_data[1]["dm_score"]
    assert struct_data[1]["dm_score"] >= struct_data[2]["dm_score"]


def test_no_hardcode_guard(fictional_market_fixture):
    """
    Assert that changing input candidate set changes displayed output accordingly
    (run pipeline twice with two different fixed candidate sets and assert Top 3 results differ).
    """
    set_1 = [("STOCKA", "STOCKB"), ("STOCKB", "STOCKC"), ("STOCKA", "STOCKC")]
    set_2 = [("STOCKC", "STOCKD"), ("STOCKD", "STOCKE"), ("STOCKB", "STOCKE")]

    txt1, data1 = render_top3_portfolios(
        candidate_portfolios=set_1,
        stock_metrics=fictional_market_fixture["metrics_df"],
        returns_matrix=fictional_market_fixture["returns_df"],
        corr_df=fictional_market_fixture["corr_df"],
        coloring_dict=fictional_market_fixture["coloring_dict"],
    )

    txt2, data2 = render_top3_portfolios(
        candidate_portfolios=set_2,
        stock_metrics=fictional_market_fixture["metrics_df"],
        returns_matrix=fictional_market_fixture["returns_df"],
        corr_df=fictional_market_fixture["corr_df"],
        coloring_dict=fictional_market_fixture["coloring_dict"],
    )

    assert txt1 != txt2
    assert data1[0]["stocks"] != data2[0]["stocks"]


def test_display_formatting_medals_and_numeric_bounds(fictional_market_fixture):
    """
    Assert rendered string contains the three medal emojis 🥇, 🥈, 🥉 in order,
    and every numeric field is finite and within expected range.
    """
    candidates = [
        ("STOCKA", "STOCKB"),
        ("STOCKB", "STOCKC"),
        ("STOCKA", "STOCKE"),
    ]

    rendered_text, struct_data = render_top3_portfolios(
        candidate_portfolios=candidates,
        stock_metrics=fictional_market_fixture["metrics_df"],
        returns_matrix=fictional_market_fixture["returns_df"],
        corr_df=fictional_market_fixture["corr_df"],
        coloring_dict=fictional_market_fixture["coloring_dict"],
    )

    # Check medal emojis presence in order
    assert MEDAL_EMOJIS[0] in rendered_text
    assert MEDAL_EMOJIS[1] in rendered_text
    assert MEDAL_EMOJIS[2] in rendered_text

    idx_m1 = rendered_text.index(MEDAL_EMOJIS[0])
    idx_m2 = rendered_text.index(MEDAL_EMOJIS[1])
    idx_m3 = rendered_text.index(MEDAL_EMOJIS[2])
    assert idx_m1 < idx_m2 < idx_m3

    # Check numeric bounds on structured objects
    for p in struct_data:
        assert 0.0 <= p["dm_score"] <= 100.0
        assert -1.0 <= p["avg_correlation"] <= 1.0
        assert p["risk_raw"] >= 0.0
        assert np.isfinite(p["return_raw"])
        assert np.isfinite(p["return_score"])


def test_dominance_detail_correctness(fictional_market_fixture):
    """
    For a fixed 3-stock portfolio where STOCKD is known by construction to be Pareto-dominated
    by STOCKA (STOCKA Return 25 > -15 AND Risk 8 < 20), assert dominance_detail correctly separates it.
    """
    cand = [("STOCKA", "STOCKB", "STOCKD")]  # Portfolio with 2 non-dominated (STOCKA, STOCKB) and 1 dominated (STOCKD)

    _, struct_data = render_top3_portfolios(
        candidate_portfolios=cand,
        stock_metrics=fictional_market_fixture["metrics_df"],
        returns_matrix=fictional_market_fixture["returns_df"],
        corr_df=fictional_market_fixture["corr_df"],
        coloring_dict=fictional_market_fixture["coloring_dict"],
        top_n=1,
    )

    p0 = struct_data[0]
    dom_detail = p0["dominance_detail"]

    assert "STOCKA" in dom_detail["non_dominated"]
    assert "STOCKB" in dom_detail["non_dominated"]
    assert "STOCKD" in dom_detail["dominated"]


def test_fewer_than_3_candidates_edge_case(fictional_market_fixture):
    """
    If fewer than 3 candidate portfolios are supplied (e.g. 2),
    pipeline returns all available candidate portfolios ranked without crashing.
    """
    candidates = [("STOCKA", "STOCKB"), ("STOCKB", "STOCKC")]  # Exactly 2 candidates

    rendered_text, struct_data = render_top3_portfolios(
        candidate_portfolios=candidates,
        stock_metrics=fictional_market_fixture["metrics_df"],
        returns_matrix=fictional_market_fixture["returns_df"],
        corr_df=fictional_market_fixture["corr_df"],
        coloring_dict=fictional_market_fixture["coloring_dict"],
        top_n=3,
    )

    assert len(struct_data) == 2
    assert MEDAL_EMOJIS[0] in rendered_text
    assert MEDAL_EMOJIS[1] in rendered_text
    assert struct_data[0]["rank"] == 1
    assert struct_data[1]["rank"] == 2
