"""Tests for backend/core/scoring.py — DM Score engine"""
import pytest
import pandas as pd
import numpy as np
from backend.core.scoring import run_scoring_pipeline, DM_WEIGHTS


@pytest.fixture
def five_stock_data():
    """
    Minimal but complete data for 5 stocks to run the scoring pipeline.
    Returns (candidates, metrics, returns_df, corr_df, coloring, non_dominated)
    """
    tickers = ["A.NS", "B.NS", "C.NS", "D.NS", "E.NS"]
    metrics = [
        {"symbol": "A.NS", "return_pct": 20.0, "risk_pct": 12.0, "avg_volume": 2e6, "sharpe": 1.2},
        {"symbol": "B.NS", "return_pct": 10.0, "risk_pct": 18.0, "avg_volume": 3e6, "sharpe": 0.4},
        {"symbol": "C.NS", "return_pct": 15.0, "risk_pct": 10.0, "avg_volume": 1e6, "sharpe": 1.1},
        {"symbol": "D.NS", "return_pct": -5.0, "risk_pct": 25.0, "avg_volume": 4e6, "sharpe": -0.3},
        {"symbol": "E.NS", "return_pct": 12.0, "risk_pct": 14.0, "avg_volume": 2.5e6, "sharpe": 0.7},
    ]

    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    returns_data = {t: np.random.normal(0.001, 0.015, 252) for t in tickers}
    returns_df = pd.DataFrame(returns_data, index=dates)

    corr_data = np.array([
        [1.00, 0.75, 0.60, 0.20, 0.55],
        [0.75, 1.00, 0.65, 0.15, 0.50],
        [0.60, 0.65, 1.00, 0.10, 0.70],
        [0.20, 0.15, 0.10, 1.00, 0.08],
        [0.55, 0.50, 0.70, 0.08, 1.00],
    ])
    corr_df = pd.DataFrame(corr_data, index=tickers, columns=tickers)

    coloring = {"A.NS": 0, "B.NS": 1, "C.NS": 0, "D.NS": 2, "E.NS": 1}
    non_dominated = {"A.NS", "C.NS"}

    # k=2 candidates
    import itertools
    candidates = list(itertools.combinations(tickers, 2))

    return candidates, metrics, returns_df, corr_df, coloring, non_dominated


class TestScoringPipeline:
    def test_returns_top_portfolios(self, five_stock_data):
        candidates, metrics, returns_df, corr_df, coloring, non_dom = five_stock_data
        out = run_scoring_pipeline(candidates, metrics, returns_df, corr_df, coloring, non_dom, top_n=3)
        assert "top_portfolios" in out
        assert len(out["top_portfolios"]) <= 3

    def test_portfolio_key_always_list(self, five_stock_data):
        """Critical: KeyError: 'portfolio' must never happen"""
        candidates, metrics, returns_df, corr_df, coloring, non_dom = five_stock_data
        out = run_scoring_pipeline(candidates, metrics, returns_df, corr_df, coloring, non_dom, top_n=3)
        for p in out["top_portfolios"]:
            assert "portfolio" in p, "portfolio key missing!"
            assert isinstance(p["portfolio"], list), "portfolio must be list[str]"

    def test_dm_score_bounded(self, five_stock_data):
        candidates, metrics, returns_df, corr_df, coloring, non_dom = five_stock_data
        out = run_scoring_pipeline(candidates, metrics, returns_df, corr_df, coloring, non_dom)
        for p in out["all_evaluated"]:
            assert 0.0 <= p["dm_score"] <= 100.0, f"DM score out of range: {p['dm_score']}"

    def test_sorted_descending(self, five_stock_data):
        candidates, metrics, returns_df, corr_df, coloring, non_dom = five_stock_data
        out = run_scoring_pipeline(candidates, metrics, returns_df, corr_df, coloring, non_dom)
        scores = [p["dm_score"] for p in out["all_evaluated"]]
        assert scores == sorted(scores, reverse=True), "Portfolios not sorted descending by DM score"

    def test_rank_assigned(self, five_stock_data):
        candidates, metrics, returns_df, corr_df, coloring, non_dom = five_stock_data
        out = run_scoring_pipeline(candidates, metrics, returns_df, corr_df, coloring, non_dom)
        ranks = [p.get("rank") for p in out["all_evaluated"]]
        assert ranks[0] == 1
        assert ranks[1] == 2

    def test_weights_sum_to_one(self):
        total = sum(DM_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9, f"Weights do not sum to 1.0: {total}"

    def test_empty_candidates(self):
        out = run_scoring_pipeline([], [], pd.DataFrame(), pd.DataFrame(), {}, set())
        assert out["top_portfolios"] == []
        assert out["total_evaluated"] == 0

    def test_all_components_present(self, five_stock_data):
        candidates, metrics, returns_df, corr_df, coloring, non_dom = five_stock_data
        out = run_scoring_pipeline(candidates, metrics, returns_df, corr_df, coloring, non_dom, top_n=1)
        p = out["top_portfolios"][0]
        for key in ["dm_score", "return_score", "risk_score", "diversification_score", "group_score", "dominance_score"]:
            assert key in p, f"Missing score component: {key}"
