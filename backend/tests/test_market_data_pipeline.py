"""
test_market_data_pipeline.py
----------------------------
Automated unit tests for market data pipeline, error handling, backup provider,
validation, symbol mapping, cache, and analytical consistency.
All tests use deterministic mocks with NO live internet dependencies.
"""

from __future__ import annotations

import datetime
import time
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from backend.core.validation import validate_and_clean_price_series, align_returns_matrix
from backend.providers.errors import (
    MarketDataErrorCategory,
    classify_yfinance_error,
)
from backend.providers.symbol_registry import symbol_registry, SymbolInfo
from backend.providers.live_provider import LiveDataProvider
from backend.pipeline import AnalysisPipeline
from backend.schemas import AnalyzeRequest


def _make_mock_df(n_days: int = 50, start_price: float = 100.0, seed: int = 42) -> pd.DataFrame:
    """Generate deterministic OHLCV DataFrame for unit tests."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="B")
    returns = rng.normal(0.001, 0.02, size=n_days)
    prices = start_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.02,
            "Low": prices * 0.98,
            "Close": prices,
            "Volume": rng.integers(10000, 100000, size=n_days),
            "Adj Close": prices,
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


# 1. Yahoo HTTP 401
def test_01_yahoo_http_401():
    exc = Exception("HTTP Error 401: Invalid Crumb")
    err = classify_yfinance_error("TCS.NS", exc)
    assert err.category == MarketDataErrorCategory.PROVIDER_AUTH_ERROR
    assert err.status_code == 401
    assert err.is_transient is False


# 2. Invalid ticker
def test_02_invalid_ticker():
    exc = Exception("HTTP Error 404: Quote not found for symbol: NONEXISTENT.NS")
    err = classify_yfinance_error("NONEXISTENT.NS", exc)
    assert err.category == MarketDataErrorCategory.INVALID_SYMBOL
    assert err.status_code == 404
    assert err.is_transient is False


# 3. Provider timeout
def test_03_provider_timeout():
    exc = Exception("HTTPSConnectionPool: Read timed out")
    err = classify_yfinance_error("INFY.NS", exc)
    assert err.category == MarketDataErrorCategory.TIMEOUT_ERROR
    assert err.is_transient is True


# 4. Rate limit
def test_04_rate_limit():
    exc = Exception("HTTP Error 429: Too Many Requests")
    err = classify_yfinance_error("RELIANCE.NS", exc)
    assert err.category == MarketDataErrorCategory.RATE_LIMIT_ERROR
    assert err.status_code == 429
    assert err.is_transient is True


# 5. Backup provider success
def test_05_backup_provider_success():
    provider = LiveDataProvider()
    # Force yfinance failure
    mock_df = _make_mock_df(30)
    with patch("yfinance.download", side_effect=Exception("HTTP Error 401: Invalid Crumb")):
        with patch.object(provider.backup_provider, "get_historical_data", return_value=mock_df):
            df = provider.get_historical_data("TATAMOTORS.NS", "2024-01-01", "2024-03-01")
            assert not df.empty
            assert len(df) == 30
            assert provider.ticker_sources.get("TATAMOTORS.NS") in ("backup_api", "local_historical_cache")


# 6. Both providers failing
def test_06_both_providers_failing():
    provider = LiveDataProvider()
    with patch("yfinance.download", side_effect=Exception("HTTP Error 401: Invalid Crumb")):
        with patch.object(provider.backup_provider, "get_historical_data", return_value=pd.DataFrame()):
            with pytest.raises(ValueError, match="No valid market data returned"):
                provider.get_historical_data("FAILED.NS", "2024-01-01", "2024-03-01")


# 7. Valid cached data
def test_07_valid_cached_data():
    provider = LiveDataProvider()
    provider._cache.clear()
    mock_data = {"TCS.NS": _make_mock_df(30)}
    symbols = ["TCS.NS"]
    start, end = "2024-01-01", "2024-03-01"

    # Pre-populate cache with timestamp = now
    cache_key = (tuple(sorted(symbols)), start, end, "1d")
    provider._cache[cache_key] = (time.time(), mock_data, {"TCS.NS": "yfinance"}, {})

    res = provider.get_bulk_historical_data(symbols, start, end, "1d")
    assert provider.last_cache_hit is True
    assert "TCS.NS" in res


# 8. Expired cached data
def test_08_expired_cached_data():
    provider = LiveDataProvider()
    provider._cache.clear()
    mock_data = {"TCS.NS": _make_mock_df(30)}
    symbols = ["TCS.NS"]
    start, end = "2024-01-01", "2024-03-01"

    # Pre-populate cache with timestamp = 2 hours ago (expired)
    cache_key = (tuple(sorted(symbols)), start, end, "1d")
    provider._cache[cache_key] = (time.time() - 7200, mock_data, {"TCS.NS": "yfinance"}, {})

    # Mock yfinance to return freshly downloaded data
    fresh_df = _make_mock_df(35)
    with patch("yfinance.download", return_value=fresh_df):
        res = provider.get_bulk_historical_data(symbols, start, end, "1d")
        assert provider.last_cache_hit is False
        assert len(res["TCS.NS"]) == 35


# 9. Duplicate dates
def test_09_duplicate_dates():
    df = _make_mock_df(25)
    # Duplicate first date
    df_dup = pd.concat([df.iloc[[0]], df])
    assert df_dup.index.duplicated().any()

    valid, reason, clean = validate_and_clean_price_series("TEST.NS", df_dup)
    assert valid is True
    assert not clean.index.duplicated().any()
    assert len(clean) == 25


# 10. Missing historical observations
def test_10_missing_historical_observations():
    df_short = _make_mock_df(5)  # Less than min 15 observations
    valid, reason, _ = validate_and_clean_price_series("SHORT.NS", df_short, min_observations=15)
    assert valid is False
    assert "Insufficient history" in reason


# 11. Invalid prices
def test_11_invalid_prices():
    df = _make_mock_df(30)
    # Inject negative and zero prices
    df.loc[df.index[5], "Close"] = -50.0
    df.loc[df.index[10], "Close"] = 0.0

    valid, reason, clean = validate_and_clean_price_series("INVALID_P.NS", df)
    assert valid is True
    assert (clean["Close"] > 0).all()
    assert len(clean) == 28  # dropped 2 invalid rows


# 12. Symbol mapping
def test_12_symbol_mapping():
    assert symbol_registry.get_provider_symbol("REC.NS", "yfinance") == "RECLTD.NS"
    assert symbol_registry.get_provider_symbol("TATAMOTORS.NS", "yfinance") == "TMPV.NS"
    # Verify LTIM is NOT mapped to LTTS
    assert symbol_registry.get_provider_symbol("LTIM.NS", "yfinance") == "LTIM.NS"
    # Verify corporate action metadata
    info_tata = symbol_registry.get_symbol_info("TATAMOTORS.NS")
    assert info_tata is not None
    assert info_tata.corporate_action == "DEMERGER_OCT_2025"
    assert "TMPV" in info_tata.corporate_action_notes


# 13. Partial success with missing tickers
def test_13_partial_success_with_missing_tickers():
    pipe = AnalysisPipeline(AnalyzeRequest(data_mode="DEMO"))
    res = pipe.run()
    # In demo mode, all valid tickers are loaded
    assert res.get("pipeline_status") in ("SUCCESS", "PARTIAL_SUCCESS")
    ctx = res.get("analysis_context", {})
    assert ctx.get("requested_count") > 0
    assert ctx.get("validated_count") > 0


# 14. Consistent eligible universe across all stages
def test_14_consistent_eligible_universe():
    pipe = AnalysisPipeline(AnalyzeRequest(data_mode="DEMO"))
    res = pipe.run()
    eligible = res.get("tickers", [])
    assert len(eligible) > 0

    # Verify Stage 5 correlation matrix strictly uses eligible tickers
    corr_dict = res.get("correlation_matrix", {})
    for sym in corr_dict:
        assert sym in eligible

    # Verify Stage 9 coloring strictly covers eligible tickers
    coloring = res.get("coloring_dict", {})
    for sym in coloring:
        assert sym in eligible

    # Verify Stage 10 dominance stocks strictly come from eligible tickers
    non_dom = res.get("non_dominated_stocks", [])
    for sym in non_dom:
        assert sym in eligible


# 15. Correlation matrix symmetry
def test_15_correlation_matrix_symmetry():
    pipe = AnalysisPipeline(AnalyzeRequest(data_mode="DEMO"))
    res = pipe.run()
    corr_df = res.get("correlation_matrix_df")
    assert corr_df is not None and not corr_df.empty

    arr = corr_df.values
    # Check symmetry: R == R.T within numerical precision
    np.testing.assert_allclose(arr, arr.T, atol=1e-5)
    # Check diagonal is 1.0
    np.testing.assert_allclose(np.diag(arr), 1.0, atol=1e-5)


# 16. Network edge consistency
def test_16_network_edge_consistency():
    threshold = 0.5
    pipe = AnalysisPipeline(AnalyzeRequest(data_mode="DEMO", corr_threshold=threshold))
    res = pipe.run()
    corr_df = res.get("correlation_matrix_df")
    assert corr_df is not None

    from backend.core.graph import build_adjacency_dict
    adj = build_adjacency_dict(res["tickers"], corr_df, threshold)

    # For each edge (u, v), check |corr(u, v)| >= threshold
    for u, neighbors in adj.items():
        for v, weight in neighbors:
            actual_corr = abs(float(corr_df.loc[u, v]))
            assert actual_corr >= threshold - 1e-5


# 17. Portfolio and backtest data consistency
def test_17_portfolio_and_backtest_consistency():
    req = AnalyzeRequest(data_mode="DEMO", portfolio_k=3)
    pipe = AnalysisPipeline(req)
    res = pipe.run()
    eligible_set = set(res["tickers"])
    top_portfolios = res.get("top_portfolios", [])

    for port in top_portfolios:
        constituents = port.get("portfolio", [])
        assert len(constituents) == 3
        for sym in constituents:
            assert sym in eligible_set
        # Validate score in valid range 0 to 100
        dm_score = port.get("dm_score", 0)
        assert 0.0 <= dm_score <= 100.0


# 18. Accurate API error responses and context
def test_18_accurate_api_error_responses():
    from fastapi.testclient import TestClient
    from backend.main import app
    import backend.routers.analysis as analysis_router

    # Seed mock result to avoid live network query during unit tests
    analysis_router._last_result = {
        "analysis_context": {
            "analysis_id": "test-mock-id",
            "pipeline_status": "PARTIAL_SUCCESS",
            "requested_universe": ["TCS.NS", "INFY.NS", "LTIM.NS"],
            "eligible_universe": ["TCS.NS", "INFY.NS"],
            "excluded_tickers": ["LTIM.NS"],
            "exclusion_reasons": {"LTIM.NS": "[PROVIDER_AUTH_ERROR] Yahoo Finance HTTP 401 Unauthorized"},
            "requested_count": 3,
            "validated_count": 2,
            "excluded_count": 1,
            "data_provider": "● LIVE • NSE (OPEN)",
            "data_freshness": "15:30:00",
        },
        "tickers": ["TCS.NS", "INFY.NS"],
        "stocks_analyzed": 2,
        "provider_label": "● LIVE • NSE (OPEN)",
        "last_updated": "15:30:00",
        "data_quality": {"score": 95.0, "status": "GOOD"},
    }

    client = TestClient(app)
    # Test /api/context
    ctx_resp = client.get("/api/context")
    assert ctx_resp.status_code == 200
    ctx_data = ctx_resp.json()
    assert ctx_data["pipeline_status"] == "PARTIAL_SUCCESS"
    assert "requested_universe" in ctx_data
    assert "eligible_universe" in ctx_data
    assert ctx_data["excluded_count"] == 1

    # Test /api/data-quality
    dq_resp = client.get("/api/data-quality")
    assert dq_resp.status_code == 200
    dq_data = dq_resp.json()
    assert dq_data["requested_stocks"] == 3
    assert dq_data["valid_stocks"] == 2
    assert dq_data["excluded_stocks"] == 1
    assert "LTIM.NS" in dq_data["exclusion_reasons"]
