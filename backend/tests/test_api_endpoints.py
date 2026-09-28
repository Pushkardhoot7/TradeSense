"""
Integration tests for all TradeSense V2 FastAPI routes and API endpoints.
Tests every HTML page and API endpoint via TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_html_pages():
    pages = [
        "/",
        "/market-scanner",
        "/stock/TCS.NS",
        "/correlation",
        "/relations-hasse",
        "/graph-coloring",
        "/sets-boolean",
        "/portfolio-optimizer",
        "/reports",
    ]
    for p in pages:
        resp = client.get(p)
        assert resp.status_code == 200, f"Page {p} returned status {resp.status_code}"
        assert len(resp.text) > 100, f"Page {p} returned empty content"


def test_market_api():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

    r = client.get("/api/market/status")
    assert r.status_code == 200
    assert "status" in r.json()

    r = client.get("/api/stocks/sectors")
    assert r.status_code == 200
    assert "sectors" in r.json()

    r = client.get("/api/stocks/")
    assert r.status_code == 200
    assert "stocks" in r.json()

    r = client.get("/api/stocks/TCS.NS")
    assert r.status_code == 200
    assert r.json()["symbol"] == "TCS.NS"


def test_analyze_and_downstream_api():
    # Run analysis
    payload = {
        "data_mode": "DEMO",
        "portfolio_k": 3,
        "corr_threshold": 0.60,
        "sector": "All Sectors",
        "period": "1y",
        "return_threshold": 5.0,
        "risk_threshold": 20.0,
    }
    r = client.post("/api/analyze", json=payload)
    assert r.status_code == 200, f"/api/analyze failed with {r.text}"
    data = r.json()
    assert data["stocks_analyzed"] > 0
    assert "top_portfolios" in data
    assert len(data["top_portfolios"]) > 0

    top1 = data["top_portfolios"][0]
    assert "portfolio" in top1, "portfolio key missing from top portfolio"
    assert isinstance(top1["portfolio"], list)
    assert len(top1["portfolio"]) == 3

    # Downstream GET endpoints
    r_corr = client.get("/api/correlation")
    assert r_corr.status_code == 200
    assert "matrix" in r_corr.json()

    r_graph = client.get("/api/graph?threshold=0.60")
    assert r_graph.status_code == 200
    assert "stats" in r_graph.json()

    r_color = client.get("/api/coloring")
    assert r_color.status_code == 200
    assert "color_groups" in r_color.json()

    r_rel = client.get("/api/relation")
    assert r_rel.status_code == 200
    assert "relation_pairs" in r_rel.json()

    r_poset = client.get("/api/poset")
    assert r_poset.status_code == 200
    assert r_poset.json()["is_partial_order"] is True

    r_hasse = client.get("/api/hasse")
    assert r_hasse.status_code == 200
    assert "cover_relation" in r_hasse.json()

    r_sets = client.get("/api/sets")
    assert r_sets.status_code == 200
    assert "A" in r_sets.json()

    r_bool = client.get("/api/boolean")
    assert r_bool.status_code == 200
    assert "rows" in r_bool.json()

    r_top = client.get("/api/portfolios/top")
    assert r_top.status_code == 200
    assert "top_portfolios" in r_top.json()

    r_all = client.get("/api/portfolios")
    assert r_all.status_code == 200
    assert "portfolios" in r_all.json()


def test_compare_portfolios():
    payload = {
        "portfolio_a": ["TCS.NS", "INFY.NS", "WIPRO.NS"],
        "portfolio_b": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS"]
    }
    r = client.post("/api/portfolios/compare", json=payload)
    assert r.status_code == 200
    res = r.json()
    assert "portfolio_a" in res
    assert "portfolio_b" in res
    assert "return_pct" in res["portfolio_a"]
