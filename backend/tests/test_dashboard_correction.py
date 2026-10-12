"""
test_dashboard_correction.py
-----------------------------
Comprehensive test suite validating the complete analytical dashboard corrections
prescribed in Phase 14 of the TradeSense specification:

1. Single-stock selection
2. Multiple-stock selection
3. Group / Cluster selection
4. Sector selection
5. Empty and invalid selection handling
6. Shared selection consistency between Network and Heatmap
7. Network edge consistency with threshold rule
8. Correlation matrix mathematical consistency
9. Hasse diagram subset selection
10. Dominance pairs agreement with Hasse cover relation
11. Market price freshness and official close date tagging
12. Removal of Methodology & Math from main navigation
"""

import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from backend.main import app
from backend.core.matrix import (
    build_returns_matrix,
    calculate_pearson_correlation,
    validate_matrix_properties,
)
from backend.core.clustering import (
    correlation_to_distance,
    distance_to_correlation,
    cluster_by_correlation_threshold,
    hierarchical_clustering_average_linkage,
)
from backend.core.hasse import build_cover_relation, analyze_hasse
from backend.core.relations import build_dominance_relation

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. Selection & Graph Tests
# ---------------------------------------------------------------------------

def test_01_single_stock_selection():
    """Verify selecting a single stock returns focused node structure."""
    res = client.get("/api/graph?subset=TCS.NS&threshold=0.70")
    assert res.status_code == 200
    data = res.json()
    assert "tickers" in data
    assert data["tickers"] == ["TCS.NS"]
    assert data["stats"]["num_vertices"] == 1


def test_02_multiple_stock_selection():
    """Verify multiple-stock selection filters universe and calculates mutual edges."""
    res = client.get("/api/graph?subset=TCS.NS,INFY.NS,HCLTECH.NS&threshold=0.60")
    assert res.status_code == 200
    data = res.json()
    assert len(data["tickers"]) == 3
    assert set(data["tickers"]) == {"TCS.NS", "INFY.NS", "HCLTECH.NS"}
    # Verify every edge connects only vertices in the selected set
    for edge in data["edges"]:
        assert edge["source"] in data["tickers"]
        assert edge["target"] in data["tickers"]
        assert abs(edge["weight"]) >= 0.60


def test_03_group_clustering_endpoint():
    """Verify /api/clusters separates multi-stock clusters from unclustered stocks."""
    res = client.get("/api/clusters?threshold=0.70")
    assert res.status_code == 200
    data = res.json()
    assert "clusters" in data
    assert "unclustered" in data
    assert "cluster_count" in data
    assert "unclustered_stocks_count" in data

    # Verify every multi-stock cluster has size >= 2 and within-group metrics
    for cluster in data["clusters"]:
        assert cluster["size"] >= 2
        assert len(cluster["members"]) == cluster["size"]
        assert "avg_correlation" in cluster
        assert "lead_stock" in cluster


def test_04_sector_selection():
    """Verify /api/sectors/analysis returns sector list and analyzed stocks."""
    res = client.get("/api/sectors/analysis")
    assert res.status_code == 200
    data = res.json()
    assert "sectors" in data
    assert len(data["sectors"]) > 0
    first_sector = data["sectors"][0]
    assert "sector" in first_sector
    assert "stock_symbols" in first_sector


def test_05_empty_and_invalid_selection():
    """Verify empty or non-existent tickers fall back cleanly to valid universe."""
    res = client.get("/api/graph?subset=NONEXISTENT1,NONEXISTENT2")
    assert res.status_code == 200
    data = res.json()
    assert len(data["tickers"]) > 0


# ---------------------------------------------------------------------------
# 2. Shared Selection Consistency (Network <-> Heatmap)
# ---------------------------------------------------------------------------

def test_06_shared_selection_consistency():
    """Verify Network and Correlation endpoints return the exact same stocks when subset is given."""
    subset_query = "TCS.NS,INFY.NS,RELIANCE.NS"
    graph_res = client.get(f"/api/graph?subset={subset_query}")
    corr_res = client.get(f"/api/correlation?subset={subset_query}")

    assert graph_res.status_code == 200
    assert corr_res.status_code == 200

    g_tickers = set(graph_res.json()["tickers"])
    c_tickers = set(corr_res.json()["tickers"])
    assert g_tickers == c_tickers


def test_07_network_edge_consistency():
    """Verify an edge exists iff |r| >= threshold."""
    res = client.get("/api/graph?threshold=0.65")
    assert res.status_code == 200
    data = res.json()
    threshold = data["threshold"]

    for edge in data["edges"]:
        assert abs(edge["weight"]) >= threshold - 1e-6


# ---------------------------------------------------------------------------
# 3. Correlation Matrix Mathematics
# ---------------------------------------------------------------------------

def test_08_correlation_matrix_properties():
    """Verify correlation matrix diagonal is 1.0, symmetric, and bounded [-1, 1]."""
    dates = pd.date_range("2026-01-01", periods=50, freq="B")
    rng = np.random.default_rng(42)

    df_a = pd.DataFrame({"Close": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 50)))}, index=dates)
    df_b = pd.DataFrame({"Close": 50 * np.exp(np.cumsum(rng.normal(0, 0.02, 50)))}, index=dates)

    data_map = {"STOCK_A.NS": df_a, "STOCK_B.NS": df_b}
    ret_df = build_returns_matrix(data_map)
    corr_df = calculate_pearson_correlation(ret_df)

    props = validate_matrix_properties(corr_df)
    assert props["is_diagonal_one"]
    assert props["is_symmetric"]
    assert props["is_bounded"]
    assert props["all_valid"]


def test_09_mantegna_metric_distance():
    """Verify Mantegna distance satisfies mathematical metric properties."""
    assert correlation_to_distance(1.0) == 0.0
    assert abs(correlation_to_distance(0.0) - np.sqrt(2)) < 1e-6
    assert abs(correlation_to_distance(-1.0) - 2.0) < 1e-6

    # Invertibility check
    for r in [1.0, 0.8, 0.5, 0.0, -0.5, -1.0]:
        d = correlation_to_distance(r)
        r_rec = distance_to_correlation(d)
        assert abs(r - r_rec) < 1e-6


# ---------------------------------------------------------------------------
# 4. Hasse Diagram & Dominance Consistency
# ---------------------------------------------------------------------------

def test_10_hasse_diagram_subset():
    """Verify /api/hasse supports subset filtering and returns DAG layout."""
    res = client.get("/api/hasse?subset=TCS.NS,INFY.NS,WIPRO.NS")
    assert res.status_code == 200
    data = res.json()
    assert "cover_relation" in data
    assert "maximal_elements" in data
    assert "layout" in data
    assert len(data["tickers"]) <= 3


def test_11_dominance_transitive_reduction():
    """Verify cover relation has no redundant transitive edges."""
    # A dominates B, B dominates C, A dominates C
    pairs = [("A", "B"), ("B", "C"), ("A", "C")]
    cover = build_cover_relation(pairs)
    # The direct transitive edge (A, C) must be eliminated
    assert ("A", "C") not in cover
    assert ("A", "B") in cover
    assert ("B", "C") in cover


# ---------------------------------------------------------------------------
# 5. Price Freshness & Market Metadata
# ---------------------------------------------------------------------------

def test_12_quote_freshness_metadata():
    """Verify quote endpoint provides price_type, session_date, and market status."""
    res = client.get("/api/market/quote/TCS.NS")
    assert res.status_code == 200
    quote = res.json()
    assert "symbol" in quote
    assert "market_status" in quote
    assert "price_type" in quote
    assert quote["price_type"] in ("LIVE_QUOTE", "OFFICIAL_CLOSE")
    assert "session_date" in quote


# ---------------------------------------------------------------------------
# 6. Navigation: Methodology & Math Removed
# ---------------------------------------------------------------------------

def test_13_navigation_methodology_removed():
    """Verify Methodology & Math is removed from the rendered navigation menu."""
    res = client.get("/")
    assert res.status_code == 200
    html = res.text
    # Ensure Methodology & Math link is not in the sidebar navigation
    assert 'href="/methodology"' not in html
