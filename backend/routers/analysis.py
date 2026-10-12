"""Analysis router — /api/analyze, /api/health, /api/refresh, /api/correlation, /api/graph"""
from __future__ import annotations
import logging
import math
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Query
from backend.schemas import AnalyzeRequest
from backend.pipeline import AnalysisPipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["Analysis"])

# In-memory cache of last analysis result
_last_result: dict = {}


def _sanitize(obj: Any) -> Any:
    """
    Recursively convert a pipeline result dict into a JSON-safe structure.

    - Drops pandas DataFrame and Series objects (stored in result for internal use only).
    - Converts np.float64 NaN / inf → None.
    - Converts numpy scalars → native Python types.
    """
    import numpy as np

    # Non-serializable internal objects — silently drop
    if isinstance(obj, (pd.DataFrame, pd.Series)):
        return None

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj

    # numpy scalar types
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_sanitize(x) for x in obj.tolist()]

    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items() if not isinstance(v, (pd.DataFrame, pd.Series))}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(x) for x in obj]

    return obj


@router.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0", "app": "TradeSense V2"}


@router.post("/analyze")
def analyze(req: AnalyzeRequest):
    global _last_result
    pipeline = AnalysisPipeline(req)
    result = pipeline.run()
    _last_result = result          # keep raw (with DataFrames) for internal use
    return _sanitize(result)       # return JSON-safe version to client


@router.post("/refresh")
def refresh(req: AnalyzeRequest):
    """Clears cache and re-fetches fresh data."""
    global _last_result
    _last_result = {}
    pipeline = AnalysisPipeline(req)
    result = pipeline.run()
    _last_result = result
    return _sanitize(result)


@router.get("/sectors/analysis")
def get_sectors_analysis():
    global _last_result
    if not _last_result or "sector_analysis" not in _last_result:
        try:
            res = AnalysisPipeline(AnalyzeRequest(data_mode="LIVE")).run()
            _last_result.update(res)
        except Exception:
            res = AnalysisPipeline(AnalyzeRequest(data_mode="DEMO")).run()
            _last_result.update(res)
    return {
        "sectors": _last_result.get("sector_analysis", []),
        "total_sectors": len(_last_result.get("sector_analysis", [])),
        "insights": _last_result.get("sector_insights", []),
        "timestamp": _last_result.get("created_at", "")
    }


@router.get("/data-quality")
def get_data_quality():
    global _last_result
    if not _last_result:
        try:
            res = AnalysisPipeline(AnalyzeRequest(data_mode="LIVE")).run()
            _last_result.update(res)
        except Exception:
            res = AnalysisPipeline(AnalyzeRequest(data_mode="DEMO")).run()
            _last_result.update(res)
    ctx = _last_result.get("analysis_context", {})
    dq = _last_result.get("data_quality", {}).copy() if isinstance(_last_result.get("data_quality"), dict) else {}
    dq.update({
        "requested_stocks": ctx.get("requested_count", len(_last_result.get("tickers", []))),
        "valid_stocks": ctx.get("validated_count", len(_last_result.get("tickers", []))),
        "excluded_stocks": ctx.get("excluded_count", 0),
        "excluded_tickers": ctx.get("excluded_tickers", []),
        "exclusion_reasons": ctx.get("exclusion_reasons", {}),
        "data_provider": ctx.get("data_provider", _last_result.get("provider_label", "UNKNOWN")),
        "data_freshness": ctx.get("data_freshness", _last_result.get("last_updated", "")),
        "data_period": ctx.get("data_period", _last_result.get("period", "")),
        "pipeline_status": ctx.get("pipeline_status", _last_result.get("pipeline_status", "SUCCESS")),
        "analysis_id": ctx.get("analysis_id", _last_result.get("analysis_id", "")),
    })
    return dq


@router.get("/context")
def get_analysis_context():
    """Return canonical analysis context containing eligible and excluded universes."""
    global _last_result
    if not _last_result or "analysis_context" not in _last_result:
        try:
            res = AnalysisPipeline(AnalyzeRequest(data_mode="LIVE")).run()
            _last_result.update(res)
        except Exception:
            res = AnalysisPipeline(AnalyzeRequest(data_mode="DEMO")).run()
            _last_result.update(res)
    return _last_result.get("analysis_context", {})


@router.get("/correlation")
def correlation(subset: Optional[str] = Query(default=None)):
    global _last_result
    if not _last_result:
        result = AnalysisPipeline(AnalyzeRequest()).run()
        _last_result.update(result)
    corr = _last_result.get("correlation_matrix", {})
    stats = _last_result.get("correlation_stats", {})
    validation = _last_result.get("matrix_validation", {})
    corr_df = _last_result.get("correlation_matrix_df")
    tickers = _last_result.get("tickers", [])

    if corr_df is not None and not corr_df.empty:
        active_tickers = tickers
        if subset:
            req_symbols = [s.strip() for s in subset.split(",") if s.strip()]
            active_tickers = [s for s in req_symbols if s in corr_df.columns]
            if not active_tickers:
                active_tickers = tickers

        sub_df = corr_df.loc[active_tickers, active_tickers]
        from backend.core.matrix import extract_upper_triangle, get_correlation_stats, validate_matrix_properties
        strong_pairs = extract_upper_triangle(sub_df)
        sub_stats = get_correlation_stats(sub_df)
        sub_valid = validate_matrix_properties(sub_df)

        return {
            "matrix": sub_df.to_dict(),
            "stats": sub_stats,
            "validation": sub_valid,
            "tickers": active_tickers,
            "all_tickers": tickers,
            "strong_pairs": strong_pairs,
        }

    return {
        "matrix": corr,
        "stats": stats,
        "validation": validation,
        "tickers": tickers,
        "all_tickers": tickers,
        "strong_pairs": [],
    }


@router.get("/graph")
def graph(
    threshold: float = Query(default=0.70, ge=0.3, le=0.95),
    subset: Optional[str] = Query(default=None),
):
    global _last_result
    if not _last_result:
        try:
            result = AnalysisPipeline(AnalyzeRequest(data_mode="LIVE")).run()
            _last_result.update(result)
        except Exception:
            result = AnalysisPipeline(AnalyzeRequest(data_mode="DEMO")).run()
            _last_result.update(result)

    # Re-build graph at requested threshold using existing correlation matrix
    corr = _last_result.get("correlation_matrix_df")
    tickers = _last_result.get("tickers", [])

    if corr is None or (isinstance(corr, pd.DataFrame) and corr.empty):
        raw_mat = _last_result.get("correlation_matrix")
        if raw_mat and isinstance(raw_mat, dict):
            corr = pd.DataFrame(raw_mat)
            _last_result["correlation_matrix_df"] = corr

    if corr is None or (isinstance(corr, pd.DataFrame) and corr.empty):
        try:
            result = AnalysisPipeline(AnalyzeRequest(data_mode="DEMO")).run()
            _last_result.update(result)
            corr = _last_result.get("correlation_matrix_df")
            tickers = _last_result.get("tickers", [])
        except Exception:
            return {"error": "No analysis data available. Run /api/analyze first."}

    if corr is None or (isinstance(corr, pd.DataFrame) and corr.empty):
        return {"error": "No analysis data available. Run /api/analyze first."}

    # Handle subset selection if provided
    active_tickers = tickers
    if subset:
        req_symbols = [s.strip() for s in subset.split(",") if s.strip()]
        filtered = [s for s in req_symbols if s in corr.columns]
        if filtered:
            active_tickers = filtered

    from backend.core.graph import build_adjacency_dict, calculate_graph_statistics, bfs, dfs, connected_components_from_scratch
    from backend.core.clustering import cluster_by_correlation_threshold

    adj = build_adjacency_dict(active_tickers, corr, threshold)
    stats = calculate_graph_statistics(adj, active_tickers)
    most_connected = stats.get("most_connected_stock", active_tickers[0] if active_tickers else "")
    bfs_order, bfs_trace = bfs(adj, most_connected) if most_connected else ([], [])
    dfs_order, dfs_trace = dfs(adj, most_connected) if most_connected else ([], [])
    components = connected_components_from_scratch(adj)
    edges = [{"source": src, "target": tgt, "weight": round(float(w), 4)}
             for src, neighbors in adj.items() for tgt, w in neighbors if src < tgt]

    # Calculate genuine correlation clusters
    clustering_res = cluster_by_correlation_threshold(active_tickers, corr, threshold=threshold)
    clusters = clustering_res.get("clusters", [])
    unclustered = clustering_res.get("unclustered", [])

    coloring_dict = _last_result.get("coloring_dict") or _last_result.get("coloring", {}).get("coloring_dict", {})

    # Phase 2 Diagnostic Logging
    valid_price_count = len(_last_result.get("price_data", {}))
    logger.info(
        "[Graph Diagnostics] Total Eligible Stocks: %d | Active Subset: %d | "
        "Valid Price Series: %d | Threshold: %.2f | Qualifying Edges: %d | "
        "Multi-Stock Clusters: %d | Unclustered: %d",
        len(tickers), len(active_tickers), valid_price_count, threshold, len(edges),
        len(clusters), len(unclustered)
    )

    diagnostics = {
        "eligible_stocks_count": len(tickers),
        "active_selection_count": len(active_tickers),
        "valid_price_series_count": valid_price_count,
        "threshold": threshold,
        "qualifying_edges_count": len(edges),
        "multi_stock_clusters_count": len(clusters),
        "unclustered_stocks_count": len(unclustered),
    }

    return {
        "threshold": threshold,
        "stats": stats,
        "edges": edges,
        "tickers": active_tickers,
        "all_tickers": tickers,
        "clusters": clusters,
        "unclustered": unclustered,
        "coloring_dict": coloring_dict,
        "bfs": {"start": most_connected, "order": bfs_order, "trace": bfs_trace},
        "dfs": {"start": most_connected, "order": dfs_order, "trace": dfs_trace},
        "connected_components": components,
        "diagnostics": diagnostics,
    }


@router.get("/clusters")
def get_clusters(
    threshold: float = Query(default=0.70, ge=0.3, le=0.95),
    subset: Optional[str] = Query(default=None),
):
    """Retrieve correlation-based co-movement stock clusters and unclustered singletons."""
    global _last_result
    if not _last_result:
        try:
            result = AnalysisPipeline(AnalyzeRequest(data_mode="LIVE")).run()
            _last_result.update(result)
        except Exception:
            result = AnalysisPipeline(AnalyzeRequest(data_mode="DEMO")).run()
            _last_result.update(result)

    corr = _last_result.get("correlation_matrix_df")
    tickers = _last_result.get("tickers", [])

    if corr is None or (isinstance(corr, pd.DataFrame) and corr.empty):
        raw_mat = _last_result.get("correlation_matrix")
        if raw_mat and isinstance(raw_mat, dict):
            corr = pd.DataFrame(raw_mat)
            _last_result["correlation_matrix_df"] = corr

    if corr is None or (isinstance(corr, pd.DataFrame) and corr.empty):
        return {"clusters": [], "unclustered": [], "total_stocks": 0}

    active_tickers = tickers
    if subset:
        req_symbols = [s.strip() for s in subset.split(",") if s.strip()]
        filtered = [s for s in req_symbols if s in corr.columns]
        if filtered:
            active_tickers = filtered

    from backend.core.clustering import cluster_by_correlation_threshold
    return cluster_by_correlation_threshold(active_tickers, corr, threshold=threshold)
