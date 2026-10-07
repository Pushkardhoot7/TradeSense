"""Analysis router — /api/analyze, /api/health, /api/refresh, /api/correlation, /api/graph"""
from __future__ import annotations
import logging
import math
from typing import Any

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
    return _last_result.get("data_quality", {
        "quality_score": 100.0,
        "completeness_pct": 100.0,
        "freshness": "Up to date",
        "coverage_ratio": "100%",
        "valid_stocks": _last_result.get("stocks_analyzed", 0),
        "total_requested": _last_result.get("stocks_analyzed", 0),
        "missing_stocks": [],
        "status": "EXCELLENT"
    })


@router.get("/correlation")
def correlation():
    if not _last_result:
        result = AnalysisPipeline(AnalyzeRequest()).run()
        _last_result.update(result)
    corr = _last_result.get("correlation_matrix", {})
    stats = _last_result.get("correlation_stats", {})
    validation = _last_result.get("matrix_validation", {})
    corr_df = _last_result.get("correlation_matrix_df")
    
    strong_pairs = []
    if corr_df is not None and not corr_df.empty:
        from backend.core.matrix import extract_upper_triangle
        strong_pairs = extract_upper_triangle(corr_df)
    
    return {
        "matrix": corr,
        "stats": stats,
        "validation": validation,
        "tickers": _last_result.get("tickers", []),
        "strong_pairs": strong_pairs,
    }


@router.get("/graph")
def graph(threshold: float = Query(default=0.70, ge=0.5, le=0.95)):
    if not _last_result:
        result = AnalysisPipeline(AnalyzeRequest()).run()
        _last_result.update(result)
    # Re-build graph at requested threshold using existing correlation matrix
    corr = _last_result.get("correlation_matrix_df")
    tickers = _last_result.get("tickers", [])
    if corr is None:
        return {"error": "No analysis data available. Run /api/analyze first."}
    from backend.core.graph import build_adjacency_dict, calculate_graph_statistics, bfs, dfs, connected_components_from_scratch
    adj = build_adjacency_dict(tickers, corr, threshold)
    stats = calculate_graph_statistics(adj, tickers)
    most_connected = stats.get("most_connected_stock", tickers[0] if tickers else "")
    bfs_order, bfs_trace = bfs(adj, most_connected) if most_connected else ([], [])
    dfs_order, dfs_trace = dfs(adj, most_connected) if most_connected else ([], [])
    components = connected_components_from_scratch(adj)
    edges = [{"source": src, "target": tgt, "weight": round(w, 4)}
             for src, neighbors in adj.items() for tgt, w in neighbors if src < tgt]
    coloring_dict = _last_result.get("coloring", {}).get("coloring_dict", {})
    return {"threshold": threshold, "stats": stats, "edges": edges, "tickers": tickers,
            "coloring_dict": coloring_dict,
            "bfs": {"start": most_connected, "order": bfs_order, "trace": bfs_trace},
            "dfs": {"start": most_connected, "order": dfs_order, "trace": dfs_trace},
            "connected_components": components}
