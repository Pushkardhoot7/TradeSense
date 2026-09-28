"""Portfolio router — /api/portfolios, /api/portfolios/top, /api/portfolios/compare"""
from __future__ import annotations
from fastapi import APIRouter, Query
from pydantic import BaseModel
from backend.schemas import AnalyzeRequest
from backend.pipeline import AnalysisPipeline

router = APIRouter(prefix="/api", tags=["Portfolio"])


def _ensure_result() -> dict:
    from backend.routers.analysis import _last_result
    if not _last_result:
        result = AnalysisPipeline(AnalyzeRequest()).run()
        _last_result.update(result)
    return _last_result


@router.get("/portfolios")
def all_portfolios(limit: int = Query(default=50, le=500)):
    r = _ensure_result()
    all_p = r.get("all_portfolios", [])
    # ALWAYS ensure 'portfolio' key exists (KeyError fix)
    safe = []
    for p in all_p[:limit]:
        item = dict(p)
        if "portfolio" not in item:
            item["portfolio"] = []
        safe.append(item)
    return {"portfolios": safe, "total": len(all_p)}


@router.get("/portfolios/top")
def top_portfolios(n: int = Query(default=3, le=10)):
    r = _ensure_result()
    top = r.get("top_portfolios", [])
    # ALWAYS ensure 'portfolio' key is a list[str] — never raises KeyError
    safe = []
    for p in top[:n]:
        item = dict(p)
        if "portfolio" not in item:
            item["portfolio"] = []
        if not isinstance(item["portfolio"], list):
            item["portfolio"] = list(item["portfolio"])
        safe.append(item)
    return {"top_portfolios": safe, "count": len(safe),
            "dm_weights": r.get("dm_weights", {}),
            "disclaimer": r.get("disclaimer", "")}


class CompareRequest(BaseModel):
    portfolio_a: list[str]
    portfolio_b: list[str]


@router.post("/portfolios/compare")
def compare_portfolios(req: CompareRequest):
    r = _ensure_result()
    metrics = r.get("stocks", [])
    returns_df = r.get("returns_df")
    corr_df = r.get("correlation_matrix_df")
    from backend.core.portfolio import evaluate_portfolio
    a = evaluate_portfolio(tuple(req.portfolio_a), metrics, returns_df, corr_df)
    b = evaluate_portfolio(tuple(req.portfolio_b), metrics, returns_df, corr_df)
    return {"portfolio_a": a, "portfolio_b": b}
