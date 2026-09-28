"""Sets & Boolean router — /api/sets, /api/boolean"""
from __future__ import annotations
from fastapi import APIRouter
from backend.schemas import AnalyzeRequest
from backend.pipeline import AnalysisPipeline

router = APIRouter(prefix="/api", tags=["Sets & Boolean"])


def _ensure_result() -> dict:
    from backend.routers.analysis import _last_result
    if not _last_result:
        result = AnalysisPipeline(AnalyzeRequest()).run()
        _last_result.update(result)
    return _last_result


@router.get("/sets")
def get_sets():
    r = _ensure_result()
    return r.get("set_operations", {"A": {}, "B": {}, "C": {}})


@router.get("/boolean")
def get_boolean():
    r = _ensure_result()
    return r.get("boolean_table", {
        "boolean_rule": "QUALIFIED = (Return > threshold) AND (Risk < threshold) AND (Volume > avg)",
        "rows": [],
        "qualified_count": 0,
    })
