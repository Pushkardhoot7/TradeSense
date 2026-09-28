"""Coloring router — /api/coloring"""
from __future__ import annotations
from fastapi import APIRouter
from backend.schemas import AnalyzeRequest
from backend.pipeline import AnalysisPipeline

router = APIRouter(prefix="/api", tags=["Graph Coloring"])


def _ensure_result() -> dict:
    from backend.routers.analysis import _last_result
    if not _last_result:
        result = AnalysisPipeline(AnalyzeRequest()).run()
        _last_result.update(result)
    return _last_result


@router.get("/coloring")
def get_coloring():
    r = _ensure_result()
    return {
        "coloring_dict":    r.get("coloring_dict", {}),
        "color_groups":     r.get("color_groups", {}),
        "num_colors":       r.get("num_color_groups", 0),
        "is_valid":         r.get("coloring_valid", False),
        "chromatic_info":   r.get("chromatic_info", {}),
        "step_trace":       r.get("coloring_step_trace", []),
        "independent_sets_verified": r.get("independent_sets_verified", {}),
    }
