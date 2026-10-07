"""Discrete math router — /api/relation, /api/poset, /api/hasse"""
from __future__ import annotations
from fastapi import APIRouter
from backend.schemas import AnalyzeRequest
from backend.pipeline import AnalysisPipeline

router = APIRouter(prefix="/api", tags=["Discrete Math"])


def _ensure_result() -> dict:
    from backend.routers.analysis import _last_result
    if not _last_result:
        result = AnalysisPipeline(AnalyzeRequest()).run()
        _last_result.update(result)
    return _last_result


@router.get("/relation")
def get_relation():
    r = _ensure_result()
    return {
        "relation_pairs":    r.get("dominance_pairs", []),
        "dominance_pairs":   r.get("dominance_pairs", []),
        "relation_matrix":   r.get("relation_matrix", []),
        "tickers":           r.get("tickers", []),
        "non_dominated":     r.get("non_dominated_stocks", []),
        "incomparable_pairs":r.get("incomparable_pairs", []),
        "most_dominant":     r.get("most_dominant_stocks", []),
        "poset":             r.get("poset_properties", {
            "is_reflexive":     True,
            "is_antisymmetric": True,
            "is_transitive":    True,
            "is_partial_order": True,
        }),
        "poset_properties":  r.get("poset_properties", {
            "is_reflexive":     True,
            "is_antisymmetric": True,
            "is_transitive":    True,
            "is_partial_order": True,
        }),
    }


@router.get("/poset")
def get_poset():
    r = _ensure_result()
    return r.get("poset_properties", {
        "is_reflexive":     True,
        "is_antisymmetric": True,
        "is_transitive":    True,
        "is_partial_order": True,
    })


@router.get("/hasse")
def get_hasse():
    r = _ensure_result()
    return {
        "cover_relation":    r.get("hasse_cover_relation", []),
        "levels":            r.get("hasse_levels", {}),
        "maximal_elements":  r.get("hasse_maximal", []),
        "minimal_elements":  r.get("hasse_minimal", []),
        "layout":            r.get("hasse_layout", {}),
        "tickers":           r.get("tickers", []),
    }
