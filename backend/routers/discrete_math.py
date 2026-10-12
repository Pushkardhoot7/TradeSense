from __future__ import annotations
from typing import Optional
from fastapi import APIRouter, Query
from backend.schemas import AnalyzeRequest
from backend.pipeline import AnalysisPipeline

router = APIRouter(prefix="/api", tags=["Discrete Math"])


def _ensure_result() -> dict:
    from backend.routers.analysis import _last_result
    if not _last_result:
        result = AnalysisPipeline(AnalyzeRequest()).run()
        _last_result.update(result)
    return _last_result


from backend.core.hasse import build_cover_relation, analyze_hasse, get_hasse_layout
from backend.core.relations import build_dominance_relation


@router.get("/relation")
def get_relation(subset: Optional[str] = Query(default=None)):
    r = _ensure_result()
    metrics_list = r.get("stocks", [])
    tickers = r.get("tickers", [])

    if subset:
        req = [s.strip() for s in subset.split(",") if s.strip()]
        active_tickers = [s for s in req if s in tickers]
        if active_tickers:
            active_metrics = [m for m in metrics_list if m.get("symbol") in active_tickers]
            dom_pairs = build_dominance_relation(active_metrics)
            return {
                "relation_pairs": [{"dominator": a, "dominated": b} for a, b in dom_pairs],
                "dominance_pairs": [{"dominator": a, "dominated": b} for a, b in dom_pairs],
                "tickers": active_tickers,
                "all_tickers": tickers,
                "stock_metrics": {m["symbol"]: m for m in active_metrics},
                "poset_properties": r.get("poset_properties", {
                    "is_reflexive": True,
                    "is_antisymmetric": True,
                    "is_transitive": True,
                    "is_partial_order": True,
                }),
            }

    metrics_map = {m["symbol"]: m for m in metrics_list if "symbol" in m}
    return {
        "relation_pairs": r.get("dominance_pairs", []),
        "dominance_pairs": r.get("dominance_pairs", []),
        "relation_matrix": r.get("relation_matrix", []),
        "tickers": tickers,
        "all_tickers": tickers,
        "stock_metrics": metrics_map,
        "non_dominated": r.get("non_dominated_stocks", []),
        "incomparable_pairs": r.get("incomparable_pairs", []),
        "most_dominant": r.get("most_dominant_stocks", []),
        "poset": r.get("poset_properties", {
            "is_reflexive": True,
            "is_antisymmetric": True,
            "is_transitive": True,
            "is_partial_order": True,
        }),
        "poset_properties": r.get("poset_properties", {
            "is_reflexive": True,
            "is_antisymmetric": True,
            "is_transitive": True,
            "is_partial_order": True,
        }),
    }


@router.get("/poset")
def get_poset():
    r = _ensure_result()
    return r.get("poset_properties", {
        "is_reflexive": True,
        "is_antisymmetric": True,
        "is_transitive": True,
        "is_partial_order": True,
    })


@router.get("/hasse")
def get_hasse(subset: Optional[str] = Query(default=None)):
    r = _ensure_result()
    tickers = r.get("tickers", [])
    metrics_list = r.get("stocks", [])
    metrics_map = {m["symbol"]: m for m in metrics_list if "symbol" in m}

    if subset:
        req = [s.strip() for s in subset.split(",") if s.strip()]
        active_tickers = [s for s in req if s in tickers]
        if active_tickers:
            active_metrics = [m for m in metrics_list if m.get("symbol") in active_tickers]
            dom_pairs = build_dominance_relation(active_metrics)
            cover_pairs = build_cover_relation(dom_pairs)
            hasse_info = analyze_hasse(cover_pairs, active_tickers)
            hasse_layout = get_hasse_layout(cover_pairs, active_tickers)

            return {
                "cover_relation": [{"from": a, "to": b} for a, b in cover_pairs],
                "dominance_pairs": [{"dominator": a, "dominated": b} for a, b in dom_pairs],
                "levels": hasse_info.get("levels", {}),
                "maximal_elements": hasse_info.get("maximal_elements", []),
                "minimal_elements": hasse_info.get("minimal_elements", []),
                "layout": hasse_layout,
                "tickers": active_tickers,
                "all_tickers": tickers,
                "stock_metrics": {m["symbol"]: m for m in active_metrics},
            }

    # Default full view: if universe is large (>15), prepare focused top-tier layout
    return {
        "cover_relation": r.get("hasse_cover_relation", []),
        "dominance_pairs": r.get("dominance_pairs", []),
        "levels": r.get("hasse_levels", {}),
        "maximal_elements": r.get("hasse_maximal", []),
        "minimal_elements": r.get("hasse_minimal", []),
        "layout": r.get("hasse_layout", {}),
        "tickers": tickers,
        "all_tickers": tickers,
        "stock_metrics": metrics_map,
    }
