"""Stocks router — /api/stocks, /api/stocks/sectors, /api/stocks/{symbol}"""
from __future__ import annotations
import csv
from pathlib import Path
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/stocks", tags=["Stocks"])
UNIVERSE_CSV = Path(__file__).parent.parent.parent / "data" / "stock_universe.csv"


def _load_universe() -> list[dict]:
    if not UNIVERSE_CSV.exists():
        return []
    stocks = []
    with open(UNIVERSE_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sym = row.get("symbol", "").strip()
            if sym and sym.endswith(".NS"):
                stocks.append({
                    "symbol":       sym,
                    "company_name": row.get("company_name", sym),
                    "sector":       row.get("sector", "Unknown"),
                    "industry":     row.get("industry", ""),
                    "exchange":     row.get("exchange", "NSE"),
                })
    return stocks


def _ensure_metrics():
    from backend.routers.analysis import _last_result
    from backend.pipeline import AnalysisPipeline
    from backend.schemas import AnalyzeRequest
    if not _last_result or not _last_result.get("stocks"):
        try:
            res = AnalysisPipeline(AnalyzeRequest(data_mode="HISTORICAL")).run()
        except Exception:
            res = AnalysisPipeline(AnalyzeRequest(data_mode="DEMO")).run()
        _last_result.update(res)
    return _last_result


@router.get("/sectors")
def get_sectors():
    universe = _load_universe()
    sectors = sorted({s["sector"] for s in universe if s["sector"]})
    return {"sectors": sectors}


@router.get("")
@router.get("/")
def list_stocks(sector: str = "All Sectors"):
    _ensure_metrics()
    from backend.routers.analysis import _last_result
    analyzed_map = {m["symbol"]: m for m in _last_result.get("stocks", [])}
    
    universe = _load_universe()
    if sector and sector != "All Sectors":
        universe = [s for s in universe if s["sector"].lower() == sector.lower()]

    # Merge computed metrics for each stock
    enriched = []
    # First, include analyzed stocks
    for s in universe:
        sym = s["symbol"]
        if sym in analyzed_map:
            enriched.append({**s, **analyzed_map[sym]})
        else:
            # If not in the active analyzed subset, attach default baseline metrics
            enriched.append({
                **s,
                "latest_price": None,
                "return_pct": None,
                "risk_pct": None,
                "sharpe": None,
                "cagr": None,
                "avg_volume": None
            })

    # If the user is on the default view, prioritize analyzed stocks at the top
    enriched.sort(key=lambda x: (x.get("return_pct") is None, -(x.get("return_pct") or 0)))
    return {"stocks": enriched, "count": len(enriched)}


@router.get("/{symbol}")
def get_stock(symbol: str):
    _ensure_metrics()
    from backend.routers.analysis import _last_result
    universe = _load_universe()
    info = next((s for s in universe if s["symbol"].upper() == symbol.upper()), None)
    if not info:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found in universe")
    metrics = {}
    if _last_result:
        for m in _last_result.get("stocks", []):
            if m.get("symbol") == symbol:
                metrics = m
                break
    return {**info, **metrics, "metrics": metrics}


@router.get("/{symbol}/history")
def get_history(symbol: str, period: str = "1y"):
    """Return historical prices for a stock."""
    from backend.routers.analysis import _last_result
    hist = _last_result.get("price_history", {}).get(symbol, [])
    return {"symbol": symbol, "period": period, "prices": hist}
