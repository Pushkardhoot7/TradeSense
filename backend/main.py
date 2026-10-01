"""
TradeSense V2 — FastAPI Application Entry Point
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

load_dotenv()

from backend.database.db import init_db
from backend.routers import (
    analysis,
    backtest_router,
    coloring_router,
    discrete_math,
    market,
    portfolio_router,
    sets_boolean,
    stocks,
    technical_router,
)

PROJECT_ROOT = Path(__file__).parent.parent
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger("tradesense")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("TradeSense V2 starting — initialising database...")
    init_db()
    logger.info("Database ready.")

    # Pre-warm the pipeline so the first browser page load is instant
    import asyncio
    async def _prewarm():
        await asyncio.sleep(0.5)   # let server finish binding
        try:
            from backend.pipeline import AnalysisPipeline
            from backend.schemas import AnalyzeRequest
            from backend.routers.analysis import _last_result
            if not _last_result:
                # Try HISTORICAL (real yfinance data) first
                logger.info("Pre-warming pipeline with HISTORICAL (yfinance) data...")
                try:
                    result = await asyncio.to_thread(
                        AnalysisPipeline(AnalyzeRequest(data_mode="HISTORICAL")).run
                    )
                    _last_result.update(result)
                    logger.info("Pre-warm complete (HISTORICAL) — %d stocks, top DM=%.2f",
                                result.get("stocks_analyzed", 0),
                                result.get("top_dm_score", 0))
                except Exception as hist_exc:
                    logger.warning("HISTORICAL pre-warm failed: %s — falling back to DEMO", hist_exc)
                    result = await asyncio.to_thread(
                        AnalysisPipeline(AnalyzeRequest(data_mode="DEMO")).run
                    )
                    _last_result.update(result)
                    logger.info("Pre-warm complete (DEMO fallback) — %d stocks, top DM=%.2f",
                                result.get("stocks_analyzed", 0),
                                result.get("top_dm_score", 0))
        except Exception as exc:
            logger.warning("Pre-warm failed (non-fatal): %s", exc)

    asyncio.create_task(_prewarm())
    yield
    logger.info("TradeSense V2 shutdown.")


app = FastAPI(
    title="TradeSense V2",
    description="Discrete Mathematics Powered Stock Market Analysis & Portfolio Intelligence",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files & templates
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")
templates = Jinja2Templates(directory=str(PROJECT_ROOT / "templates"))

# API routers
app.include_router(analysis.router)
app.include_router(market.router)
app.include_router(stocks.router)
app.include_router(discrete_math.router)
app.include_router(portfolio_router.router)
app.include_router(sets_boolean.router)
app.include_router(coloring_router.router)
app.include_router(technical_router.router)
app.include_router(backtest_router.router)


# ---------------------------------------------------------------------------
# HTML page routes — serve Jinja2 templates
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/live-market", include_in_schema=False)
@app.get("/market-scanner", include_in_schema=False)
async def live_market_page(request: Request):
    return templates.TemplateResponse(request=request, name="live_market.html")

@app.get("/stock-explorer", include_in_schema=False)
async def stock_explorer_page(request: Request):
    return templates.TemplateResponse(request=request, name="stock_explorer.html")

@app.get("/stock/{symbol}", include_in_schema=False)
async def stock_detail(request: Request, symbol: str):
    return templates.TemplateResponse(request=request, name="stock_detail.html", context={"symbol": symbol})

@app.get("/relationships", include_in_schema=False)
@app.get("/correlation", include_in_schema=False)
async def relationships_page(request: Request):
    return templates.TemplateResponse(request=request, name="relationships.html")

@app.get("/market-network", include_in_schema=False)
async def market_network_page(request: Request):
    return templates.TemplateResponse(request=request, name="market_network.html")

@app.get("/hasse-ranking", include_in_schema=False)
@app.get("/relations-hasse", include_in_schema=False)
async def hasse_ranking_page(request: Request):
    return templates.TemplateResponse(request=request, name="hasse_ranking.html")

@app.get("/stock-groups", include_in_schema=False)
@app.get("/graph-coloring", include_in_schema=False)
async def stock_groups_page(request: Request):
    return templates.TemplateResponse(request=request, name="stock_groups.html")

@app.get("/portfolio-lab", include_in_schema=False)
@app.get("/portfolio-optimizer", include_in_schema=False)
@app.get("/sets-boolean", include_in_schema=False)
async def portfolio_lab_page(request: Request):
    return templates.TemplateResponse(request=request, name="portfolio_lab.html")

@app.get("/how-tradesense-thinks", include_in_schema=False)
async def how_tradesense_thinks_page(request: Request):
    return templates.TemplateResponse(request=request, name="how_tradesense_thinks.html")

@app.get("/market-news", include_in_schema=False)
async def market_news_page(request: Request):
    return templates.TemplateResponse(request=request, name="market_news.html")

@app.get("/academic-mode", include_in_schema=False)
async def academic_mode_page(request: Request):
    return templates.TemplateResponse(request=request, name="academic.html")

@app.get("/viva-mode", include_in_schema=False)
async def viva_mode_page(request: Request):
    return templates.TemplateResponse(request=request, name="viva.html")

@app.get("/backtesting", include_in_schema=False)
@app.get("/scenario", include_in_schema=False)
async def backtesting_page(request: Request):
    return templates.TemplateResponse(request=request, name="backtesting.html")

@app.get("/reports", include_in_schema=False)
async def reports_page(request: Request):
    return templates.TemplateResponse(request=request, name="reports.html")

@app.get("/methodology", include_in_schema=False)
async def methodology_page(request: Request):
    return templates.TemplateResponse(request=request, name="academic.html")


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc), "path": str(request.url)})
