# TradeSense V2 — `backend/database/crud.py`
"""
CRUD helpers for SQLite via SQLAlchemy ORM.

All functions accept a `db: Session` from `get_db()` dependency injection.
Data is persisted from pipeline results so runs survive server restarts
in multi-session use.
"""
from __future__ import annotations

from datetime import datetime
import json
from typing import Any

from sqlalchemy.orm import Session
from sqlalchemy import desc

from backend.database.models import (
    Stock,
    HistoricalPrice,
    StockMetric,
    Correlation,
    DominanceRelation,
    ColorAssignment,
    PortfolioCandidate,
    PortfolioScore,
    Analysis,
)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def create_analysis(db: Session, analysis_id: str, config: dict) -> Analysis:
    obj = Analysis(
        id=analysis_id,
        data_mode=config.get("data_mode", "DEMO"),
        sector=config.get("sector", "All Sectors"),
        period=config.get("period", "1y"),
        corr_threshold=config.get("corr_threshold", 0.70),
        portfolio_k=config.get("portfolio_k", 3),
        stocks_analyzed=config.get("stocks_analyzed", 0),
        portfolios_evaluated=config.get("total_portfolios_evaluated", 0),
        top_dm_score=config.get("top_dm_score"),
        is_complete=True,
    )
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_latest_analysis(db: Session) -> Analysis | None:
    return db.query(Analysis).order_by(desc(Analysis.created_at)).first()


# ---------------------------------------------------------------------------
# Stocks
# ---------------------------------------------------------------------------

def upsert_stock(db: Session, symbol: str, company_name: str, sector: str, industry: str = "", exchange: str = "NSE") -> Stock:
    obj = db.query(Stock).filter(Stock.symbol == symbol).first()
    if obj is None:
        obj = Stock(symbol=symbol, company_name=company_name, sector=sector, industry=industry, exchange=exchange)
        db.add(obj)
    else:
        obj.company_name = company_name
        obj.sector = sector
        obj.industry = industry
        obj.exchange = exchange
        obj.last_updated = datetime.utcnow()
    db.commit()
    db.refresh(obj)
    return obj


def get_stock(db: Session, symbol: str) -> Stock | None:
    return db.query(Stock).filter(Stock.symbol == symbol).first()


def get_all_stocks(db: Session) -> list[Stock]:
    return db.query(Stock).all()


# ---------------------------------------------------------------------------
# Historical Prices
# ---------------------------------------------------------------------------

def bulk_upsert_prices(db: Session, stock_id: int, prices: list[dict]) -> int:
    """Insert price rows, skipping duplicates by (stock_id, timestamp)."""
    inserted = 0
    for row in prices:
        date_val = row.get("date") or row.get("timestamp")
        if not date_val:
            continue
        if isinstance(date_val, str):
            dt = datetime.strptime(date_val[:10], "%Y-%m-%d")
        else:
            dt = date_val

        existing = (
            db.query(HistoricalPrice)
            .filter(HistoricalPrice.stock_id == stock_id, HistoricalPrice.timestamp == dt)
            .first()
        )
        if existing is None:
            db.add(HistoricalPrice(
                stock_id=stock_id,
                timestamp=dt,
                open=row.get("open") or row.get("Open"),
                high=row.get("high") or row.get("High"),
                low=row.get("low") or row.get("Low"),
                close=float(row.get("close") or row.get("Close") or 0.0),
                volume=row.get("volume") or row.get("Volume"),
                adjusted_close=row.get("adj_close") or row.get("Adj Close") or row.get("close"),
            ))
            inserted += 1
    db.commit()
    return inserted


def get_prices_by_stock_id(db: Session, stock_id: int, limit: int = 500) -> list[HistoricalPrice]:
    return (
        db.query(HistoricalPrice)
        .filter(HistoricalPrice.stock_id == stock_id)
        .order_by(HistoricalPrice.timestamp)
        .limit(limit)
        .all()
    )


# ---------------------------------------------------------------------------
# Stock Metrics
# ---------------------------------------------------------------------------

def upsert_metric(db: Session, stock_id: int, period: str, metrics: dict) -> StockMetric:
    obj = (
        db.query(StockMetric)
        .filter(StockMetric.stock_id == stock_id, StockMetric.period == period)
        .first()
    )
    if obj is None:
        obj = StockMetric(stock_id=stock_id, period=period)
        db.add(obj)
    obj.return_pct       = metrics.get("return_pct")
    obj.risk_pct         = metrics.get("risk_pct")
    obj.avg_volume       = metrics.get("avg_volume")
    obj.sharpe_like_score = metrics.get("sharpe")
    obj.cagr             = metrics.get("cagr_pct") or metrics.get("cagr")
    obj.max_drawdown     = metrics.get("max_drawdown_pct") or metrics.get("max_drawdown")
    obj.trading_days     = metrics.get("trading_days")
    obj.calculated_at    = datetime.utcnow()
    db.commit()
    db.refresh(obj)
    return obj


# ---------------------------------------------------------------------------
# Portfolio Scores
# ---------------------------------------------------------------------------

def bulk_insert_portfolio_scores(db: Session, analysis_id: str, portfolios: list[dict]) -> None:
    # Clear previous scores for this analysis
    db.query(PortfolioScore).filter(PortfolioScore.analysis_id == analysis_id).delete()
    for p in portfolios:
        port_list = p.get("portfolio", [])
        port_id = "-".join(port_list)
        obj = PortfolioScore(
            portfolio_id=port_id,
            analysis_id=analysis_id,
            dm_score=p.get("dm_score"),
            return_score=p.get("return_score"),
            risk_score=p.get("risk_score"),
            diversification_score=p.get("diversification_score"),
            group_score=p.get("group_score"),
            dominance_score=p.get("dominance_score"),
            raw_return=p.get("raw_return"),
            raw_risk=p.get("raw_risk"),
            avg_correlation=p.get("avg_correlation"),
            rank=p.get("rank"),
        )
        db.add(obj)
    db.commit()


def get_top_portfolio_scores(db: Session, analysis_id: str, n: int = 10) -> list[PortfolioScore]:
    return (
        db.query(PortfolioScore)
        .filter(PortfolioScore.analysis_id == analysis_id)
        .order_by(PortfolioScore.rank)
        .limit(n)
        .all()
    )
