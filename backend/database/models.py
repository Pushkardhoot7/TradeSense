"""
TradeSense V2 — SQLAlchemy ORM Models

All 10 database tables per the V2 schema specification.
Designed for future PostgreSQL migration — all types are portable.
"""

from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Boolean,
    Text, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from backend.database.db import Base


class Stock(Base):
    """Master stock registry — one row per unique stock symbol."""
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    company_name = Column(String(200), nullable=False)
    sector = Column(String(100), nullable=False)
    industry = Column(String(200))
    exchange = Column(String(20), default="NSE")
    market_cap = Column(Float)  # In Crores INR — populated when available
    last_updated = Column(DateTime, default=datetime.utcnow)

    # Relationships
    historical_prices = relationship("HistoricalPrice", back_populates="stock", cascade="all, delete-orphan")
    quotes = relationship("Quote", back_populates="stock", cascade="all, delete-orphan")
    metrics = relationship("StockMetric", back_populates="stock", cascade="all, delete-orphan")


class HistoricalPrice(Base):
    """Daily OHLCV price bars per stock per trading day."""
    __tablename__ = "historical_prices"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Float)
    adjusted_close = Column(Float)

    stock = relationship("Stock", back_populates="historical_prices")

    __table_args__ = (
        UniqueConstraint("stock_id", "timestamp", name="uq_stock_date"),
        Index("ix_hist_stock_timestamp", "stock_id", "timestamp"),
    )


class Quote(Base):
    """Latest market quote per stock (live or most recent close)."""
    __tablename__ = "quotes"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    price = Column(Float, nullable=False)
    change = Column(Float)          # Absolute price change
    change_percent = Column(Float)  # Percentage change
    volume = Column(Float)
    market_status = Column(String(20), default="UNKNOWN")  # OPEN | CLOSED | PRE-MARKET | UNKNOWN

    stock = relationship("Stock", back_populates="quotes")


class StockMetric(Base):
    """Computed financial metrics per stock per analysis period."""
    __tablename__ = "stock_metrics"

    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False, index=True)
    period = Column(String(20), nullable=False)     # e.g. "1y", "6m", "3m"
    return_pct = Column(Float)
    risk_pct = Column(Float)
    avg_volume = Column(Float)
    sharpe_like_score = Column(Float)
    cagr = Column(Float)
    max_drawdown = Column(Float)
    trading_days = Column(Integer)
    calculated_at = Column(DateTime, default=datetime.utcnow)

    stock = relationship("Stock", back_populates="metrics")

    __table_args__ = (
        UniqueConstraint("stock_id", "period", name="uq_metric_period"),
    )


class Correlation(Base):
    """Pairwise Pearson correlation between two stocks for a given period."""
    __tablename__ = "correlations"

    id = Column(Integer, primary_key=True, index=True)
    stock_a = Column(String(20), nullable=False, index=True)
    stock_b = Column(String(20), nullable=False, index=True)
    correlation = Column(Float, nullable=False)
    period = Column(String(20), nullable=False)
    calculated_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        UniqueConstraint("stock_a", "stock_b", "period", name="uq_corr_pair_period"),
        Index("ix_corr_period", "period"),
    )


class GraphEdge(Base):
    """Edges in the correlation graph G=(V,E) per analysis run."""
    __tablename__ = "graph_edges"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(20), nullable=False)
    target = Column(String(20), nullable=False)
    weight = Column(Float, nullable=False)   # Pearson correlation
    threshold = Column(Float, nullable=False)
    analysis_id = Column(String(50), index=True)

    __table_args__ = (
        Index("ix_graph_analysis", "analysis_id"),
    )


class DominanceRelation(Base):
    """Stored dominance relation pairs (A ≽ B) for a given analysis."""
    __tablename__ = "relations"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(20), nullable=False)      # Dominator A
    target = Column(String(20), nullable=False)      # Dominated B
    relation_type = Column(String(50), default="return_risk_dominance")
    analysis_id = Column(String(50), index=True)


class ColorAssignment(Base):
    """Welsh-Powell graph coloring assignment per stock per analysis."""
    __tablename__ = "color_assignments"

    id = Column(Integer, primary_key=True, index=True)
    stock = Column(String(20), nullable=False)
    color_group = Column(Integer, nullable=False)
    algorithm = Column(String(50), default="welsh_powell")
    analysis_id = Column(String(50), index=True)


class PortfolioCandidate(Base):
    """Generated portfolio candidate (list of k stocks)."""
    __tablename__ = "portfolio_candidates"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String(100), nullable=False, index=True)
    stocks = Column(Text, nullable=False)     # JSON-serialized list of symbols
    k_size = Column(Integer)
    analysis_id = Column(String(50), index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class PortfolioScore(Base):
    """DM Score and component scores for a portfolio candidate."""
    __tablename__ = "portfolio_scores"

    id = Column(Integer, primary_key=True, index=True)
    portfolio_id = Column(String(100), nullable=False, index=True)
    analysis_id = Column(String(50), index=True)
    return_score = Column(Float)
    risk_score = Column(Float)
    diversification_score = Column(Float)
    group_score = Column(Float)
    dominance_score = Column(Float)
    dm_score = Column(Float)
    raw_return = Column(Float)
    raw_risk = Column(Float)
    avg_correlation = Column(Float)
    rank = Column(Integer)
    calculated_at = Column(DateTime, default=datetime.utcnow)


class Analysis(Base):
    """Analysis session metadata — links all computed results."""
    __tablename__ = "analyses"

    id = Column(String(50), primary_key=True)
    sector = Column(String(100))
    period = Column(String(20))
    corr_threshold = Column(Float)
    portfolio_k = Column(Integer)
    data_mode = Column(String(20))     # DEMO | HISTORICAL | LIVE
    stocks_analyzed = Column(Integer)
    portfolios_evaluated = Column(Integer)
    top_dm_score = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    is_complete = Column(Boolean, default=False)
