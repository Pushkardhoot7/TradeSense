"""
TradeSense V2 — Pydantic v2 Request & Response Schemas
=======================================================

All API-level data contracts.  Schemas are intentionally kept flat
(no deep nesting beyond what is needed) to simplify JSON serialisation
and JavaScript consumption.
"""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    """Parameters that control a full analysis pipeline run."""

    sector: str = Field(
        default="All Sectors",
        description="Sector filter — pass 'All Sectors' to include every stock.",
    )
    period: str = Field(
        default="1y",
        description="Historical look-back period (e.g. '1y', '6m', '3m', '2y').",
    )
    corr_threshold: float = Field(
        default=0.70,
        ge=0.5,
        le=0.9,
        description="Pearson correlation threshold for graph edge construction.",
    )
    portfolio_k: int = Field(
        default=3,
        ge=2,
        le=6,
        description="Portfolio size k for C(n,k) combinatorial generation.",
    )
    data_mode: str = Field(
        default="HISTORICAL",
        description="Data source mode: 'HISTORICAL' (yfinance), 'LIVE', 'STALE', 'UNAVAILABLE', or 'DEMO'.",
    )
    return_threshold: float = Field(
        default=5.0,
        description="Minimum annualised return (%) to classify a stock as high-return.",
    )
    risk_threshold: float = Field(
        default=20.0,
        description="Maximum annualised volatility (%) to classify a stock as low-risk.",
    )

    @field_validator("data_mode")
    @classmethod
    def validate_data_mode(cls, v: str) -> str:
        allowed = {"HISTORICAL", "LIVE", "STALE", "UNAVAILABLE", "DEMO"}
        if v.upper() not in allowed:
            raise ValueError(f"data_mode must be one of {allowed}; got '{v}'.")
        return v.upper()

    @field_validator("period")
    @classmethod
    def validate_period(cls, v: str) -> str:
        allowed = {"1m", "3m", "6m", "1y", "2y", "3y", "5y"}
        if v.lower() not in allowed:
            raise ValueError(f"period must be one of {allowed}; got '{v}'.")
        return v.lower()


# ---------------------------------------------------------------------------
# Core data schemas
# ---------------------------------------------------------------------------

class StockMetricSchema(BaseModel):
    """Financial metrics for a single stock computed over the analysis period."""

    symbol: str
    company_name: str
    sector: str
    latest_price: float
    return_pct: float = Field(description="Annualised return %")
    risk_pct: float = Field(description="Annualised volatility %")
    avg_volume: float
    cagr: float = Field(description="Compound Annual Growth Rate %")
    sharpe: float = Field(description="Sharpe-like ratio (return/risk)")
    max_drawdown: float = Field(description="Maximum drawdown %")
    trading_days: int
    data_mode: str


class CorrelationPairSchema(BaseModel):
    """A single pairwise Pearson correlation between two stocks."""

    stock_a: str
    stock_b: str
    correlation: float


# ---------------------------------------------------------------------------
# Graph schemas
# ---------------------------------------------------------------------------

class GraphStatsSchema(BaseModel):
    """Summary statistics for the correlation graph G=(V,E)."""

    num_vertices: int
    num_edges: int
    avg_degree: float
    max_degree: int
    density: float = Field(description="Edge density = 2|E| / (|V|*(|V|-1))")
    num_connected_components: int
    most_connected_stock: str


class BFSResultSchema(BaseModel):
    """Result of a Breadth-First Search traversal."""

    start_node: str
    traversal_order: list[str]
    step_trace: list[str] = Field(
        description="Human-readable step-by-step trace of the BFS queue."
    )


class DFSResultSchema(BaseModel):
    """Result of a Depth-First Search traversal."""

    start_node: str
    traversal_order: list[str]
    step_trace: list[str] = Field(
        description="Human-readable step-by-step trace of the DFS stack."
    )


# ---------------------------------------------------------------------------
# Relations / Poset schemas
# ---------------------------------------------------------------------------

class RelationPairSchema(BaseModel):
    """A single dominance pair (A ≽ B)."""

    dominator: str
    dominated: str


class PosetPropertiesSchema(BaseModel):
    """Verification of partial-order axioms on the dominance relation."""

    is_reflexive: bool
    is_antisymmetric: bool
    is_transitive: bool
    is_partial_order: bool
    reflexive_note: str
    antisymmetric_note: str
    transitive_note: str


class HasseLevelSchema(BaseModel):
    """A single node in the Hasse diagram with its layout coordinates."""

    node: str
    level: int
    x_pos: float
    y_pos: float


# ---------------------------------------------------------------------------
# Graph coloring schemas
# ---------------------------------------------------------------------------

class ColorGroupSchema(BaseModel):
    """One color group (independent set) produced by Welsh-Powell."""

    color_id: int
    stocks: list[str]
    size: int
    is_valid_independent_set: bool


class ColoringResultSchema(BaseModel):
    """Full Welsh-Powell graph coloring result."""

    coloring_dict: dict[str, int] = Field(
        description="Maps each stock symbol to its color integer."
    )
    color_groups: list[ColorGroupSchema]
    num_colors: int
    chromatic_info: str = Field(
        description="Human-readable explanation of the chromatic number."
    )
    is_valid: bool = Field(description="True if no two adjacent nodes share a color.")
    step_trace: list[str] = Field(
        description="Step-by-step log of the Welsh-Powell algorithm."
    )


# ---------------------------------------------------------------------------
# Set theory schemas
# ---------------------------------------------------------------------------

class SetOperationsSchema(BaseModel):
    """All set-theory operations on sets A (High Return), B (High Volume), C (Low Risk)."""

    A: list[str]
    A_size: int
    B: list[str]
    B_size: int
    C: list[str]
    C_size: int
    A_union_B: list[str]
    A_union_B_size: int
    A_inter_B: list[str]
    A_inter_B_size: int
    A_inter_C: list[str]
    A_inter_C_size: int
    B_inter_C: list[str]
    B_inter_C_size: int
    A_minus_B: list[str]
    A_minus_B_size: int
    A_union_B_union_C: list[str]
    A_union_B_union_C_size: int


# ---------------------------------------------------------------------------
# Boolean logic schemas
# ---------------------------------------------------------------------------

class BooleanRowSchema(BaseModel):
    """One row of the boolean screening truth table."""

    symbol: str
    return_condition: bool = Field(description="return_pct >= return_threshold")
    risk_condition: bool = Field(description="risk_pct <= risk_threshold")
    volume_condition: bool = Field(description="avg_volume >= volume_threshold")
    final_result: bool = Field(description="return_condition AND risk_condition AND volume_condition")
    return_value: float
    risk_value: float
    volume_value: float


# ---------------------------------------------------------------------------
# Portfolio schemas
# ---------------------------------------------------------------------------

class PortfolioSchema(BaseModel):
    """A ranked portfolio candidate with DM score components."""

    rank: int
    portfolio: list[str]
    dm_score: float
    return_score: float
    risk_score: float
    diversification_score: float
    group_score: float
    dominance_score: float
    raw_return: float = Field(description="Unweighted average return %")
    raw_risk: float = Field(description="Unweighted average risk %")
    avg_correlation: float = Field(description="Average pairwise Pearson correlation")


# ---------------------------------------------------------------------------
# Composite analysis result schema
# ---------------------------------------------------------------------------

class AnalysisResultSchema(BaseModel):
    """Complete output of one AnalysisPipeline.run() invocation."""

    analysis_id: str
    data_mode: str
    sector: str
    period: str
    corr_threshold: float
    stocks_analyzed: int
    stocks: list[StockMetricSchema]
    correlation_matrix: dict[str, dict[str, float]]
    graph_stats: GraphStatsSchema
    bfs: BFSResultSchema | None = None
    dfs: DFSResultSchema | None = None
    poset: PosetPropertiesSchema
    hasse_nodes: list[HasseLevelSchema] = Field(default_factory=list)
    coloring: ColoringResultSchema
    sets: SetOperationsSchema
    boolean_table: list[BooleanRowSchema]
    top_portfolios: list[PortfolioSchema]
    total_portfolios_evaluated: int
    top_dm_score: float
    created_at: str


# ---------------------------------------------------------------------------
# Misc API response schemas
# ---------------------------------------------------------------------------

class MarketStatusSchema(BaseModel):
    """Current market open/closed status."""

    status: str = Field(description="'OPEN', 'CLOSED', or 'UNKNOWN'")
    message: str
    timestamp: str


class HealthSchema(BaseModel):
    """API health check response."""

    status: str
    version: str


class PortfolioCompareRequest(BaseModel):
    """Request body for side-by-side portfolio comparison."""

    portfolio_a: list[str] = Field(min_length=1)
    portfolio_b: list[str] = Field(min_length=1)


class PortfolioCompareResult(BaseModel):
    """Side-by-side comparison of two portfolios."""

    portfolio_a: list[str]
    portfolio_b: list[str]
    metrics_a: dict[str, Any]
    metrics_b: dict[str, Any]
    winner: str
    summary: str
