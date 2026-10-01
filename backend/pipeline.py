"""
TradeSense V2.1 — Full Analysis Pipeline Orchestrator
=====================================================
Runs all analysis stages in sequence, wrapping each in try/except
so that a failure in one stage never aborts the entire run.

Supports 60+ stocks across 12 NSE sectors.
"""

from __future__ import annotations

import csv
import logging
import uuid
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from backend.schemas import AnalyzeRequest

logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path(__file__).parent.parent
UNIVERSE_CSV  = PROJECT_ROOT / "data" / "stock_universe.csv"

_PERIOD_DAYS: dict[str, int] = {
    "1m": 30, "3m": 90, "6m": 180,
    "1y": 365, "2y": 730, "3y": 1095, "5y": 1825,
}


def _period_to_dates(period: str) -> tuple[str, str]:
    days = _PERIOD_DAYS.get(period, 365)
    end_dt   = datetime.utcnow()
    start_dt = end_dt - timedelta(days=days)
    return start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d")


def _load_universe(sector: str) -> list[dict]:
    rows: list[dict] = []
    if not UNIVERSE_CSV.exists():
        logger.error("stock_universe.csv not found at %s", UNIVERSE_CSV)
        return rows
    with UNIVERSE_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sym = row.get("symbol", "").strip()
            if not sym or not sym.endswith(".NS"):
                continue
            if sector == "All Sectors" or row.get("sector", "") == sector:
                rows.append({
                    "symbol":       sym,
                    "company_name": row.get("company_name", sym).strip(),
                    "sector":       row.get("sector", "Unknown").strip(),
                    "industry":     row.get("industry", "").strip(),
                    "exchange":     row.get("exchange", "NSE").strip(),
                })
    logger.info("Loaded %d stocks from universe (sector=%s)", len(rows), sector)
    return rows


def _get_all_sectors() -> list[dict]:
    """Return list of all sectors with their stock counts from the universe CSV."""
    if not UNIVERSE_CSV.exists():
        return []
    sector_counts: dict[str, int] = defaultdict(int)
    with UNIVERSE_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sym = row.get("symbol", "").strip()
            sec = row.get("sector", "").strip()
            if sym and sym.endswith(".NS") and sec:
                sector_counts[sec] += 1
    return [{"sector": s, "stock_count": c} for s, c in sorted(sector_counts.items())]


def _compute_data_quality(valid_count: int, total_count: int,
                          missing_data_pct: float = 0.0) -> dict:
    """Compute data quality metrics."""
    if total_count == 0:
        return {"level": "UNAVAILABLE", "score": 0, "valid_stocks": 0,
                "total_stocks": 0, "coverage_pct": 0}
    coverage = (valid_count / total_count) * 100
    if coverage >= 90 and missing_data_pct < 5:
        level = "HIGH"
    elif coverage >= 70:
        level = "MEDIUM"
    else:
        level = "LOW"
    return {
        "level": level,
        "score": round(coverage, 1),
        "valid_stocks": valid_count,
        "total_stocks": total_count,
        "coverage_pct": round(coverage, 1),
    }


def _compute_sector_analysis(metrics_list: list[dict], corr_df: pd.DataFrame,
                             adj: dict, threshold: float) -> list[dict]:
    """Compute per-sector analytics from the analyzed stock metrics."""
    sector_stocks: dict[str, list[dict]] = defaultdict(list)
    for m in metrics_list:
        sec = m.get("sector", "Unknown")
        sector_stocks[sec].append(m)

    sector_results = []
    for sector_name, stocks in sorted(sector_stocks.items()):
        returns = [s.get("return_pct", 0) or 0 for s in stocks]
        risks = [s.get("risk_pct", 0) or 0 for s in stocks]

        # Average pairwise correlation within sector
        syms = [s["symbol"] for s in stocks]
        corr_vals = []
        if corr_df is not None and not corr_df.empty:
            for i in range(len(syms)):
                for j in range(i + 1, len(syms)):
                    if syms[i] in corr_df.columns and syms[j] in corr_df.columns:
                        val = float(corr_df.loc[syms[i], syms[j]])
                        if not pd.isna(val):
                            corr_vals.append(val)

        # Sector network density
        sector_edges = 0
        if adj:
            for sym in syms:
                for nb, _ in adj.get(sym, []):
                    if nb in syms and sym < nb:
                        sector_edges += 1
        n_s = len(syms)
        sector_density = (2 * sector_edges / (n_s * (n_s - 1))) if n_s >= 2 else 0

        avg_ret = sum(returns) / len(returns) if returns else 0
        avg_risk = sum(risks) / len(risks) if risks else 0
        avg_corr = sum(corr_vals) / len(corr_vals) if corr_vals else 0

        # Simple sector score (diversification-aware)
        ret_component = min(avg_ret / 50 * 100, 100) if avg_ret > 0 else 0
        risk_component = max(0, 100 - avg_risk * 2) if avg_risk > 0 else 50
        div_component = max(0, 100 * (1 - avg_corr)) if avg_corr >= 0 else 50
        sector_score = round(0.4 * ret_component + 0.3 * risk_component + 0.3 * div_component, 1)

        sector_results.append({
            "sector": sector_name,
            "stocks_analyzed": len(stocks),
            "avg_return": round(avg_ret, 2),
            "avg_risk": round(avg_risk, 2),
            "avg_correlation": round(avg_corr, 4),
            "network_density": round(sector_density, 4),
            "sector_edges": sector_edges,
            "model_score": max(0, min(100, sector_score)),
            "stock_symbols": syms,
        })

    return sector_results


def _generate_network_conclusion(stats: dict, strong_rels: list, components: list,
                                 metrics_list: list) -> dict:
    """Generate dynamic plain-English network interpretation."""
    v = stats.get("num_vertices", 0)
    e = stats.get("num_edges", 0)
    density = stats.get("density", 0)
    most_conn = stats.get("most_connected_stock", "N/A")
    num_comp = stats.get("connected_components", 0)

    density_pct = round(density * 100, 1)

    if density_pct > 60:
        density_desc = "very high"
        density_insight = "Many stocks in this universe show strong co-movement. Holding multiple stocks from this set may provide less diversification than expected."
    elif density_pct > 30:
        density_desc = "moderate"
        density_insight = "There is a reasonable mix of connected and independent stocks, suggesting some diversification opportunities exist."
    elif density_pct > 10:
        density_desc = "low"
        density_insight = "Most stocks have weak relationships at the current threshold. This suggests good diversification potential across the universe."
    else:
        density_desc = "very low"
        density_insight = "Very few strong connections exist between stocks at the current threshold. The universe appears well-diversified."

    conclusion_text = f"The market network contains {v} stocks with {e} strong connections (density: {density_pct}%). {density_insight}"
    if most_conn and most_conn != "N/A":
        mc_display = most_conn.replace(".NS", "")
        conclusion_text += f" {mc_display} is the most connected stock in the network."
    if num_comp > 1:
        conclusion_text += f" There are {num_comp} distinct groups of connected stocks."

    return {
        "summary": conclusion_text,
        "density_level": density_desc,
        "density_pct": density_pct,
        "stocks_analyzed": v,
        "connections": e,
        "connected_groups": num_comp,
        "most_connected": most_conn,
    }


class AnalysisPipeline:
    """Orchestrates the full TradeSense V2.1 analysis pipeline."""

    def __init__(self, config: AnalyzeRequest) -> None:
        self.config      = config
        self.analysis_id = str(uuid.uuid4())

    def run(self) -> dict[str, Any]:
        cfg    = self.config
        result: dict[str, Any] = {
            "analysis_id":   self.analysis_id,
            "data_mode":     cfg.data_mode,
            "sector":        cfg.sector,
            "period":        cfg.period,
            "corr_threshold": cfg.corr_threshold,
            "created_at":    datetime.utcnow().isoformat(),
        }

        # ── Stage 1: Provider ──────────────────────────────────────────────
        provider = None
        try:
            if cfg.data_mode == "DEMO":
                from backend.providers.demo_provider import DemoDataProvider
                provider = DemoDataProvider()
            else:
                from backend.providers.yfinance_provider import YFinanceProvider
                provider = YFinanceProvider()
            result["provider_label"] = provider.data_mode_label
            logger.info("[Stage 1] Provider: %s", provider.data_mode_label)
        except Exception as exc:
            logger.error("[Stage 1] Provider init failed: %s", exc)

        # ── Stage 2: Stock universe ────────────────────────────────────────
        universe: list[dict] = []
        try:
            universe = _load_universe(cfg.sector)
        except Exception as exc:
            logger.error("[Stage 2] Universe load failed: %s", exc)

        # Sector info
        all_sectors = _get_all_sectors()
        result["all_sectors"] = all_sectors
        result["sector_count"] = len(all_sectors)

        # In demo mode, use demo provider's built-in symbol list
        if cfg.data_mode == "DEMO" and provider:
            symbols = provider.supported_tickers
        else:
            # NO CAP — use all stocks from the universe for the selected sector
            symbols = [r["symbol"] for r in universe]

        symbol_meta: dict[str, dict] = {r["symbol"]: r for r in universe}
        result["tickers"] = symbols
        result["universe_total"] = len(symbols)

        # ── Stage 3: Historical price data ────────────────────────────────
        price_data: dict[str, pd.DataFrame] = {}
        start_date, end_date = _period_to_dates(cfg.period)
        failed_symbols: list[str] = []
        if provider:
            for sym in symbols:
                try:
                    df = provider.get_historical_data(sym, start_date, end_date, "1d")
                    if not df.empty:
                        price_data[sym] = df
                except Exception as sym_exc:
                    failed_symbols.append(sym)
                    logger.warning("[Stage 3] No data for %s: %s", sym, sym_exc)
        valid_symbols = list(price_data.keys())
        result["tickers"] = valid_symbols
        result["failed_symbols"] = failed_symbols
        logger.info("[Stage 3] Price data: %d stocks (%d failed)", len(valid_symbols), len(failed_symbols))

        # Data quality
        result["data_quality"] = _compute_data_quality(
            len(valid_symbols), len(symbols),
            missing_data_pct=len(failed_symbols) / max(len(symbols), 1) * 100
        )

        # Save price history for /stock/{symbol}/history endpoint
        price_history: dict[str, list] = {}
        for sym, df in price_data.items():
            col = "Close" if "Close" in df.columns else df.columns[0]
            price_history[sym] = [
                {"date": str(idx.date()), "close": round(float(row[col]), 2)}
                for idx, row in df.iterrows()
                if not pd.isna(row[col])
            ]
        result["price_history"] = price_history

        # ── Stage 4: Financial metrics ─────────────────────────────────────
        metrics_list: list[dict] = []
        try:
            from backend.core.metrics import calculate_stock_metrics
            for sym, df in price_data.items():
                m = calculate_stock_metrics(sym, df)
                # Normalise key names
                m["return_pct"]  = m.get("period_return_pct", m.get("return_pct", 0.0))
                m["risk_pct"]    = m.get("annualized_risk_pct", m.get("risk_pct", 0.0))
                m["sharpe"]      = m.get("sharpe_ratio", m.get("sharpe", 0.0))
                if "Volume" in df.columns:
                    m["avg_volume"] = float(df["Volume"].dropna().mean())
                else:
                    m["avg_volume"] = 0.0
                # Enrich with company info
                info = symbol_meta.get(sym, {})
                m["company_name"] = info.get("company_name", sym)
                m["sector"]       = info.get("sector", "Unknown")
                m["data_mode"]    = cfg.data_mode
                col = "Close" if "Close" in df.columns else df.columns[0]
                m["latest_price"] = round(float(df[col].dropna().iloc[-1]), 2)
                metrics_list.append(m)
            logger.info("[Stage 4] Metrics: %d stocks", len(metrics_list))
        except Exception as exc:
            logger.error("[Stage 4] Metrics failed: %s", exc)

        result["stocks_analyzed"] = len(metrics_list)
        result["stocks"]          = metrics_list

        # ── Stage 5: Returns matrix + Pearson correlation ─────────────────
        returns_df = pd.DataFrame()
        corr_df    = pd.DataFrame()
        try:
            from backend.core.matrix import build_returns_matrix, calculate_pearson_correlation, validate_matrix_properties, get_correlation_stats
            returns_df = build_returns_matrix(price_data)
            corr_df    = calculate_pearson_correlation(returns_df)
            validation = validate_matrix_properties(corr_df)
            stats      = get_correlation_stats(corr_df)
            result["correlation_matrix"]    = corr_df.to_dict()
            result["correlation_matrix_df"] = corr_df
            result["returns_df"]            = returns_df
            result["matrix_validation"]     = validation
            result["correlation_stats"]     = stats
            logger.info("[Stage 5] Correlation matrix: %dx%d", len(corr_df), len(corr_df))
        except Exception as exc:
            logger.error("[Stage 5] Correlation failed: %s", exc)

        # ── Stage 6: Adjacency dict ────────────────────────────────────────
        adj: dict[str, list[tuple[str, float]]] = {}
        try:
            from backend.core.graph import build_adjacency_dict
            if not corr_df.empty:
                adj = build_adjacency_dict(valid_symbols, corr_df, cfg.corr_threshold)
            logger.info("[Stage 6] Adjacency built, threshold=%.2f", cfg.corr_threshold)
        except Exception as exc:
            logger.error("[Stage 6] Adjacency build failed: %s", exc)

        # ── Stage 7: BFS + DFS ────────────────────────────────────────────
        try:
            from backend.core.graph import bfs, dfs, all_degrees
            if adj:
                degrees   = all_degrees(adj)
                start     = max(degrees, key=lambda n: degrees[n]) if degrees else (valid_symbols[0] if valid_symbols else "")
                bfs_order, bfs_trace = bfs(adj, start)
                dfs_order, dfs_trace = dfs(adj, start)
                result["bfs"] = {"start": start, "order": bfs_order, "trace": bfs_trace}
                result["dfs"] = {"start": start, "order": dfs_order, "trace": dfs_trace}
                logger.info("[Stage 7] BFS/DFS from %s", start)
        except Exception as exc:
            logger.error("[Stage 7] BFS/DFS failed: %s", exc)

        # ── Stage 8: Connected components + graph stats ────────────────────
        graph_stats = {}
        components = []
        strong_rels = []
        try:
            from backend.core.graph import connected_components_from_scratch, calculate_graph_statistics, get_strong_relationships
            if adj:
                components = connected_components_from_scratch(adj)
                graph_stats = calculate_graph_statistics(adj, valid_symbols)
                strong_rels = get_strong_relationships(adj, top_n=10)
                result["graph_stats"]         = graph_stats
                result["connected_components"] = components
                result["strong_relationships"] = strong_rels
                logger.info("[Stage 8] Components: %d", len(components))
        except Exception as exc:
            logger.error("[Stage 8] Components failed: %s", exc)
            graph_stats = {
                "num_vertices": len(valid_symbols), "num_edges": 0,
                "avg_degree": 0.0, "max_degree": 0, "density": 0.0,
                "connected_components": 0, "most_connected_stock": "",
            }
            result.setdefault("graph_stats", graph_stats)

        # Network conclusion
        result["network_conclusion"] = _generate_network_conclusion(
            graph_stats, strong_rels, components, metrics_list
        )

        # ── Stage 9: Welsh-Powell coloring ────────────────────────────────
        coloring_dict: dict[str, int] = {}
        color_groups:  dict[int, list[str]] = {}
        try:
            from backend.core.coloring import welsh_powell, validate_coloring, get_color_groups, verify_independent_sets, exact_chromatic_number
            if adj:
                coloring_dict, step_trace = welsh_powell(adj)
                is_valid     = validate_coloring(adj, coloring_dict)
                color_groups = get_color_groups(coloring_dict)
                ind_sets     = verify_independent_sets(adj, color_groups)
                num_colors   = len(color_groups)
                chrom_info   = exact_chromatic_number(adj)
                result["coloring_dict"]          = coloring_dict
                result["color_groups"]           = {str(k): v for k, v in color_groups.items()}
                result["num_color_groups"]       = num_colors
                result["coloring_valid"]         = is_valid
                result["coloring_step_trace"]    = step_trace
                result["chromatic_info"]         = chrom_info
                result["independent_sets_verified"] = {str(k): v for k, v in ind_sets.items()}
                logger.info("[Stage 9] Coloring: %d colors, valid=%s", num_colors, is_valid)
        except Exception as exc:
            logger.error("[Stage 9] Coloring failed: %s", exc)
            result.setdefault("num_color_groups", 0)

        # ── Stage 10: Dominance relation + poset ──────────────────────────
        dominance_pairs: list[tuple[str, str]] = []
        try:
            from backend.core.relations import build_dominance_relation, build_relation_matrix, analyze_poset, get_non_dominated, get_incomparable_pairs, get_most_dominant
            if metrics_list:
                dominance_pairs = build_dominance_relation(metrics_list)
                rel_matrix      = build_relation_matrix(valid_symbols, dominance_pairs)
                poset_props     = analyze_poset(dominance_pairs, valid_symbols)
                non_dom         = get_non_dominated(dominance_pairs, valid_symbols)
                incomparable    = get_incomparable_pairs(dominance_pairs, valid_symbols)
                most_dom        = get_most_dominant(dominance_pairs, valid_symbols)
                result["dominance_pairs"]       = [{"dominator": a, "dominated": b} for a, b in dominance_pairs]
                result["relation_matrix"]       = rel_matrix
                result["poset_properties"]      = poset_props
                result["non_dominated_stocks"]  = non_dom
                result["non_dominated_count"]   = len(non_dom)
                result["incomparable_pairs"]    = [{"a": a, "b": b} for a, b in incomparable]
                result["most_dominant_stocks"]  = most_dom
                logger.info("[Stage 10] Dominance pairs: %d", len(dominance_pairs))
        except Exception as exc:
            logger.error("[Stage 10] Relations failed: %s", exc)
            result.setdefault("non_dominated_count", 0)

        non_dominated_set = set(result.get("non_dominated_stocks", []))

        # ── Stage 11: Hasse diagram ───────────────────────────────────────
        try:
            from backend.core.hasse import build_cover_relation, analyze_hasse, get_hasse_layout
            if dominance_pairs:
                cover_pairs = build_cover_relation(dominance_pairs)
                hasse_info  = analyze_hasse(cover_pairs, valid_symbols)
                hasse_layout = get_hasse_layout(cover_pairs, valid_symbols)
                result["hasse_cover_relation"] = [{"from": a, "to": b} for a, b in cover_pairs]
                result["hasse_levels"]   = hasse_info.get("levels", {})
                result["hasse_maximal"]  = hasse_info.get("maximal_elements", [])
                result["hasse_minimal"]  = hasse_info.get("minimal_elements", [])
                result["hasse_layout"]   = hasse_layout
                logger.info("[Stage 11] Cover relation: %d edges", len(cover_pairs))
        except Exception as exc:
            logger.error("[Stage 11] Hasse failed: %s", exc)

        # ── Stage 12: Set theory ──────────────────────────────────────────
        try:
            from backend.core.sets import define_high_return_set, define_low_risk_set, define_high_volume_set, compute_all_set_operations
            if metrics_list:
                A = define_high_return_set(metrics_list, cfg.return_threshold)
                B = define_high_volume_set(metrics_list)
                C = define_low_risk_set(metrics_list, cfg.risk_threshold)
                set_ops = compute_all_set_operations(A, B, C)
                result["set_operations"] = set_ops
                logger.info("[Stage 12] Sets: |A|=%d |B|=%d |C|=%d", len(A), len(B), len(C))
        except Exception as exc:
            logger.error("[Stage 12] Set operations failed: %s", exc)

        # ── Stage 13: Boolean truth table ─────────────────────────────────
        try:
            from backend.core.boolean_logic import build_truth_table
            if metrics_list:
                bool_table = build_truth_table(metrics_list, cfg.return_threshold, cfg.risk_threshold)
                result["boolean_table"] = bool_table
                logger.info("[Stage 13] Boolean table: %d rows, %d qualified",
                            bool_table["total_count"], bool_table["qualified_count"])
        except Exception as exc:
            logger.error("[Stage 13] Boolean table failed: %s", exc)

        # ── Stage 14: Portfolio candidates C(n,k) ─────────────────────────
        candidates: list[tuple[str, ...]] = []
        total_possible = 0
        try:
            from backend.core.combinatorics import select_candidate_pool, generate_combinations, combination_count, format_combinatorics_display
            if metrics_list:
                # Dynamically adjust pool size: for large universes use smart filtering
                n_stocks = len(metrics_list)
                if n_stocks > 30:
                    pool_size = 20  # Top 20 by Sharpe for large universes
                    max_combos = 3000
                elif n_stocks > 15:
                    pool_size = 15
                    max_combos = 2000
                else:
                    pool_size = n_stocks
                    max_combos = 2000

                pool           = select_candidate_pool(metrics_list, pool_size=pool_size)
                candidates, total_possible = generate_combinations(pool, cfg.portfolio_k, max_combinations=max_combos)
                combo_display  = format_combinatorics_display(len(pool), cfg.portfolio_k, len(candidates), total_possible)
                result["candidate_pool"]            = pool
                result["candidate_pool_size"]        = len(pool)
                result["combinatorics"]             = combo_display
                result["total_portfolios_evaluated"] = len(candidates)
                logger.info("[Stage 14] C(%d,%d): %d candidates (pool=%d)",
                            len(pool), cfg.portfolio_k, len(candidates), pool_size)
        except Exception as exc:
            logger.error("[Stage 14] Combinatorics failed: %s", exc)
            result.setdefault("total_portfolios_evaluated", 0)

        # ── Stage 15: DM Score ranking ────────────────────────────────────
        try:
            from backend.core.scoring import run_scoring_pipeline, DISCLAIMER, DM_WEIGHTS
            if candidates and metrics_list and not returns_df.empty:
                scoring_out = run_scoring_pipeline(
                    candidates      = candidates,
                    metrics         = metrics_list,
                    returns_df      = returns_df,
                    corr_df         = corr_df,
                    coloring        = coloring_dict,
                    non_dominated   = non_dominated_set,
                    top_n           = 10,
                )
                result["top_portfolios"]  = scoring_out["top_portfolios"]
                result["all_portfolios"]  = scoring_out["all_evaluated"]
                result["dm_weights"]      = DM_WEIGHTS
                result["disclaimer"]      = DISCLAIMER
                top_score = scoring_out["top_portfolios"][0]["dm_score"] if scoring_out["top_portfolios"] else 0.0
                result["top_dm_score"] = top_score
                logger.info("[Stage 15] Scored %d portfolios; top DM=%.2f", len(scoring_out["all_evaluated"]), top_score)
        except Exception as exc:
            logger.error("[Stage 15] Scoring failed: %s", exc)
            result.setdefault("top_portfolios", [])
            result.setdefault("all_portfolios", [])
            result.setdefault("top_dm_score", 0.0)

        # ── Stage 16: Sector Analysis ─────────────────────────────────────
        try:
            sector_analysis = _compute_sector_analysis(metrics_list, corr_df, adj, cfg.corr_threshold)
            result["sector_analysis"] = sector_analysis
            logger.info("[Stage 16] Sector analysis: %d sectors", len(sector_analysis))
        except Exception as exc:
            logger.error("[Stage 16] Sector analysis failed: %s", exc)
            result.setdefault("sector_analysis", [])

        logger.info("Pipeline complete [%s]: %d stocks, %d portfolios scored",
                    self.analysis_id, len(metrics_list), len(result.get("all_portfolios", [])))
        return result
