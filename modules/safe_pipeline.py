"""
Safe Pipeline Execution Wrapper for TradeSense

Implements defensive error-handling around every major mathematical pipeline stage.
Prevents unhandled tracebacks or blank screen crashes.
Handles network failures, invalid tickers, missing data, empty sectors, and combinatorial caps gracefully.
"""

from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from modules.stock_selector import select_top_stocks_per_sector
from modules.data_loader import get_historical_data, fetch_aligned_close_prices, DATA_SOURCE_LIVE
from modules.financial_metrics import generate_stock_metrics_table
from modules.correlation import calculate_return_matrix, calculate_correlation_matrix, validate_matrix_properties, calculate_correlation_summary_stats
from modules.graph_analysis import build_correlation_graph, calculate_graph_statistics
from modules.relations import build_dominance_relation, get_non_dominated_stocks, get_incomparable_pairs
from modules.hasse import check_reflexive, check_antisymmetric, check_transitive, is_partial_order, build_cover_relation, build_hasse_graph, find_maximal_minimal_elements
from modules.coloring import color_graph, validate_coloring, get_color_groups, calculate_exact_chromatic_number_if_feasible
from modules.set_theory import define_high_return_set, define_low_risk_set, define_high_volume_set, set_intersection, set_union, set_difference, evaluate_screening_rule
from modules.portfolio import calculate_combination_count, select_candidate_pool, generate_candidate_portfolios
from modules.dm_scoring import run_dm_portfolio_scoring
from modules.top3_display import render_top3_portfolios
from modules.demo_data import get_demo_stock_universe, get_demo_historical_data_map, generate_demo_price_dataframe

MAX_EVALUATED_COMBINATIONS_CAP = 1000
MIN_REQUIRED_OBSERVATIONS = 20


def execute_safe_pipeline(inputs: dict, is_demo_mode: bool = True) -> dict:
    """
    Execute full pipeline defensively with error handling and fallback reporting.
    """
    pipeline_warnings = []
    skipped_tickers = []
    excluded_stocks_notes = []

    # Stage 1: Stock Selection & Sector Validation
    try:
        if is_demo_mode:
            all_demo_stocks = get_demo_stock_universe()
            target_sec = inputs.get("sector")
            if target_sec and target_sec != "All Sectors":
                filtered_stocks = [s for s in all_demo_stocks if s["sector"] == target_sec]
            else:
                filtered_stocks = all_demo_stocks

            if not filtered_stocks:
                return {
                    "error": f"No stocks found for sector '{target_sec}'. Please select 'All Sectors' or a different sector.",
                    "error_stage": "Stock Selection",
                    "error_type": "EMPTY_SECTOR",
                }

            selected_tickers = [s["symbol"] for s in filtered_stocks]
            selection_table = [
                {
                    "Sector": s["sector"],
                    "Symbol": s["symbol"],
                    "Company Name": s["company_name"],
                    "Industry": s.get("industry", "N/A"),
                    "Rank": s["rank"],
                    "Selection Summary": "Demo Universe Active",
                }
                for s in filtered_stocks
            ]
        else:
            target_sector = None if inputs.get("sector") == "All Sectors" else inputs.get("sector")
            selection_results = select_top_stocks_per_sector(target_sector)

            if not selection_results or all(res["selected_count"] == 0 for res in selection_results):
                return {
                    "error": f"No stocks found for '{inputs.get('sector', 'Selected Sector')}'. Please select 'All Sectors' or another sector.",
                    "error_stage": "Stock Selection",
                    "error_type": "EMPTY_SECTOR",
                }

            selected_tickers = []
            selection_table = []
            for res in selection_results:
                sec = res["sector"]
                avail_cnt = res["available_count"]
                sel_cnt = res["selected_count"]
                summary_str = f"{sel_cnt}/15 selected ({avail_cnt} eligible)" if avail_cnt <= 15 else "15/15 selected"
                for stock in res["stock_details"]:
                    selected_tickers.append(stock["symbol"])
                    selection_table.append({
                        "Sector": sec,
                        "Symbol": stock["symbol"],
                        "Company Name": stock["company_name"],
                        "Industry": stock.get("industry", "DATA NOT AVAILABLE"),
                        "Rank": stock["rank"],
                        "Selection Summary": summary_str,
                    })

    except Exception as e:
        return {
            "error": f"Failed during stock selection stage: {str(e)}",
            "error_stage": "Stock Selection",
            "error_type": "SELECTION_FAILED",
        }

    if len(selected_tickers) < 2:
        return {
            "error": "At least 2 valid stocks are required to run pipeline analysis.",
            "error_stage": "Stock Selection",
            "error_type": "INSUFFICIENT_STOCKS",
        }

    # Stage 2: Data Retrieval & Cleaning
    try:
        if is_demo_mode:
            price_df = generate_demo_price_dataframe(num_days=252)
            data_map = get_demo_historical_data_map(num_days=252)
            valid_cols = [c for c in selected_tickers if c in price_df.columns]
            price_df = price_df[valid_cols]
            data_map = {c: data_map[c] for c in valid_cols if c in data_map}
            fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_source_label = "DEMO DATA — Sample Data"
        else:
            data_map = {}
            valid_tickers = []
            for t in list(selected_tickers):
                try:
                    res = get_historical_data(
                        t,
                        period=inputs.get("period_code", "1y"),
                        interval=inputs.get("interval_code", "1d"),
                        allow_demo=False
                    )
                    if res["status"] == "failed" or res["data"].empty:
                        skipped_tickers.append(t)
                        pipeline_warnings.append(f"Skipped {t}: Unable to retrieve live market data from Yahoo Finance.")
                        continue

                    raw_data = res["data"]
                    missing_ratio = raw_data["Close"].isnull().mean() if "Close" in raw_data.columns else 0.0
                    if missing_ratio > 0.10:
                        skipped_tickers.append(t)
                        excluded_stocks_notes.append(f"Excluded {t}: High missing data ratio ({missing_ratio:.1%}).")
                        continue

                    if len(raw_data) < MIN_REQUIRED_OBSERVATIONS:
                        skipped_tickers.append(t)
                        excluded_stocks_notes.append(f"Excluded {t}: Insufficient trading days ({len(raw_data)} < {MIN_REQUIRED_OBSERVATIONS}).")
                        continue

                    data_map[t] = raw_data
                    valid_tickers.append(t)

                except Exception as ex:
                    skipped_tickers.append(t)
                    pipeline_warnings.append(f"Skipped {t}: Data retrieval error ({str(ex)}).")

            selected_tickers = valid_tickers
            if len(selected_tickers) < 2:
                return {
                    "error": "DATA SOURCE ERROR: Could not retrieve market data from Yahoo Finance. Network error or invalid ticker response.",
                    "error_stage": "Data Retrieval",
                    "error_type": "DATA_SOURCE_ERROR",
                    "suggestion": "Click [Switch to Demo Mode] in Settings/Data Source for offline presentation.",
                }

            price_df, _ = fetch_aligned_close_prices(
                selected_tickers,
                period=inputs.get("period_code", "1y"),
                interval=inputs.get("interval_code", "1d"),
                allow_demo=False
            )

            if price_df.empty:
                return {
                    "error": "DATA SOURCE ERROR: Failed to assemble aligned price dataframe from live market data.",
                    "error_stage": "Data Retrieval",
                    "error_type": "ALIGNMENT_ERROR",
                    "suggestion": "Click [Switch to Demo Mode] in Settings/Data Source for guaranteed offline demo.",
                }

            price_df = price_df.ffill().bfill()
            fetch_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            data_source_label = "Historical Data / Yahoo Finance"

    except Exception as e:
        return {
            "error": f"DATA SOURCE ERROR: {str(e)}",
            "error_stage": "Data Retrieval",
            "error_type": "DATA_SOURCE_ERROR",
            "suggestion": "Switch to Demo Mode in Settings/Data Source for offline presentation.",
        }

    # Stage 3: Financial Metrics Calculation
    try:
        df_metrics = generate_stock_metrics_table(data_map)
        return_matrix = calculate_return_matrix(data_map)
    except Exception as e:
        return {"error": f"Financial metrics calculation error: {str(e)}", "error_stage": "Financial Metrics", "error_type": "METRICS_ERROR"}

    # Stage 4: Correlation Matrix & Graph Topology
    try:
        corr_df = calculate_correlation_matrix(return_matrix)
        corr_df = corr_df.fillna(0.0)
        matrix_props = validate_matrix_properties(corr_df)
        corr_stats = calculate_correlation_summary_stats(corr_df)

        G = build_correlation_graph(corr_df, threshold=inputs.get("corr_threshold", 0.70))
        g_stats = calculate_graph_statistics(G)
    except Exception as e:
        return {"error": f"Correlation engine error: {str(e)}", "error_stage": "Correlation Matrix", "error_type": "CORRELATION_ERROR"}

    # Stage 5: Graph Coloring
    try:
        coloring_dict = color_graph(G, strategy="largest_first")
        is_coloring_valid = validate_coloring(G, coloring_dict)
        color_groups = get_color_groups(coloring_dict)
        chromatic_info = calculate_exact_chromatic_number_if_feasible(G, max_vertices=15)
    except Exception as e:
        pipeline_warnings.append(f"Graph coloring fallback applied: {str(e)}")
        coloring_dict = {s: 0 for s in selected_tickers}
        is_coloring_valid = True
        color_groups = {0: selected_tickers}
        chromatic_info = {"exact": False, "reason": "Graph coloring degraded gracefully"}

    # Stage 6: Dominance Relation & Poset Hasse Diagram
    try:
        dominance_pairs = build_dominance_relation(df_metrics)
        non_dominated_stocks = get_non_dominated_stocks(dominance_pairs, selected_tickers)
        incomparable_pairs = get_incomparable_pairs(dominance_pairs, selected_tickers)
        is_refl = check_reflexive(dominance_pairs, selected_tickers)
        antisym_info = check_antisymmetric(dominance_pairs, selected_tickers)
        is_trans = check_transitive(dominance_pairs)
        is_poset = is_partial_order(dominance_pairs, selected_tickers)
        cover_pairs = build_cover_relation(dominance_pairs)
        hasse_dag = build_hasse_graph(cover_pairs, stocks=selected_tickers)
        maximal_elems, minimal_elems = find_maximal_minimal_elements(hasse_dag)
    except Exception as e:
        return {"error": f"Dominance relation engine error: {str(e)}", "error_stage": "Dominance Relation", "error_type": "DOMINANCE_ERROR"}

    # Stage 7: Set Theory & Boolean Screening
    try:
        set_high_return = define_high_return_set(df_metrics, return_threshold=inputs.get("return_threshold", 5.0))
        set_low_risk = define_low_risk_set(df_metrics, risk_threshold=inputs.get("risk_threshold", 20.0))
        set_high_volume = define_high_volume_set(df_metrics)
        df_boolean_screening = evaluate_screening_rule(
            df_metrics, return_threshold=inputs.get("return_threshold", 5.0), risk_threshold=inputs.get("risk_threshold", 20.0)
        )

        # Explicit Set Operations
        set_A_intersect_B = set_intersection(set_high_return, set_high_volume)
        set_A_intersect_C = set_intersection(set_high_return, set_low_risk)
        set_B_intersect_C = set_intersection(set_high_volume, set_low_risk)
        set_union_all = set_union(set_union(set_high_return, set_high_volume), set_low_risk)
        set_A_minus_B = set_difference(set_high_return, set_high_volume)

    except Exception as e:
        return {"error": f"Set theory screening error: {str(e)}", "error_stage": "Set Theory", "error_type": "SET_ERROR"}

    # Stage 8: Portfolio Combinations & DM Scoring Engine
    try:
        portfolio_k = inputs.get("portfolio_k", 3)
        candidate_pool = select_candidate_pool(df_metrics, pool_size=15)
        possible_combinations = calculate_combination_count(len(candidate_pool), portfolio_k)
        raw_combinations = generate_candidate_portfolios(candidate_pool, k=portfolio_k)

        if len(raw_combinations) > MAX_EVALUATED_COMBINATIONS_CAP:
            eval_combinations = raw_combinations[:MAX_EVALUATED_COMBINATIONS_CAP]
            cap_note = f"Evaluated {MAX_EVALUATED_COMBINATIONS_CAP:,} of {possible_combinations:,} possible combinations due to combinatorial size limits."
            pipeline_warnings.append(cap_note)
        else:
            eval_combinations = raw_combinations
            cap_note = None

        top3_rendered_text, top3_structured_data = render_top3_portfolios(
            candidate_portfolios=eval_combinations,
            stock_metrics=df_metrics,
            returns_matrix=return_matrix,
            corr_df=corr_df,
            coloring_dict=coloring_dict,
            top_n=3,
        )

        top_dm_score = top3_structured_data[0]["dm_score"] if top3_structured_data else 0.0
    except Exception as e:
        return {"error": f"DM Portfolio Scoring Engine error: {str(e)}", "error_stage": "DM Scoring", "error_type": "SCORING_ERROR"}

    return {
        "inputs": inputs,
        "is_demo_mode": is_demo_mode,
        "fetch_time": fetch_time,
        "data_source_label": data_source_label,
        "pipeline_warnings": pipeline_warnings,
        "skipped_tickers": skipped_tickers,
        "excluded_stocks_notes": excluded_stocks_notes,
        "selected_tickers": selected_tickers,
        "selection_table": selection_table,
        "price_df": price_df,
        "df_metrics": df_metrics,
        "return_matrix": return_matrix,
        "corr_df": corr_df,
        "matrix_props": matrix_props,
        "corr_stats": corr_stats,
        "G": G,
        "g_stats": g_stats,
        "coloring_dict": coloring_dict,
        "is_coloring_valid": is_coloring_valid,
        "color_groups": color_groups,
        "chromatic_info": chromatic_info,
        "dominance_pairs": dominance_pairs,
        "non_dominated_stocks": non_dominated_stocks,
        "incomparable_pairs": incomparable_pairs,
        "is_refl": is_refl,
        "antisym_info": antisym_info,
        "is_trans": is_trans,
        "is_poset": is_poset,
        "cover_pairs": cover_pairs,
        "hasse_dag": hasse_dag,
        "maximal_elems": maximal_elems,
        "minimal_elems": minimal_elems,
        "set_high_return": set_high_return,
        "set_low_risk": set_low_risk,
        "set_high_volume": set_high_volume,
        "set_A_intersect_B": set_A_intersect_B,
        "set_A_intersect_C": set_A_intersect_C,
        "set_B_intersect_C": set_B_intersect_C,
        "set_union_all": set_union_all,
        "set_A_minus_B": set_A_minus_B,
        "df_boolean_screening": df_boolean_screening,
        "eligible_stocks_count": len(selected_tickers),
        "candidate_pool": candidate_pool,
        "possible_combinations": possible_combinations,
        "evaluated_count": len(eval_combinations),
        "cap_note": cap_note,
        "top3_rendered_text": top3_rendered_text,
        "top3_structured_data": top3_structured_data,
        "top_dm_score": top_dm_score,
    }
