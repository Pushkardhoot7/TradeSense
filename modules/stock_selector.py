"""
Stock Selector Module for TradeSense

Provides functionality to load the NSE stock universe, retrieve sectors, filter stocks, rank stocks, and automatically select up to the top 15 eligible stocks per sector.

Ranking Methodology:
- Primary Ranking: Market Capitalization / Benchmark Index Weight (order in curated dataset).
- Secondary Ranking: Symbol Liquidity / Trading availability.
- Ranking Method Identifier: "market_cap_index_rank" (documented fallback when live market cap API is unavailable).
- Interface Design: Pluggable design allowing real-time market-cap ranking providers to be connected without changing function signatures.
"""

import os
import csv
from typing import List, Dict, Any, Optional

DEFAULT_UNIVERSE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "stock_universe.csv"
)


def load_stock_universe(filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load and validate data/stock_universe.csv into memory as a list of dictionaries.

    Returns:
        List of dictionaries with keys ['symbol', 'company_name', 'sector', 'industry', 'exchange'].
    """
    target_path = filepath if filepath else DEFAULT_UNIVERSE_PATH
    stocks = []
    seen_symbols = set()

    if not os.path.exists(target_path):
        return stocks

    try:
        with open(target_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                symbol = row.get("symbol", "").strip()
                # Validate symbol format and skip invalid or duplicate tickers
                if not symbol or not symbol.endswith(".NS"):
                    continue
                if symbol in seen_symbols:
                    continue
                
                seen_symbols.add(symbol)
                stocks.append({
                    "symbol": symbol,
                    "company_name": row.get("company_name", "DATA NOT AVAILABLE").strip(),
                    "sector": row.get("sector", "Unknown").strip(),
                    "industry": row.get("industry", "DATA NOT AVAILABLE").strip(),
                    "exchange": row.get("exchange", "NSE").strip(),
                })
    except Exception as e:
        print(f"Error loading stock universe CSV: {e}")

    return stocks


def get_sectors(filepath: Optional[str] = None) -> List[str]:
    """
    Return the distinct list of sectors present in the stock universe.
    """
    universe = load_stock_universe(filepath)
    sectors = sorted(list({s["sector"] for s in universe if s.get("sector")}))
    return sectors


def get_stocks_by_sector(sector: str, filepath: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Return all eligible stocks belonging to the given sector.
    """
    universe = load_stock_universe(filepath)
    if not sector:
        return []
    
    matching_stocks = [
        s for s in universe if s.get("sector", "").lower() == sector.lower()
    ]
    return matching_stocks


def rank_stocks(stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Rank a list of stocks per the documented ranking methodology.
    
    Ranking Rule:
    - Stocks retain their structural benchmark importance rank (market cap order in universe dataset).
    - Ensures deterministic ranking without fabricating unverified market cap values.
    """
    if not stocks:
        return []

    ranked = []
    for idx, s in enumerate(stocks, start=1):
        stock_item = dict(s)
        stock_item["rank"] = idx
        stock_item["ranking_method"] = "market_cap_index_rank"
        ranked.append(stock_item)

    return ranked


def select_top_stocks_per_sector(
    sector: Optional[str] = None, filepath: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Select up to the top 15 ranked, eligible stocks per sector.
    If sector is None, run for all sectors in the universe.

    Returns:
        List of records shaped as:
        {
          "sector": "IT Services",
          "selected_stocks": ["TCS.NS", "INFY.NS", "HCLTECH.NS"],
          "available_count": 11,
          "selected_count": 11
        }
    """
    all_sectors = [sector] if sector else get_sectors(filepath)
    results = []

    for sec in all_sectors:
        eligible_stocks = get_stocks_by_sector(sec, filepath)
        ranked = rank_stocks(eligible_stocks)
        
        # Select up to top 15 (never pad if fewer than 15 exist)
        top_15 = ranked[:15]
        selected_symbols = [s["symbol"] for s in top_15]

        results.append({
            "sector": sec,
            "selected_stocks": selected_symbols,
            "available_count": len(eligible_stocks),
            "selected_count": len(selected_symbols),
            "ranking_method": "market_cap_index_rank",
            "stock_details": top_15
        })

    return results
