"""
Unit Tests for Stock Selector Module
"""

import os
import tempfile
import pytest
from modules.stock_selector import (
    load_stock_universe,
    get_sectors,
    get_stocks_by_sector,
    rank_stocks,
    select_top_stocks_per_sector,
)


@pytest.fixture
def temp_stock_universe_csv():
    """Create a temporary stock_universe.csv file for testing edge cases."""
    content = """symbol,company_name,sector,industry,exchange
TCS.NS,Tata Consultancy Services,IT Services,Software,NSE
INFY.NS,Infosys Ltd.,IT Services,Software,NSE
HCLTECH.NS,HCL Technologies,IT Services,Software,NSE
INVALID_TICKER,Bad Company,IT Services,Software,NSE
TCS.NS,Duplicate TCS,IT Services,Software,NSE
HDFCBANK.NS,HDFC Bank,Financial Services,Banking,NSE
ICICIBANK.NS,ICICI Bank,Financial Services,Banking,NSE
RELIANCE.NS,Reliance Industries,Energy,Oil & Gas,NSE
"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv", encoding="utf-8") as f:
        f.write(content)
        temp_path = f.name

    yield temp_path

    if os.path.exists(temp_path):
        os.remove(temp_path)


def test_sector_retrieval(temp_stock_universe_csv):
    """Test get_sectors() returns distinct list of sectors."""
    sectors = get_sectors(temp_stock_universe_csv)
    assert isinstance(sectors, list)
    assert len(sectors) == 3
    assert set(sectors) == {"Energy", "Financial Services", "IT Services"}


def test_sector_filtering(temp_stock_universe_csv):
    """Test get_stocks_by_sector() returns only matching-sector stocks."""
    it_stocks = get_stocks_by_sector("IT Services", temp_stock_universe_csv)
    assert len(it_stocks) == 3
    for stock in it_stocks:
        assert stock["sector"] == "IT Services"
        assert stock["symbol"].endswith(".NS")


def test_duplicate_and_invalid_ticker_handling(temp_stock_universe_csv):
    """Test duplicate ticker removal and invalid ticker skip without crashing."""
    stocks = load_stock_universe(temp_stock_universe_csv)
    symbols = [s["symbol"] for s in stocks]
    
    # Duplicate TCS.NS should be ignored
    assert symbols.count("TCS.NS") == 1
    # INVALID_TICKER (doesn't end with .NS) should be skipped
    assert "INVALID_TICKER" not in symbols


def test_fewer_than_15_handling(temp_stock_universe_csv):
    """Test fewer-than-15 stock selection (no padding, correct counts)."""
    result = select_top_stocks_per_sector("IT Services", temp_stock_universe_csv)
    assert len(result) == 1
    record = result[0]
    assert record["sector"] == "IT Services"
    assert record["available_count"] == 3
    assert record["selected_count"] == 3
    assert len(record["selected_stocks"]) == 3
    assert record["selected_stocks"] == ["TCS.NS", "INFY.NS", "HCLTECH.NS"]


def test_maximum_15_selection_enforcement():
    """Test max-15 selection enforcement when sector has more than 15 stocks."""
    # Test on full stock_universe.csv
    results = select_top_stocks_per_sector()
    assert len(results) > 0
    for res in results:
        assert res["selected_count"] <= 15
        assert len(res["selected_stocks"]) <= 15
        if res["available_count"] > 15:
            assert res["selected_count"] == 15
        else:
            assert res["selected_count"] == res["available_count"]


def test_rank_stocks():
    """Test stock ranking methodology returns rank positions."""
    stocks = [{"symbol": "A.NS"}, {"symbol": "B.NS"}]
    ranked = rank_stocks(stocks)
    assert len(ranked) == 2
    assert ranked[0]["rank"] == 1
    assert ranked[1]["rank"] == 2
    assert ranked[0]["ranking_method"] == "market_cap_index_rank"
