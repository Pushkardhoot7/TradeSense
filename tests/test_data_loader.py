"""
Unit Tests for Data Loader Module
"""

import os
import numpy as np
import pandas as pd
import pytest
from modules.data_loader import (
    get_historical_data,
    download_stock_data,
    clean_stock_data,
    align_stock_data,
    get_data_status,
    _generate_demo_data,
    DATA_SOURCE_LIVE,
    DATA_SOURCE_DEMO,
)


def test_clean_stock_data():
    """Test clean_stock_data handles missing values, duplicates, and column normalization."""
    raw_data = {
        "Close": [100.0, 105.0, 105.0, np.nan, 110.0],
        "Adj Close": [99.0, 104.0, 104.0, np.nan, 109.0],
        "Open": [98.0, 102.0, 102.0, 105.0, 108.0],
    }
    dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-02", "2023-01-03", "2023-01-04"])
    df = pd.DataFrame(raw_data, index=dates)

    cleaned = clean_stock_data(df)

    assert not cleaned.empty
    # Duplicate 2023-01-02 removed
    assert len(cleaned) == 4
    # Price column derived correctly
    assert "Price" in cleaned.columns
    # Missing values filled
    assert not cleaned["Price"].isna().any()


def test_align_stock_data():
    """Test align_stock_data aligns date indices across multiple stocks."""
    idx1 = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    idx2 = pd.to_datetime(["2023-01-02", "2023-01-03", "2023-01-04"])

    df1 = pd.DataFrame({"Price": [10, 11, 12]}, index=idx1)
    df2 = pd.DataFrame({"Price": [20, 21, 22]}, index=idx2)

    aligned = align_stock_data({"STOCK_A": df1, "STOCK_B": df2})

    assert "STOCK_A" in aligned
    assert "STOCK_B" in aligned
    # Common intersection index: 2023-01-02, 2023-01-03
    assert len(aligned["STOCK_A"]) == 2
    assert (aligned["STOCK_A"].index == aligned["STOCK_B"].index).all()


def test_demo_mode_reproducibility():
    """Test demo mode data generation is deterministic for fixed random seed."""
    df_demo1 = _generate_demo_data("TCS.NS", period="1y")
    df_demo2 = _generate_demo_data("TCS.NS", period="1y")

    assert not df_demo1.empty
    assert len(df_demo1) == 252
    assert "Price" in df_demo1.columns
    # Must be deterministic for fixed seed
    assert np.isclose(df_demo1["Price"].iloc[0], df_demo2["Price"].iloc[0])


def test_get_historical_data_fallback_and_status():
    """Test get_historical_data falls back to demo mode on invalid ticker and updates status."""
    res = get_historical_data("INVALID_TEST_TICKER_999.NS", period="3m")

    assert res["symbol"] == "INVALID_TEST_TICKER_999.NS"
    assert res["status"] in ["ok", "demo"]
    assert res["rows"] > 0
    assert not res["data"].empty

    # Status registry verification
    status_info = get_data_status("INVALID_TEST_TICKER_999.NS")
    assert status_info["symbol"] == "INVALID_TEST_TICKER_999.NS"


def test_download_stock_data_invalid():
    """Test download_stock_data returns None on invalid ticker without crashing."""
    df = download_stock_data("NON_EXISTENT_XYZ_12345.NS", period="1mo")
    assert df is None
