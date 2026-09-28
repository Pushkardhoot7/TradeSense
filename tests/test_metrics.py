"""
Unit Tests for Financial Metrics Engine using Hand-Computable Datasets
"""

import math
import numpy as np
import pandas as pd
import pytest
from modules.financial_metrics import (
    calculate_daily_returns,
    calculate_period_return,
    calculate_volatility,
    calculate_annualized_risk,
    calculate_volume_metrics,
    calculate_stock_metrics,
    generate_stock_metrics_table,
    ANNUAL_TRADING_DAYS_NSE,
)


def test_calculate_period_return():
    """Test period return formula against hand-verified expected value."""
    # Initial: 100.0, Final: 125.0 -> Return = ((125 - 100) / 100) * 100 = 25.0%
    ret = calculate_period_return(100.0, 125.0)
    assert np.isclose(ret, 25.0)

    # Initial: 200.0, Final: 150.0 -> Return = ((150 - 200) / 200) * 100 = -25.0%
    ret_loss = calculate_period_return(200.0, 150.0)
    assert np.isclose(ret_loss, -25.0)

    # Invalid initial price
    assert calculate_period_return(0.0, 100.0) == 0.0


def test_calculate_daily_returns():
    """Test daily returns calculation r_t = (P_t - P_(t-1)) / P_(t-1) against small series."""
    prices = pd.Series([100.0, 105.0, 120.0])
    returns = calculate_daily_returns(prices)

    assert len(returns) == 2
    # r_1 = (105 - 100) / 100 = 0.05
    assert np.isclose(returns.iloc[0], 0.05)
    # r_2 = (120 - 105) / 105 = 0.142857...
    assert np.isclose(returns.iloc[1], 15.0 / 105.0)


def test_calculate_volatility_and_annualization():
    """Test standard deviation and annualization (daily_vol * sqrt(252)) against hand-verified values."""
    # Hand computable daily returns: [0.01, -0.01, 0.02, -0.02]
    # mean = 0.0
    # diff sq = 0.0001 + 0.0001 + 0.0004 + 0.0004 = 0.0010
    # sample variance = 0.0010 / (4 - 1) = 0.0003333333333333333
    # std = sqrt(0.0003333333333) = 0.018257418583505537
    returns = pd.Series([0.01, -0.01, 0.02, -0.02])
    daily_vol = calculate_volatility(returns)

    expected_std = np.std([0.01, -0.01, 0.02, -0.02], ddof=1)
    assert np.isclose(daily_vol, expected_std)

    # Annualized Risk % = daily_vol * sqrt(252) * 100
    risk_pct = calculate_annualized_risk(daily_vol, trading_days_per_year=252)
    expected_risk = expected_std * math.sqrt(252) * 100.0
    assert np.isclose(risk_pct, expected_risk)


def test_calculate_volume_metrics():
    """Test volume metrics calculation."""
    volume = pd.Series([1000, 2000, 3000, 4000])
    vol_dict = calculate_volume_metrics(volume)

    assert vol_dict["avg_volume"] == 2500.0
    assert vol_dict["min_volume"] == 1000.0
    assert vol_dict["max_volume"] == 4000.0


def test_missing_and_insufficient_data_handling():
    """Test missing/insufficient data does not crash and reports status correctly."""
    # Empty DataFrame
    df_empty = pd.DataFrame()
    metrics_empty = calculate_stock_metrics("TEST.NS", df_empty)
    assert metrics_empty["Data Status"] == "DATA NOT AVAILABLE"
    assert metrics_empty["Trading Days"] == 0

    # Insufficient data (1 row)
    df_short = pd.DataFrame({"Price": [100.0]}, index=pd.to_datetime(["2023-01-01"]))
    metrics_short = calculate_stock_metrics("TEST.NS", df_short)
    assert metrics_short["Data Status"] == "INSUFFICIENT DATA"
    assert metrics_short["Trading Days"] == 1


def test_generate_stock_metrics_table():
    """Test generating output dataframe per Section 7 schema."""
    dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"])
    df1 = pd.DataFrame({"Price": [100.0, 102.0, 105.0], "Volume": [500, 600, 700]}, index=dates)
    df2 = pd.DataFrame({"Price": [200.0, 198.0, 195.0], "Volume": [1000, 1100, 1200]}, index=dates)

    table = generate_stock_metrics_table({"STOCK_A.NS": df1, "STOCK_B.NS": df2})

    assert not table.empty
    assert len(table) == 2
    required_cols = ["Stock", "Latest Price", "Return %", "Risk %", "Average Volume", "Trading Days"]
    for col in required_cols:
        assert col in table.columns
