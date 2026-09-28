"""
Pytest configuration for TradeSense V2 backend tests.
Sets up the test database (in-memory SQLite) and provides shared fixtures.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Ensure project root is on sys.path for all test imports
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_tickers():
    return ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS", "RELIANCE.NS"]


@pytest.fixture
def sample_adj_dict():
    """Simple adjacency dict: TCS-INFY, INFY-HDFC, TCS-WIPRO"""
    return {
        "TCS.NS": [("INFY.NS", 0.85), ("WIPRO.NS", 0.75)],
        "INFY.NS": [("TCS.NS", 0.85), ("HDFCBANK.NS", 0.72)],
        "HDFCBANK.NS": [("INFY.NS", 0.72)],
        "WIPRO.NS": [("TCS.NS", 0.75)],
        "RELIANCE.NS": [],
    }


@pytest.fixture
def sample_corr_df():
    """5x5 Pearson correlation DataFrame (symmetric, diagonal=1)."""
    tickers = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS", "RELIANCE.NS"]
    data = np.array([
        [1.00, 0.85, 0.45, 0.75, 0.30],
        [0.85, 1.00, 0.72, 0.68, 0.25],
        [0.45, 0.72, 1.00, 0.40, 0.55],
        [0.75, 0.68, 0.40, 1.00, 0.20],
        [0.30, 0.25, 0.55, 0.20, 1.00],
    ])
    return pd.DataFrame(data, index=tickers, columns=tickers)


@pytest.fixture
def sample_price_df():
    """252-day synthetic price DataFrame for 5 stocks."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252, freq="B")
    tickers = ["TCS.NS", "INFY.NS", "HDFCBANK.NS", "WIPRO.NS", "RELIANCE.NS"]
    prices = {}
    for i, t in enumerate(tickers):
        daily_returns = np.random.normal(0.0005 * (i + 1), 0.012, 252)
        price_series = 1000.0 * np.cumprod(1 + daily_returns)
        prices[t] = price_series
    return pd.DataFrame(prices, index=dates)


@pytest.fixture
def sample_metrics():
    """List of stock metric dicts matching backend schema."""
    return [
        {"symbol": "TCS.NS", "return_pct": 18.5, "risk_pct": 14.2, "avg_volume": 2500000, "sharpe": 0.95},
        {"symbol": "INFY.NS", "return_pct": 12.3, "risk_pct": 16.8, "avg_volume": 4200000, "sharpe": 0.43},
        {"symbol": "HDFCBANK.NS", "return_pct": 8.7, "risk_pct": 12.1, "avg_volume": 8100000, "sharpe": 0.31},
        {"symbol": "WIPRO.NS", "return_pct": -3.2, "risk_pct": 19.5, "avg_volume": 1800000, "sharpe": -0.42},
        {"symbol": "RELIANCE.NS", "return_pct": 21.4, "risk_pct": 15.0, "avg_volume": 6500000, "sharpe": 1.09},
    ]
