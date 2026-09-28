"""
Fixed Reproducible Sample Dataset for Demo Mode in TradeSense

Stock Count: 8 stocks
Sectors Covered: IT (TCS.NS, INFY.NS), Financials (HDFCBANK.NS, ICICIBANK.NS), Auto (TATAMOTORS.NS, MARUTI.NS), Energy (RELIANCE.NS, NTPC.NS)
Date Range: 252 trading days (1 full year)
Characteristics: Includes high positive correlations, uncorrelated pairs, diverse returns & risk profiles, non-dominated Pareto frontier stocks.
100% deterministic — no seeds or random generation.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np
import pandas as pd


def get_demo_stock_universe() -> List[Dict[str, Any]]:
    """Return static demo stock universe across 4 sectors."""
    return [
        {"symbol": "TCS.NS", "company_name": "Tata Consultancy Services", "sector": "Information Technology", "industry": "IT Services", "rank": 1},
        {"symbol": "INFY.NS", "company_name": "Infosys Limited", "sector": "Information Technology", "industry": "IT Services", "rank": 2},
        {"symbol": "HDFCBANK.NS", "company_name": "HDFC Bank Limited", "sector": "Financial Services", "industry": "Private Bank", "rank": 1},
        {"symbol": "ICICIBANK.NS", "company_name": "ICICI Bank Limited", "sector": "Financial Services", "industry": "Private Bank", "rank": 2},
        {"symbol": "TATAMOTORS.NS", "company_name": "Tata Motors Limited", "sector": "Automobile", "industry": "Auto Passenger Cars", "rank": 1},
        {"symbol": "MARUTI.NS", "company_name": "Maruti Suzuki India Ltd", "sector": "Automobile", "industry": "Auto Passenger Cars", "rank": 2},
        {"symbol": "RELIANCE.NS", "company_name": "Reliance Industries Ltd", "sector": "Energy", "industry": "Refineries & Marketing", "rank": 1},
        {"symbol": "NTPC.NS", "company_name": "NTPC Limited", "sector": "Energy", "industry": "Power Generation", "rank": 2},
    ]


def generate_demo_price_dataframe(num_days: int = 252) -> pd.DataFrame:
    """
    Generate 100% deterministic 252-day historical price dataframe for demo mode.
    Uses analytical trigonometric and step functions so output is identical on every run.
    """
    start_date = datetime(2025, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    t = np.linspace(0, 4 * np.pi, num_days)

    # Deterministic price series curves
    prices = {
        "TCS.NS": 3500.0 + 300.0 * np.sin(t) + 2.0 * np.arange(num_days),
        "INFY.NS": 1600.0 + 150.0 * np.sin(t + 0.1) + 1.2 * np.arange(num_days),  # Highly correlated with TCS
        "HDFCBANK.NS": 1500.0 + 100.0 * np.cos(t) + 0.8 * np.arange(num_days),    # Uncorrelated with IT
        "ICICIBANK.NS": 1000.0 + 80.0 * np.cos(t + 0.1) + 1.0 * np.arange(num_days),
        "TATAMOTORS.NS": 700.0 + 200.0 * np.sin(2 * t) + 1.5 * np.arange(num_days), # Volatile high return
        "MARUTI.NS": 10000.0 + 500.0 * np.sin(2 * t + 0.2) + 5.0 * np.arange(num_days),
        "RELIANCE.NS": 2800.0 + 200.0 * np.cos(1.5 * t) + 2.5 * np.arange(num_days),
        "NTPC.NS": 320.0 + 30.0 * np.cos(1.5 * t + 0.3) + 0.3 * np.arange(num_days),
    }

    df = pd.DataFrame(prices, index=dates)
    df.index.name = "Date"
    return df


def get_demo_historical_data_map(num_days: int = 252) -> Dict[str, pd.DataFrame]:
    """Return dictionary of individual stock price dataframes for demo mode."""
    price_df = generate_demo_price_dataframe(num_days)
    data_map = {}

    for col in price_df.columns:
        series = price_df[col]
        stock_df = pd.DataFrame({
            "Open": series * 0.995,
            "High": series * 1.01,
            "Low": series * 0.99,
            "Close": series,
            "Adj Close": series,
            "Volume": 1000000 + np.abs(series * 100),
        }, index=series.index)
        data_map[col] = stock_df

    return data_map
