"""
Financial Metrics Module for TradeSense

Calculates period returns, daily returns, daily volatility, annualized risk, volume statistics, and price metrics.

Annualization Rationale:
252 trading days per year is used as it approximates the number of active trading sessions on the National Stock Exchange of India (NSE) after accounting for 52 weekends and statutory market holidays. Under the standard financial random walk assumption of independent and identically distributed (i.i.d.) daily log/simple returns, return variance scales linearly with time (T), and standard deviation (volatility) scales with the square root of time (sqrt(252)).
"""

import math
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

ANNUAL_TRADING_DAYS_NSE = 252


def calculate_daily_returns(prices: pd.Series) -> pd.Series:
    """
    Compute daily percentage return r_t = (P_t - P_(t-1)) / P_(t-1) from a price series.

    Args:
        prices: pandas Series of historical prices.

    Returns:
        pandas Series of daily returns.
    """
    if prices is None or prices.empty or len(prices) < 2:
        return pd.Series(dtype=float)

    returns = prices.pct_change().dropna()
    return returns


def calculate_period_return(initial_price: float, final_price: float) -> float:
    """
    Compute period return %: Return % = ((Final Price - Initial Price) / Initial Price) * 100

    Args:
        initial_price: Starting price of the period.
        final_price: Ending price of the period.

    Returns:
        Period return percentage (e.g. 15.5 for +15.5%).
    """
    if initial_price is None or final_price is None or initial_price <= 0:
        return 0.0

    return float(((final_price - initial_price) / initial_price) * 100.0)


def calculate_volatility(daily_returns: pd.Series) -> float:
    """
    Compute daily volatility: sample standard deviation std(daily_returns).

    Args:
        daily_returns: Series of daily returns.

    Returns:
        Daily standard deviation as a decimal ratio.
    """
    if daily_returns is None or daily_returns.empty or len(daily_returns) < 2:
        return 0.0

    std_dev = daily_returns.std(ddof=1)
    if math.isnan(std_dev):
        return 0.0
    return float(std_dev)


def calculate_annualized_risk(daily_volatility: float, trading_days_per_year: int = ANNUAL_TRADING_DAYS_NSE) -> float:
    """
    Annualize daily volatility: Risk % = daily_volatility * sqrt(trading_days_per_year) * 100

    Rationale:
    252 approximates NSE annual trading days. Scaling daily standard deviation by sqrt(252) annualizes the risk estimate under i.i.d. daily return assumptions.

    Args:
        daily_volatility: Daily standard deviation (decimal ratio).
        trading_days_per_year: Number of trading days in a year (default 252 for NSE).

    Returns:
        Annualized volatility percentage (Risk %).
    """
    if daily_volatility is None or daily_volatility < 0:
        return 0.0

    annualized_vol = daily_volatility * math.sqrt(trading_days_per_year) * 100.0
    return float(annualized_vol)


def calculate_volume_metrics(volume: pd.Series) -> Dict[str, float]:
    """
    Compute volume metrics over the selected period.

    Returns:
        Dictionary with keys 'avg_volume', 'min_volume', 'max_volume'.
    """
    if volume is None or volume.empty or volume.dropna().empty:
        return {"avg_volume": 0.0, "min_volume": 0.0, "max_volume": 0.0}

    clean_vol = volume.dropna()
    return {
        "avg_volume": float(clean_vol.mean()),
        "min_volume": float(clean_vol.min()),
        "max_volume": float(clean_vol.max()),
    }


def calculate_stock_metrics(symbol: str, historical_data: pd.DataFrame) -> Dict[str, Any]:
    """
    High-level entry point: compute the full metric set for one stock.
    Gracefully handles missing or insufficient data without crashing.

    Returns:
        Dictionary containing all price, return, risk, volume, and trading day metrics.
    """
    empty_result = {
        "Stock": symbol,
        "Latest Price": 0.0,
        "Initial Price": 0.0,
        "Min Price": 0.0,
        "Max Price": 0.0,
        "Average Price": 0.0,
        "Return %": 0.0,
        "Daily Volatility": 0.0,
        "Risk %": 0.0,
        "Average Volume": 0.0,
        "Min Volume": 0.0,
        "Max Volume": 0.0,
        "Trading Days": 0,
        "CAGR": 0.0,
        "Sharpe Ratio": 0.0,
        "Max Drawdown": 0.0,
        "Data Status": "INSUFFICIENT DATA",
    }

    if historical_data is None or historical_data.empty:
        empty_result["Data Status"] = "DATA NOT AVAILABLE"
        return empty_result

    # Determine consistent price series (Price or Adj Close or Close)
    if "Price" in historical_data.columns:
        price_series = historical_data["Price"].dropna()
    elif "Adj Close" in historical_data.columns:
        price_series = historical_data["Adj Close"].dropna()
    elif "Close" in historical_data.columns:
        price_series = historical_data["Close"].dropna()
    else:
        empty_result["Data Status"] = "DATA NOT AVAILABLE"
        return empty_result

    if price_series.empty or len(price_series) < 2:
        empty_result["Trading Days"] = len(price_series)
        empty_result["Data Status"] = "INSUFFICIENT DATA"
        return empty_result

    # Volume Series
    volume_series = historical_data["Volume"] if "Volume" in historical_data.columns else pd.Series(dtype=float)
    vol_metrics = calculate_volume_metrics(volume_series)

    # Price Extremes
    initial_p = float(price_series.iloc[0])
    latest_p = float(price_series.iloc[-1])
    min_p = float(price_series.min())
    max_p = float(price_series.max())
    avg_p = float(price_series.mean())

    # Returns & Volatility
    period_ret = calculate_period_return(initial_p, latest_p)
    daily_rets = calculate_daily_returns(price_series)
    daily_vol = calculate_volatility(daily_rets)
    annual_risk = calculate_annualized_risk(daily_vol, trading_days_per_year=ANNUAL_TRADING_DAYS_NSE)

    # Compound Annual Growth Rate (CAGR)
    trading_days = len(price_series)
    years = trading_days / float(ANNUAL_TRADING_DAYS_NSE)
    cagr_pct = (((latest_p / initial_p) ** (1.0 / max(years, 0.001))) - 1.0) * 100.0 if initial_p > 0 else 0.0

    # Sharpe Ratio (5% Risk-free rate)
    sharpe = (cagr_pct - 5.0) / (annual_risk + 1e-6) if annual_risk > 0 else 0.0

    # Max Drawdown
    cumulative = (1 + daily_rets).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_dd_pct = float(drawdown.min() * 100.0) if not drawdown.empty else 0.0

    return {
        "Stock": symbol,
        "Latest Price": round(latest_p, 2),
        "Initial Price": round(initial_p, 2),
        "Min Price": round(min_p, 2),
        "Max Price": round(max_p, 2),
        "Average Price": round(avg_p, 2),
        "Return %": round(period_ret, 2),
        "Daily Volatility": round(daily_vol, 4),
        "Risk %": round(annual_risk, 2),
        "Average Volume": round(vol_metrics["avg_volume"], 0),
        "Min Volume": round(vol_metrics["min_volume"], 0),
        "Max Volume": round(vol_metrics["max_volume"], 0),
        "Trading Days": trading_days,
        "CAGR": round(cagr_pct, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Max Drawdown": round(max_dd_pct, 2),
        "Data Status": "OK",
    }


def generate_stock_metrics_table(data_map: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate clean output dataframe with one row per stock.
    Primary columns: ['Stock', 'Latest Price', 'Return %', 'Risk %', 'Average Volume', 'Trading Days'].
    All remaining calculated fields included as columns.
    """
    if not data_map:
        return pd.DataFrame(
            columns=[
                "Stock",
                "Latest Price",
                "Return %",
                "Risk %",
                "Average Volume",
                "Trading Days",
                "Initial Price",
                "Min Price",
                "Max Price",
                "Average Price",
                "CAGR",
                "Sharpe Ratio",
                "Max Drawdown",
            ]
        )

    rows = []
    for symbol, df in data_map.items():
        metrics = calculate_stock_metrics(symbol, df)
        rows.append(metrics)

    df_result = pd.DataFrame(rows)
    primary_cols = [
        "Stock",
        "Latest Price",
        "Return %",
        "Risk %",
        "Average Volume",
        "Trading Days",
        "Initial Price",
        "Min Price",
        "Max Price",
        "Average Price",
        "CAGR",
        "Sharpe Ratio",
        "Max Drawdown",
        "Data Status",
    ]
    existing_cols = [c for c in primary_cols if c in df_result.columns]
    return df_result[existing_cols]


# Legacy helper compatibility functions
def calculate_annualized_volatility(daily_returns: pd.DataFrame, trading_days: int = ANNUAL_TRADING_DAYS_NSE) -> pd.Series:
    """Legacy helper compatibility."""
    if daily_returns is None or daily_returns.empty:
        return pd.Series(dtype=float)
    return daily_returns.std() * np.sqrt(trading_days)


def calculate_cagr(price_df: pd.DataFrame, trading_days: int = ANNUAL_TRADING_DAYS_NSE) -> pd.Series:
    """Legacy helper compatibility."""
    if price_df is None or price_df.empty or len(price_df) < 2:
        return pd.Series(dtype=float)
    years = len(price_df) / float(trading_days)
    return (price_df.iloc[-1] / price_df.iloc[0]).clip(lower=1e-6) ** (1.0 / max(years, 0.001)) - 1.0


def generate_metrics_summary(price_df: pd.DataFrame, risk_free_rate: float = 0.05) -> pd.DataFrame:
    """Legacy helper compatibility."""
    if price_df is None or price_df.empty:
        return pd.DataFrame(columns=["CAGR", "Volatility", "Sharpe Ratio", "Max Drawdown"])

    data_map = {col: pd.DataFrame({"Price": price_df[col]}) for col in price_df.columns}
    df_metrics = generate_stock_metrics_table(data_map)
    
    # Format for legacy summary tab
    df_summary = pd.DataFrame({
        "CAGR": df_metrics["CAGR"] / 100.0 if "CAGR" in df_metrics.columns else 0.0,
        "Volatility": df_metrics["Risk %"] / 100.0 if "Risk %" in df_metrics.columns else 0.0,
        "Sharpe Ratio": df_metrics["Sharpe Ratio"] if "Sharpe Ratio" in df_metrics.columns else 0.0,
        "Max Drawdown": df_metrics["Max Drawdown"] / 100.0 if "Max Drawdown" in df_metrics.columns else 0.0,
    }, index=df_metrics["Stock"])
    return df_summary
