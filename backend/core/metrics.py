"""
metrics.py
----------
Financial metrics computation module for TradeSense V2.

All formulas are implemented from scratch using pandas and the Python
standard library — no QuantLib / TA-Lib wrappers.

Formulas
--------
Daily return      : r_t = (P_t - P_{t-1}) / P_{t-1}
Period return     : ((P_final - P_initial) / P_initial) * 100
Volatility        : sample std of daily returns (ddof=1)
Annualised risk   : daily_vol * sqrt(252) * 100
CAGR              : ((P_final / P_initial)^(1/years) - 1) * 100
Sharpe ratio      : (CAGR% - rf%) / annual_risk%
Max drawdown      : max peak-to-trough decline as fraction
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Low-level scalar / series computations
# ---------------------------------------------------------------------------

def calculate_daily_returns(prices: pd.Series) -> pd.Series:
    """Compute simple daily returns from a price series.

    Parameters
    ----------
    prices:
        Time-ordered closing prices (any numeric ``pd.Series``).

    Returns
    -------
    pd.Series
        Daily returns ``r_t = (P_t - P_{t-1}) / P_{t-1}``.
        The first element is ``NaN`` (no predecessor).
    """
    if prices.empty:
        return pd.Series(dtype=float)
    returns = (prices - prices.shift(1)) / prices.shift(1)
    returns.name = "daily_return"
    return returns


def calculate_period_return(initial_price: float, final_price: float) -> float:
    """Compute the total percentage return over a period.

    Parameters
    ----------
    initial_price:
        Price at the beginning of the period.
    final_price:
        Price at the end of the period.

    Returns
    -------
    float
        ``((P_final - P_initial) / P_initial) * 100``

    Raises
    ------
    ZeroDivisionError
        If *initial_price* is zero.
    """
    if initial_price == 0:
        raise ZeroDivisionError("initial_price must not be zero.")
    return ((final_price - initial_price) / initial_price) * 100.0


def calculate_volatility(daily_returns: pd.Series) -> float:
    """Compute the sample standard deviation of daily returns.

    Parameters
    ----------
    daily_returns:
        Series of daily returns (NaN values are dropped internally).

    Returns
    -------
    float
        Sample standard deviation (ddof=1).  Returns 0.0 when fewer
        than 2 non-NaN observations are present.
    """
    clean = daily_returns.dropna()
    if len(clean) < 2:
        return 0.0
    n = len(clean)
    mean = clean.mean()
    variance = ((clean - mean) ** 2).sum() / (n - 1)
    return math.sqrt(variance)


def calculate_annualized_risk(
    daily_volatility: float,
    trading_days: int = 252,
) -> float:
    """Scale daily volatility to an annualised figure.

    Parameters
    ----------
    daily_volatility:
        Standard deviation of daily returns (raw fraction, not percent).
    trading_days:
        Number of trading days per year (default 252).

    Returns
    -------
    float
        Annualised risk as a **percentage**:
        ``daily_volatility * sqrt(trading_days) * 100``.
    """
    return daily_volatility * math.sqrt(trading_days) * 100.0


def calculate_cagr(initial: float, final: float, years: float) -> float:
    """Compound Annual Growth Rate (CAGR).

    Parameters
    ----------
    initial:
        Initial portfolio / stock value.
    final:
        Final portfolio / stock value.
    years:
        Investment horizon in years.

    Returns
    -------
    float
        CAGR as a **percentage**:
        ``((final / initial)^(1 / years) - 1) * 100``.

    Raises
    ------
    ZeroDivisionError
        If *initial* or *years* is zero.
    ValueError
        If ``final / initial`` is negative (cannot take real root).
    """
    if initial == 0:
        raise ZeroDivisionError("initial must not be zero.")
    if years == 0:
        raise ZeroDivisionError("years must not be zero.")
    ratio = final / initial
    if ratio < 0:
        raise ValueError(
            "final / initial ratio is negative; CAGR is not defined."
        )
    return (ratio ** (1.0 / years) - 1.0) * 100.0


def calculate_sharpe(
    cagr_pct: float,
    risk_pct: float,
    risk_free_rate: float = 5.0,
) -> float:
    """Sharpe ratio using annualised CAGR and risk (both as percentages).

    Parameters
    ----------
    cagr_pct:
        CAGR expressed as a percentage (e.g. 12.5 for 12.5 %).
    risk_pct:
        Annualised standard deviation expressed as a percentage.
    risk_free_rate:
        Annual risk-free rate as a percentage (default 5.0 %).

    Returns
    -------
    float
        Sharpe ratio = ``(CAGR% - rf%) / risk%``.
        Returns 0.0 when *risk_pct* is zero.
    """
    if risk_pct == 0:
        return 0.0
    return (cagr_pct - risk_free_rate) / risk_pct


def calculate_max_drawdown(prices: pd.Series) -> float:
    """Maximum peak-to-trough drawdown over the price series.

    Parameters
    ----------
    prices:
        Time-ordered closing prices.

    Returns
    -------
    float
        Maximum drawdown as a **negative fraction** in [-1, 0].
        e.g. ``-0.35`` means a 35 % peak-to-trough decline.
        Returns 0.0 if the series has fewer than 2 values.
    """
    prices = prices.dropna()
    if len(prices) < 2:
        return 0.0

    max_drawdown: float = 0.0
    peak: float = prices.iloc[0]

    for price in prices:
        if price > peak:
            peak = price
        if peak > 0:
            drawdown = (price - peak) / peak
            if drawdown < max_drawdown:
                max_drawdown = drawdown

    return max_drawdown


# ---------------------------------------------------------------------------
# Per-stock metrics dictionary
# ---------------------------------------------------------------------------

def calculate_stock_metrics(symbol: str, price_df: pd.DataFrame) -> dict[str, Any]:
    """Compute a full set of financial metrics for a single stock.

    Parameters
    ----------
    symbol:
        Ticker symbol (used only as metadata in the returned dict).
    price_df:
        DataFrame with at least a ``Close`` column and a ``DatetimeIndex``.

    Returns
    -------
    dict with keys:
        ``symbol``, ``initial_price``, ``final_price``,
        ``period_return_pct``, ``daily_volatility``,
        ``annualized_risk_pct``, ``years``, ``cagr_pct``,
        ``sharpe_ratio``, ``max_drawdown_pct``,
        ``trading_days``, ``start_date``, ``end_date``.
    """
    close: pd.Series = price_df["Close"].dropna()

    if len(close) < 2:
        return {
            "symbol": symbol,
            "error": "Insufficient data (need at least 2 closing prices).",
        }

    initial_price: float = float(close.iloc[0])
    final_price: float = float(close.iloc[-1])
    trading_days: int = len(close)

    # Date range
    start_date: str = str(price_df.index[0].date())
    end_date: str = str(price_df.index[-1].date())
    years: float = trading_days / 252.0

    # Core metrics
    daily_returns: pd.Series = calculate_daily_returns(close)
    period_return: float = calculate_period_return(initial_price, final_price)
    daily_vol: float = calculate_volatility(daily_returns)
    ann_risk: float = calculate_annualized_risk(daily_vol)
    max_dd: float = calculate_max_drawdown(close)

    try:
        cagr: float = calculate_cagr(initial_price, final_price, years)
    except (ZeroDivisionError, ValueError):
        cagr = 0.0

    sharpe: float = calculate_sharpe(cagr, ann_risk)

    return {
        "symbol": symbol,
        "initial_price": round(initial_price, 2),
        "final_price": round(final_price, 2),
        "period_return_pct": round(period_return, 4),
        "daily_volatility": round(daily_vol, 6),
        "annualized_risk_pct": round(ann_risk, 4),
        "years": round(years, 4),
        "cagr_pct": round(cagr, 4),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "trading_days": trading_days,
        "start_date": start_date,
        "end_date": end_date,
    }


# ---------------------------------------------------------------------------
# Multi-stock metrics table
# ---------------------------------------------------------------------------

def generate_metrics_table(
    data_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Build a comparison DataFrame of financial metrics for many stocks.

    Parameters
    ----------
    data_map:
        Mapping of ``{symbol: price_df}`` where each DataFrame has at
        least a ``Close`` column.

    Returns
    -------
    pd.DataFrame
        Rows = stocks, columns = metric fields from
        :func:`calculate_stock_metrics`.  An ``error`` column is
        present only when at least one stock fails.
    """
    records: list[dict[str, Any]] = []
    for symbol, df in data_map.items():
        metrics = calculate_stock_metrics(symbol, df)
        records.append(metrics)

    if not records:
        return pd.DataFrame()

    result_df = pd.DataFrame(records)
    if "symbol" in result_df.columns:
        result_df = result_df.set_index("symbol")
    return result_df
