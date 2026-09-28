"""
technical.py - Technical Analysis Indicators for TradeSense V2
"""
from __future__ import annotations
import math
from typing import Any
import numpy as np
import pandas as pd

def calculate_sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=1).mean()

def calculate_ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def calculate_macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9):
    fast_ema = calculate_ema(series, fast_period)
    slow_ema = calculate_ema(series, slow_period)
    macd_line = fast_ema - slow_ema
    signal_line = calculate_ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

def calculate_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0):
    middle_band = calculate_sma(series, window)
    rolling_std = series.rolling(window=window, min_periods=1).std(ddof=0)
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    return upper_band, middle_band, lower_band

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High'] if 'High' in df else df['Close']
    low = df['Low'] if 'Low' in df else df['Close']
    close = df['Close']
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=1).mean()

def calculate_vwap(df: pd.DataFrame) -> pd.Series:
    if 'Volume' not in df or 'Close' not in df:
        return df['Close'] if 'Close' in df else pd.Series(dtype=float)
    typical_price = df['Close']
    if 'High' in df and 'Low' in df:
        typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    vol = df['Volume'].fillna(0)
    cum_pv = (typical_price * vol).cumsum()
    cum_vol = vol.cumsum().replace(0, np.nan)
    return (cum_pv / cum_vol).fillna(typical_price)

def compute_all_technicals(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {}

    # Flatten MultiIndex columns if yfinance returns (Price, Ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    if 'Close' not in df.columns:
        return {}

    close = df['Close'].dropna()
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    if len(close) < 5:
        return {}
    sma20 = calculate_sma(close, 20)
    sma50 = calculate_sma(close, 50)
    sma200 = calculate_sma(close, 200) if len(close) >= 200 else calculate_sma(close, len(close))
    ema20 = calculate_ema(close, 20)
    ema50 = calculate_ema(close, 50)
    rsi14 = calculate_rsi(close, 14)
    macd, signal, hist = calculate_macd(close)
    upper_bb, mid_bb, lower_bb = calculate_bollinger_bands(close, 20, 2.0)
    atr = calculate_atr(df, 14)
    vwap = calculate_vwap(df)
    latest_close = float(close.iloc[-1])
    latest_rsi = round(float(rsi14.iloc[-1]), 2) if not pd.isna(rsi14.iloc[-1]) else 50.0
    rsi_condition = 'Neutral (40-60)'
    if latest_rsi >= 70:
        rsi_condition = 'Overbought (>= 70)'
    elif latest_rsi <= 30:
        rsi_condition = 'Oversold (<= 30)'
    elif latest_rsi > 60:
        rsi_condition = 'Moderately Bullish (60-70)'
    elif latest_rsi < 40:
        rsi_condition = 'Moderately Bearish (30-40)'
    return {
        'latest': {
            'close': round(latest_close, 2),
            'sma20': round(float(sma20.iloc[-1]), 2) if not pd.isna(sma20.iloc[-1]) else None,
            'sma50': round(float(sma50.iloc[-1]), 2) if not pd.isna(sma50.iloc[-1]) else None,
            'sma200': round(float(sma200.iloc[-1]), 2) if not pd.isna(sma200.iloc[-1]) else None,
            'ema20': round(float(ema20.iloc[-1]), 2) if not pd.isna(ema20.iloc[-1]) else None,
            'ema50': round(float(ema50.iloc[-1]), 2) if not pd.isna(ema50.iloc[-1]) else None,
            'rsi': latest_rsi,
            'rsi_condition': rsi_condition,
            'macd': round(float(macd.iloc[-1]), 2) if not pd.isna(macd.iloc[-1]) else None,
            'macd_signal': round(float(signal.iloc[-1]), 2) if not pd.isna(signal.iloc[-1]) else None,
            'macd_hist': round(float(hist.iloc[-1]), 2) if not pd.isna(hist.iloc[-1]) else None,
            'bb_upper': round(float(upper_bb.iloc[-1]), 2) if not pd.isna(upper_bb.iloc[-1]) else None,
            'bb_mid': round(float(mid_bb.iloc[-1]), 2) if not pd.isna(mid_bb.iloc[-1]) else None,
            'bb_lower': round(float(lower_bb.iloc[-1]), 2) if not pd.isna(lower_bb.iloc[-1]) else None,
            'atr': round(float(atr.iloc[-1]), 2) if not pd.isna(atr.iloc[-1]) else None,
            'vwap': round(float(vwap.iloc[-1]), 2) if not pd.isna(vwap.iloc[-1]) else None,
        },
        'series': {
            'dates': [str(d.date()) if hasattr(d, 'date') else str(d) for d in close.index],
            'close': [round(float(v), 2) for v in close],
            'sma20': [round(float(v), 2) if not pd.isna(v) else None for v in sma20],
            'sma50': [round(float(v), 2) if not pd.isna(v) else None for v in sma50],
            'ema20': [round(float(v), 2) if not pd.isna(v) else None for v in ema20],
            'rsi': [round(float(v), 2) if not pd.isna(v) else None for v in rsi14],
            'macd': [round(float(v), 2) if not pd.isna(v) else None for v in macd],
            'macd_signal': [round(float(v), 2) if not pd.isna(v) else None for v in signal],
            'macd_hist': [round(float(v), 2) if not pd.isna(v) else None for v in hist],
            'bb_upper': [round(float(v), 2) if not pd.isna(v) else None for v in upper_bb],
            'bb_lower': [round(float(v), 2) if not pd.isna(v) else None for v in lower_bb],
        },
        'disclaimer': 'Technical indicators are descriptive historical calculations and do NOT predict future performance or constitute investment advice.'
    }
