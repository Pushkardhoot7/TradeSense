r"""
Set Theory and Boolean Logic Screening Module for TradeSense

Defines stock set partitions (High Return, Low Risk, High Volume), standard set operations (Intersection, Union, Difference), and per-stock Boolean Screening Rule evaluations.

Discrete Mathematics Concept:
- Set Theory: A collection of distinct elements.
  - High Return S_R = { s | Return(s) > return_threshold }
  - Low Risk S_V = { s | Risk(s) < risk_threshold }
  - High Volume S_Vol = { s | Volume(s) > mean(Volume) }
- Set Operations:
  - Intersection (A ∩ B): { s | s ∈ A ∧ s ∈ B }
  - Union (A ∪ B): { s | s ∈ A ∨ s ∈ B }
  - Difference (A \ B): { s | s ∈ A ∧ s ∉ B }
- Boolean Logic Screening Rule:
  QUALIFIED(s) = (Return(s) > return_threshold) ∧ (Risk(s) < risk_threshold) ∧ (Volume(s) > mean_volume)

Stock Market Framing:
This is a mathematical filter over defined threshold constraints, not a trading signal or buy recommendation.
"""

from typing import Set, Dict, Any, List
import pandas as pd


def define_high_return_set(stock_metrics: pd.DataFrame, return_threshold: float) -> Set[str]:
    """
    Return the set of stocks with Return > return_threshold (strict inequality).

    Args:
        stock_metrics: DataFrame containing 'Stock' and 'Return %'.
        return_threshold: Threshold value for return percentage.

    Returns:
        Set of stock symbols.
    """
    if stock_metrics is None or stock_metrics.empty:
        return set()

    ret_col = "Return %" if "Return %" in stock_metrics.columns else "CAGR"
    if ret_col not in stock_metrics.columns or "Stock" not in stock_metrics.columns:
        return set()

    filtered = stock_metrics[stock_metrics[ret_col] > return_threshold]
    return set(filtered["Stock"].tolist())


def define_low_risk_set(stock_metrics: pd.DataFrame, risk_threshold: float) -> Set[str]:
    """
    Return the set of stocks with Risk < risk_threshold (strict inequality).

    Args:
        stock_metrics: DataFrame containing 'Stock' and 'Risk %'.
        risk_threshold: Threshold value for annualized risk percentage.

    Returns:
        Set of stock symbols.
    """
    if stock_metrics is None or stock_metrics.empty:
        return set()

    risk_col = "Risk %" if "Risk %" in stock_metrics.columns else "Volatility"
    if risk_col not in stock_metrics.columns or "Stock" not in stock_metrics.columns:
        return set()

    filtered = stock_metrics[stock_metrics[risk_col] < risk_threshold]
    return set(filtered["Stock"].tolist())


def define_high_volume_set(stock_metrics: pd.DataFrame) -> Set[str]:
    """
    Return the set of stocks with Volume > average volume across the current selection (strict inequality).

    Args:
        stock_metrics: DataFrame containing 'Stock' and 'Average Volume'.

    Returns:
        Set of stock symbols.
    """
    if stock_metrics is None or stock_metrics.empty:
        return set()

    vol_col = "Average Volume"
    if vol_col not in stock_metrics.columns or "Stock" not in stock_metrics.columns:
        return set()

    clean_vols = stock_metrics[vol_col].dropna()
    if clean_vols.empty:
        return set()

    avg_volume = float(clean_vols.mean())
    filtered = stock_metrics[stock_metrics[vol_col] > avg_volume]
    return set(filtered["Stock"].tolist())


def set_intersection(set_a: Set[str], set_b: Set[str]) -> Set[str]:
    """
    Compute intersection A ∩ B = { x | x ∈ A ∧ x ∈ B }.
    """
    if set_a is None or set_b is None:
        return set()
    return set_a.intersection(set_b)


def set_union(set_a: Set[str], set_b: Set[str]) -> Set[str]:
    """
    Compute union A ∪ B = { x | x ∈ A ∨ x ∈ B }.
    """
    if set_a is None and set_b is None:
        return set()
    if set_a is None:
        return set(set_b)
    if set_b is None:
        return set(set_a)
    return set_a.union(set_b)


def set_difference(set_a: Set[str], set_b: Set[str]) -> Set[str]:
    r"""
    Compute difference A \ B = { x | x ∈ A ∧ x ∉ B }.
    """
    if set_a is None:
        return set()
    if set_b is None:
        return set(set_a)
    return set_a.difference(set_b)


def evaluate_screening_rule(
    stock_metrics: pd.DataFrame, return_threshold: float, risk_threshold: float
) -> pd.DataFrame:
    """
    Evaluate QUALIFIED per Section 2 for every stock and return the per-stock condition table:
    QUALIFIED = (Return > return_threshold) AND (Risk < risk_threshold) AND (Volume > average_volume)

    Args:
        stock_metrics: DataFrame containing stock metrics.
        return_threshold: Return % threshold.
        risk_threshold: Risk % threshold.

    Returns:
        DataFrame with columns ['Stock', 'Return Condition', 'Risk Condition', 'Volume Condition', 'Final Result'].
    """
    if stock_metrics is None or stock_metrics.empty:
        return pd.DataFrame(
            columns=["Stock", "Return Condition", "Risk Condition", "Volume Condition", "Final Result"]
        )

    ret_col = "Return %" if "Return %" in stock_metrics.columns else "CAGR"
    risk_col = "Risk %" if "Risk %" in stock_metrics.columns else "Volatility"
    vol_col = "Average Volume"

    avg_volume = float(stock_metrics[vol_col].mean()) if vol_col in stock_metrics.columns else 0.0

    rows = []
    for _, row in stock_metrics.iterrows():
        stock_sym = row.get("Stock", "UNKNOWN")
        ret_val = float(row.get(ret_col, 0.0))
        risk_val = float(row.get(risk_col, 0.0))
        vol_val = float(row.get(vol_col, 0.0))

        ret_cond = bool(ret_val > return_threshold)
        risk_cond = bool(risk_val < risk_threshold)
        vol_cond = bool(vol_val > avg_volume)
        final_result = bool(ret_cond and risk_cond and vol_cond)

        rows.append({
            "Stock": stock_sym,
            "Return Condition": ret_cond,
            "Risk Condition": risk_cond,
            "Volume Condition": vol_cond,
            "Final Result": final_result,
        })

    return pd.DataFrame(rows)
