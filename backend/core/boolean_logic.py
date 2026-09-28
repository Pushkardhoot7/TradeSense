"""
boolean_logic.py - Boolean Screening Rule for TradeSense V2
============================================================
Implements the Boolean stock screening formula and generates a
truth table showing each condition and the final qualified/disqualified
result for every stock.

Rule:
    QUALIFIED = (Return > Return_Threshold)
                AND (Risk < Risk_Threshold)
                AND (Volume > Average_Volume)
"""

from __future__ import annotations
from typing import Any


BOOLEAN_RULE = "QUALIFIED = (Return > Return_Threshold) AND (Risk < Risk_Threshold) AND (Volume > Average_Volume)"


def evaluate_stock_boolean(
    metric: dict,
    return_threshold: float,
    risk_threshold: float,
    avg_volume: float,
) -> dict[str, Any]:
    """
    Evaluate the three Boolean conditions for a single stock.

    Returns
    -------
    dict with keys:
        symbol, return_condition, risk_condition, volume_condition,
        final_result, return_value, risk_value, volume_value
    """
    ret_val  = float(metric.get("return_pct", 0.0))
    risk_val = float(metric.get("risk_pct",   999.0))
    vol_val  = float(metric.get("avg_volume", 0.0))

    cond_ret  = ret_val  > return_threshold
    cond_risk = risk_val < risk_threshold
    cond_vol  = vol_val  > avg_volume
    final     = cond_ret and cond_risk and cond_vol

    return {
        "symbol":           metric.get("symbol", "UNKNOWN"),
        "return_condition": cond_ret,
        "risk_condition":   cond_risk,
        "volume_condition": cond_vol,
        "final_result":     final,
        "return_value":     round(ret_val, 2),
        "risk_value":       round(risk_val, 2),
        "volume_value":     vol_val,
    }


def build_truth_table(
    metrics: list[dict],
    return_threshold: float = 5.0,
    risk_threshold: float = 20.0,
) -> dict[str, Any]:
    """
    Build a complete Boolean truth table for all stocks.

    Parameters
    ----------
    metrics          : list of per-stock metric dicts
    return_threshold : Return% threshold (default 5.0)
    risk_threshold   : Risk% threshold (default 20.0)

    Returns
    -------
    dict with:
        boolean_rule, headers, rows, qualified_stocks,
        disqualified_stocks, qualified_count, total_count
    """
    if not metrics:
        return {
            "boolean_rule":       BOOLEAN_RULE,
            "return_threshold":   return_threshold,
            "risk_threshold":     risk_threshold,
            "avg_volume":         0.0,
            "headers":            ["Stock", "Return>thresh", "Risk<thresh", "Volume>avg", "QUALIFIED"],
            "rows":               [],
            "qualified_stocks":   [],
            "disqualified_stocks": [],
            "qualified_count":    0,
            "total_count":        0,
        }

    volumes = [float(m.get("avg_volume", 0.0)) for m in metrics]
    avg_volume = sum(volumes) / len(volumes) if volumes else 0.0

    rows: list[dict] = []
    qualified:    list[str] = []
    disqualified: list[str] = []

    for m in metrics:
        row = evaluate_stock_boolean(m, return_threshold, risk_threshold, avg_volume)
        rows.append(row)
        if row["final_result"]:
            qualified.append(row["symbol"])
        else:
            disqualified.append(row["symbol"])

    return {
        "boolean_rule":       BOOLEAN_RULE,
        "return_threshold":   return_threshold,
        "risk_threshold":     risk_threshold,
        "avg_volume":         round(avg_volume, 0),
        "headers":            ["Stock", "Return>thresh", "Risk<thresh", "Volume>avg", "QUALIFIED"],
        "rows":               rows,
        "qualified_stocks":   qualified,
        "disqualified_stocks": disqualified,
        "qualified_count":    len(qualified),
        "total_count":        len(rows),
    }
