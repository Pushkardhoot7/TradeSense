"""
Relations Module for TradeSense

Defines the mathematical Return-Risk Dominance binary relation ≽ over the stock set:
Stock A ≽ Stock B iff Return(A) >= Return(B) AND Risk(A) <= Risk(B)

Discrete Mathematics Framing:
The relation ≽ constitutes a strict partial order on the set of stocks.
- Reflexive: A ≽ A (trivial self-dominance).
- Antisymmetric: If A ≽ B and B ≽ A, then Return(A) = Return(B) and Risk(A) = Risk(B).
- Transitive: If A ≽ B and B ≽ C, then A ≽ C.
- Incomparability: Pairs where neither A ≽ B nor B ≽ A holds are mathematically incomparable.

This relation represents a defined formal binary structure, not an investment recommendation.
"""

from typing import Dict, Any, List, Tuple, Optional
import pandas as pd


def dominates(stock_a: Dict[str, Any], stock_b: Dict[str, Any]) -> bool:
    """
    Return True if stock_a ≽ stock_b per the definition:
    Return(A) >= Return(B) AND Risk(A) <= Risk(B).

    At least one inequality must be strict for non-trivial dominance between distinct stocks.

    Args:
        stock_a: Metric dictionary or series for Stock A containing 'Return %' and 'Risk %'.
        stock_b: Metric dictionary or series for Stock B containing 'Return %' and 'Risk %'.

    Returns:
        True if stock_a dominates stock_b under the relation, False otherwise.
    """
    if not stock_a or not stock_b:
        return False

    ret_a = float(stock_a.get("Return %", stock_a.get("CAGR", 0.0)))
    risk_a = float(stock_a.get("Risk %", stock_a.get("Volatility", 0.0)))

    ret_b = float(stock_b.get("Return %", stock_b.get("CAGR", 0.0)))
    risk_b = float(stock_b.get("Risk %", stock_b.get("Volatility", 0.0)))

    return bool(ret_a >= ret_b and risk_a <= risk_b)


def build_dominance_relation(stock_metrics: pd.DataFrame) -> List[Tuple[str, str]]:
    """
    Evaluate the relation over every ordered pair of distinct stocks and return it
    as a list of ordered pairs (A, B) meaning A ≽ B.
    Excludes trivial self-pairs (A ≽ A).

    Args:
        stock_metrics: DataFrame where rows are stocks and columns contain 'Stock', 'Return %', 'Risk %'.

    Returns:
        List of tuples (stock_A, stock_B) representing ordered pairs in the relation.
    """
    if stock_metrics is None or stock_metrics.empty:
        return []

    # Format DataFrame rows as dictionaries
    if "Stock" in stock_metrics.columns:
        metrics_dict = {row["Stock"]: row.to_dict() for _, row in stock_metrics.iterrows()}
    else:
        metrics_dict = {idx: row.to_dict() for idx, row in stock_metrics.iterrows()}

    tickers = list(metrics_dict.keys())
    relation_pairs = []

    for i in range(len(tickers)):
        for j in range(len(tickers)):
            if i == j:
                continue
            sym_a = tickers[i]
            sym_b = tickers[j]

            if dominates(metrics_dict[sym_a], metrics_dict[sym_b]):
                relation_pairs.append((sym_a, sym_b))

    return relation_pairs


def get_dominated_stocks(relation: List[Tuple[str, str]], stock: str) -> List[str]:
    """
    Return all stocks dominated by the given stock (out-neighbors in the dominance DAG).

    Args:
        relation: List of ordered pairs (A, B) meaning A ≽ B.
        stock: Target stock symbol A.

    Returns:
        Sorted list of stock symbols B such that A ≽ B.
    """
    if not relation or not stock:
        return []

    dominated = [b for (a, b) in relation if a == stock]
    return sorted(list(set(dominated)))


def get_non_dominated_stocks(relation: List[Tuple[str, str]], all_stocks: List[str]) -> List[str]:
    """
    Return stocks that are not dominated by any other stock in the set
    (i.e. elements with zero incoming dominance edges).

    Args:
        relation: List of ordered pairs (A, B) meaning A ≽ B.
        all_stocks: Complete list of selected stock symbols.

    Returns:
        Sorted list of non-dominated stock symbols (Pareto undominated leaders).
    """
    if not all_stocks:
        return []

    dominated_set = {b for (a, b) in relation}
    non_dominated = [s for s in all_stocks if s not in dominated_set]
    return sorted(non_dominated)


def get_incomparable_pairs(relation: List[Tuple[str, str]], all_stocks: List[str]) -> List[Tuple[str, str]]:
    """
    Return unique pairs of distinct stocks (A, B) where neither A ≽ B nor B ≽ A holds under the relation.

    Args:
        relation: List of ordered pairs (A, B).
        all_stocks: Complete list of stock symbols.

    Returns:
        Sorted list of unique unordered tuples (A, B) with A < B.
    """
    if not all_stocks or len(all_stocks) < 2:
        return []

    relation_set = set(relation)
    incomparable = []

    for i in range(len(all_stocks)):
        for j in range(i + 1, len(all_stocks)):
            sym_a = all_stocks[i]
            sym_b = all_stocks[j]

            has_a_dom_b = (sym_a, sym_b) in relation_set
            has_b_dom_a = (sym_b, sym_a) in relation_set

            if not has_a_dom_b and not has_b_dom_a:
                pair = (sym_a, sym_b) if sym_a < sym_b else (sym_b, sym_a)
                incomparable.append(pair)

    return sorted(list(set(incomparable)))


def get_most_dominant_stocks(relation: List[Tuple[str, str]], all_stocks: List[str]) -> List[Dict[str, Any]]:
    """
    Return stocks ranked by out-degree (number of other stocks they dominate under the relation).

    Args:
        relation: List of ordered pairs (A, B).
        all_stocks: Complete list of stock symbols.

    Returns:
        List of dictionaries with keys ['stock', 'out_degree', 'dominated_stocks'].
    """
    out_degree_map = {s: [] for s in all_stocks}
    for a, b in relation:
        if a in out_degree_map:
            out_degree_map[a].append(b)

    ranked = []
    for stock, dom_list in out_degree_map.items():
        ranked.append({
            "stock": stock,
            "out_degree": len(dom_list),
            "dominated_stocks": sorted(dom_list),
        })

    ranked_sorted = sorted(ranked, key=lambda x: x["out_degree"], reverse=True)
    return ranked_sorted


# Legacy compatibility functions
def build_poset_relation(metrics_df: pd.DataFrame) -> List[Tuple[str, str]]:
    """Legacy helper compatibility."""
    return build_dominance_relation(metrics_df)


def analyze_poset_properties(tickers: List[str], relation_pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Legacy helper compatibility."""
    rel_set = set(relation_pairs)
    is_irreflexive = all((t, t) not in rel_set for t in tickers)
    is_antisymmetric = all(not ((a, b) in rel_set and (b, a) in rel_set) for a, b in relation_pairs if a != b)
    
    is_transitive = True
    for a, b in relation_pairs:
        for c in tickers:
            if (b, c) in rel_set and (a, c) not in rel_set:
                is_transitive = False
                break

    return {
        "is_irreflexive": is_irreflexive,
        "is_antisymmetric": is_antisymmetric,
        "is_transitive": is_transitive,
        "is_strict_partial_order": is_irreflexive and is_antisymmetric and is_transitive,
    }
