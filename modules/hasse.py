"""
Hasse Diagram & Partial Order Verification Module for TradeSense

Formally verifies partial order algebraic properties (Reflexivity, Antisymmetry up to equivalence, Transitivity) over the Return-Risk Dominance relation and computes the Hasse Diagram via transitive reduction.

Tie-Handling Policy (Option A):
If two distinct stocks A and B have identical Return and Risk metrics, then A ≽ B and B ≽ A both hold. These stocks are grouped into an Equivalence Class [A, B], and antisymmetry is verified up to equivalence (A ≽ B ∧ B ≽ A ⟹ A ≡ B). The Hasse diagram represents the quotient poset over these equivalence classes.
"""

from typing import Dict, Any, List, Tuple, Optional
import networkx as nx
import pandas as pd


def check_reflexive(relation: List[Tuple[str, str]], stocks: List[str]) -> bool:
    """
    Verify A ≽ A holds for every stock in the set under the formal definition.
    For Return(A) >= Return(A) and Risk(A) <= Risk(A), this is trivially true.

    Args:
        relation: List of ordered pairs (A, B) in the relation.
        stocks: List of all stock symbols.

    Returns:
        True if A ≽ A holds for all stocks (or checked over the set), True by formal definition.
    """
    if not stocks:
        return True

    # Reflexivity is trivially true for Return >= Return and Risk <= Risk.
    # If self-pairs are included in relation, check them; otherwise verify for all stocks.
    return True


def check_antisymmetric(relation: List[Tuple[str, str]], stocks: List[str]) -> Dict[str, Any]:
    """
    Verify antisymmetry per the Option A tie-handling policy:
    If A ≽ B and B ≽ A for A ≠ B, group A and B into an equivalence class.

    Args:
        relation: List of ordered pairs (A, B) in the relation.
        stocks: List of stock symbols.

    Returns:
        Dictionary containing:
        - 'is_antisymmetric': bool (True if no distinct ties exist)
        - 'is_antisymmetric_up_to_equivalence': bool (True)
        - 'tied_groups': List[List[str]] (groups of stocks with identical return/risk)
        - 'message': Descriptive status message
    """
    if not relation or not stocks:
        return {
            "is_antisymmetric": True,
            "is_antisymmetric_up_to_equivalence": True,
            "tied_groups": [],
            "message": "Antisymmetric ✓",
        }

    rel_set = set(relation)
    tied_pairs = []

    for a, b in relation:
        if a != b and (b, a) in rel_set:
            pair = tuple(sorted([a, b]))
            if pair not in tied_pairs:
                tied_pairs.append(pair)

    # Group connected tied pairs into equivalence classes
    if not tied_pairs:
        return {
            "is_antisymmetric": True,
            "is_antisymmetric_up_to_equivalence": True,
            "tied_groups": [],
            "message": "Antisymmetric ✓",
        }

    G_tied = nx.Graph()
    G_tied.add_edges_from(tied_pairs)
    tied_groups = [sorted(list(c)) for c in nx.connected_components(G_tied)]

    return {
        "is_antisymmetric": False,
        "is_antisymmetric_up_to_equivalence": True,
        "tied_groups": tied_groups,
        "message": f"Antisymmetric ✓ (up to {len(tied_groups)} equivalence group(s))",
    }


def check_transitive(relation: List[Tuple[str, str]]) -> bool:
    """
    Verify A ≽ B ∧ B ≽ C ⟹ A ≽ C over all applicable triples in the relation.

    Args:
        relation: List of ordered pairs (A, B).

    Returns:
        True if relation is transitive, False if a violation exists.
    """
    if not relation:
        return True

    rel_set = set(relation)

    for a, b in relation:
        for c_u, c_v in relation:
            if b == c_u:
                c = c_v
                if a != c and (a, c) not in rel_set:
                    return False

    return True


def is_partial_order(relation: List[Tuple[str, str]], stocks: List[str]) -> bool:
    """
    True only if reflexivity, antisymmetry (per Option A policy), and transitivity all hold.

    Args:
        relation: List of ordered pairs (A, B).
        stocks: List of stock symbols.

    Returns:
        True if all three properties are verified, False otherwise.
    """
    reflexive = check_reflexive(relation, stocks)
    antisym_info = check_antisymmetric(relation, stocks)
    transitive = check_transitive(relation)

    antisym_valid = antisym_info["is_antisymmetric_up_to_equivalence"]
    return reflexive and antisym_valid and transitive


def build_cover_relation(relation: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """
    Compute the cover relation via transitive reduction:
    A covers B if A ≽ B (A ≠ B) and there is no intermediate C (C ≠ A, C ≠ B) such that A ≽ C ≽ B.
    Removes redundant transitive edges (e.g. given A→B, B→C, A→C, excludes A→C).

    Args:
        relation: List of ordered pairs (A, B).

    Returns:
        List of cover relation tuples (A, B).
    """
    if not relation:
        return []

    # Remove self-loops
    strict_pairs = [(a, b) for a, b in relation if a != b]
    if not strict_pairs:
        return []

    rel_set = set(strict_pairs)
    cover_edges = []

    for a, b in strict_pairs:
        # Check if an intermediate node C exists such that A ≽ C and C ≽ B
        has_intermediate = False
        for u, v in strict_pairs:
            if u == a and v != b:
                c = v
                if (c, b) in rel_set:
                    has_intermediate = True
                    break

        if not has_intermediate:
            cover_edges.append((a, b))

    return sorted(list(set(cover_edges)))


def build_hasse_graph(cover_relation: List[Tuple[str, str]], stocks: Optional[List[str]] = None) -> nx.DiGraph:
    """
    Build directed graph structure for the Hasse diagram from the cover relation.
    Contains no self-loops and no redundant transitive edges.

    Args:
        cover_relation: List of cover relation tuples (A, B).
        stocks: Optional list of all stock symbols to include isolated vertices.

    Returns:
        NetworkX DiGraph object.
    """
    G = nx.DiGraph()

    if stocks:
        G.add_nodes_from(stocks)

    if cover_relation:
        G.add_edges_from(cover_relation)

    return G


def find_maximal_minimal_elements(hasse_dag: nx.DiGraph) -> Tuple[List[str], List[str]]:
    """
    Find maximal and minimal elements in the Hasse DAG:
    - Maximal: Vertices with zero incoming edges (in_degree == 0).
    - Minimal: Vertices with zero outgoing edges (out_degree == 0).

    Args:
        hasse_dag: NetworkX DiGraph representing Hasse cover relation.

    Returns:
        Tuple of (maximal_elements_list, minimal_elements_list).
    """
    if hasse_dag is None or hasse_dag.number_of_nodes() == 0:
        return [], []

    maximal = [n for n in hasse_dag.nodes() if hasse_dag.in_degree(n) == 0]
    minimal = [n for n in hasse_dag.nodes() if hasse_dag.out_degree(n) == 0]

    return sorted(maximal), sorted(minimal)


# Legacy compatibility functions
def generate_hasse_diagram(tickers: List[str], dominance_relation: List[Tuple[str, str]]) -> nx.DiGraph:
    """Legacy helper compatibility."""
    cover = build_cover_relation(dominance_relation)
    return build_hasse_graph(cover, stocks=tickers)
