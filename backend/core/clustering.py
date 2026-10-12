"""
clustering.py
-------------
Correlation-based stock clustering and co-movement analysis for TradeSense V2.

Implements rigorous correlation clustering algorithms:
1. Mantegna Metric Distance:
   d(u, v) = sqrt(2 * (1 - r_uv))
   Maps Pearson r in [-1, 1] to a true metric distance in [0, 2].
   d = 0 for r = 1 (identical co-movement)
   d = sqrt(2) ~ 1.414 for r = 0 (uncorrelated)
   d = 2 for r = -1 (perfect inverse co-movement)

2. Threshold-Based Co-Movement Clustering:
   Partitions the universe into genuine multi-stock clusters (size >= 2)
   of mutually co-moving stocks where pairwise positive correlation r >= theta,
   while strictly separating unclustered singletons (size = 1) and
   documenting within-cluster correlation metrics.

3. Hierarchical Agglomerative Clustering (Average Linkage):
   Constructs a hierarchical clustering hierarchy from scratch
   based on the Mantegna correlation distance.
"""

from __future__ import annotations

import math
from typing import Any
import pandas as pd
import numpy as np


def correlation_to_distance(r: float) -> float:
    """Convert Pearson correlation to Mantegna metric distance.

    d(u, v) = sqrt(2 * max(0.0, 1.0 - r))
    """
    if math.isnan(r):
        return 2.0
    # Clamp r to [-1.0, 1.0]
    r_clamped = max(-1.0, min(1.0, r))
    return math.sqrt(max(0.0, 2.0 * (1.0 - r_clamped)))


def distance_to_correlation(d: float) -> float:
    """Convert Mantegna metric distance back to Pearson correlation.

    r = 1 - (d^2) / 2
    """
    d_clamped = max(0.0, min(2.0, d))
    return 1.0 - (d_clamped ** 2) / 2.0


def compute_distance_matrix(corr_df: pd.DataFrame) -> pd.DataFrame:
    """Compute square Mantegna distance matrix from correlation DataFrame."""
    tickers = list(corr_df.columns)
    n = len(tickers)
    dist_vals = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                dist_vals[i, j] = 0.0
            elif j < i:
                dist_vals[i, j] = dist_vals[j, i]
            else:
                r = float(corr_df.iloc[i, j])
                d = correlation_to_distance(r)
                dist_vals[i, j] = d
                dist_vals[j, i] = d

    return pd.DataFrame(dist_vals, index=tickers, columns=tickers)


def cluster_by_correlation_threshold(
    tickers: list[str],
    corr_df: pd.DataFrame,
    threshold: float = 0.70,
    min_cluster_size: int = 2,
) -> dict[str, Any]:
    """Identify clusters of mutually co-moving stocks based on positive correlation.

    Rule:
    An edge (u, v) exists if and only if r(u, v) >= threshold (positive co-movement).
    Components of size >= min_cluster_size form valid multi-stock clusters.
    Components of size 1 are categorized as unclustered singletons.

    Parameters
    ----------
    tickers:
        Validated universe of tickers.
    corr_df:
        Square Pearson correlation DataFrame.
    threshold:
        Cutoff for positive correlation (default 0.70).
    min_cluster_size:
        Minimum stocks to qualify as a cluster (default 2).

    Returns
    -------
    dict with:
        - clusters: list of multi-stock clusters with within-group stats
        - unclustered: list of singleton stocks with explanation
        - total_stocks: int
        - clustered_stocks_count: int
        - unclustered_stocks_count: int
        - avg_cluster_size: float
    """
    if corr_df is None or corr_df.empty or not tickers:
        return {
            "clusters": [],
            "unclustered": [{"ticker": t, "reason": "No correlation data available"} for t in tickers],
            "total_stocks": len(tickers),
            "clustered_stocks_count": 0,
            "unclustered_stocks_count": len(tickers),
            "cluster_count": 0,
            "threshold": threshold,
        }

    # Filter to tickers present in corr_df
    valid_tickers = [t for t in tickers if t in corr_df.columns]
    
    # Build positive correlation adjacency dict
    adj: dict[str, list[tuple[str, float]]] = {t: [] for t in valid_tickers}
    n = len(valid_tickers)

    for i in range(n):
        u = valid_tickers[i]
        for j in range(i + 1, n):
            v = valid_tickers[j]
            val = float(corr_df.loc[u, v])
            if not math.isnan(val) and val >= threshold:
                adj[u].append((v, val))
                adj[v].append((u, val))

    # Connected components using BFS
    visited: set[str] = set()
    components: list[list[str]] = []

    for start in valid_tickers:
        if start in visited:
            continue
        comp: list[str] = []
        queue = [start]
        visited.add(start)
        while queue:
            curr = queue.pop(0)
            comp.append(curr)
            for neighbor, _ in adj.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        components.append(sorted(comp))

    # Separate multi-stock clusters from unclustered singletons
    multi_clusters: list[list[str]] = [c for c in components if len(c) >= min_cluster_size]
    singletons: list[str] = [c[0] for c in components if len(c) < min_cluster_size]

    # Sort multi-clusters by descending size
    multi_clusters.sort(key=len, reverse=True)

    formatted_clusters: list[dict[str, Any]] = []
    for idx, members in enumerate(multi_clusters, start=1):
        # Calculate within-cluster pairwise correlations
        pairwise_corrs: list[float] = []
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                c_val = float(corr_df.loc[members[i], members[j]])
                if not math.isnan(c_val):
                    pairwise_corrs.append(c_val)

        avg_corr = sum(pairwise_corrs) / len(pairwise_corrs) if pairwise_corrs else threshold
        min_corr = min(pairwise_corrs) if pairwise_corrs else threshold
        max_corr = max(pairwise_corrs) if pairwise_corrs else threshold

        # Find most central stock in cluster (highest degree inside cluster)
        deg_map = {m: sum(1 for nb, _ in adj.get(m, []) if nb in members) for m in members}
        lead_stock = max(deg_map, key=lambda m: deg_map[m]) if deg_map else members[0]

        formatted_clusters.append({
            "cluster_id": idx,
            "name": f"Co-Movement Cluster #{idx}",
            "size": len(members),
            "members": members,
            "lead_stock": lead_stock,
            "avg_correlation": round(avg_corr, 4),
            "min_correlation": round(min_corr, 4),
            "max_correlation": round(max_corr, 4),
            "cohesion_level": "Very Strong" if avg_corr >= 0.75 else "Strong",
            "description": (
                f"Stocks in this cluster move together with average correlation r = {avg_corr:.2f}. "
                f"Led by {lead_stock.replace('.NS', '')} with {deg_map.get(lead_stock, 0)} internal connections."
            ),
        })

    # Formatted singletons
    formatted_unclustered: list[dict[str, Any]] = []
    for s in sorted(singletons):
        # Check max correlation with any other stock in universe
        other_corrs = [
            float(corr_df.loc[s, o])
            for o in valid_tickers
            if o != s and not math.isnan(float(corr_df.loc[s, o]))
        ]
        max_r = max(other_corrs) if other_corrs else 0.0

        formatted_unclustered.append({
            "ticker": s,
            "max_correlation": round(max_r, 4),
            "reason": f"No qualifying pair meets r >= {threshold:.2f} (highest observed: r = {max_r:.2f})",
        })

    total_valid = len(valid_tickers)
    total_clustered = sum(len(c["members"]) for c in formatted_clusters)

    return {
        "threshold": threshold,
        "total_stocks": total_valid,
        "cluster_count": len(formatted_clusters),
        "clustered_stocks_count": total_clustered,
        "unclustered_stocks_count": len(formatted_unclustered),
        "clusters": formatted_clusters,
        "unclustered": formatted_unclustered,
        "method": "Correlation Threshold Co-Movement Graph",
        "distance_metric": "Mantegna Metric Distance d = sqrt(2(1 - r))",
    }


def hierarchical_clustering_average_linkage(
    tickers: list[str],
    corr_df: pd.DataFrame,
    distance_cutoff: float = 0.775, # d = sqrt(2*(1 - 0.70)) ~ 0.775
) -> dict[str, Any]:
    """Agglomerative Hierarchical Clustering using Average Linkage (UPGMA).

    Implemented from scratch for complete algorithmic transparency.

    Parameters
    ----------
    tickers:
        Ordered list of tickers.
    corr_df:
        Square correlation DataFrame.
    distance_cutoff:
        Distance threshold below which clusters are merged.
        Distance d = sqrt(2*(1-r)).
        r = 0.70 -> d = sqrt(2*0.30) ~ 0.7746
        r = 0.60 -> d = sqrt(2*0.40) ~ 0.8944
        r = 0.50 -> d = 1.0000

    Returns
    -------
    dict with:
        - clusters: list of clusters formed at cutoff
        - unclustered: singletons
        - merge_history: step-by-step linkage trace
    """
    valid_tickers = [t for t in tickers if t in corr_df.columns]
    n = len(valid_tickers)
    if n < 2:
        return cluster_by_correlation_threshold(valid_tickers, corr_df)

    dist_df = compute_distance_matrix(corr_df.loc[valid_tickers, valid_tickers])

    # Initial state: each ticker in its own cluster
    clusters: dict[int, list[str]] = {i: [valid_tickers[i]] for i in range(n)}
    # Active cluster IDs
    active_ids = set(range(n))

    # Cluster distances cache
    cluster_dist: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i + 1, n):
            cluster_dist[(i, j)] = float(dist_df.iloc[i, j])

    next_id = n
    merge_history = []

    while len(active_ids) > 1:
        # Find closest pair of active clusters
        best_pair = None
        min_dist = float("inf")

        active_list = sorted(active_ids)
        for i_idx in range(len(active_list)):
            c_a = active_list[i_idx]
            for j_idx in range(i_idx + 1, len(active_list)):
                c_b = active_list[j_idx]
                key = (min(c_a, c_b), max(c_a, c_b))
                d = cluster_dist.get(key, float("inf"))
                if d < min_dist:
                    min_dist = d
                    best_pair = (c_a, c_b)

        if best_pair is None or min_dist > distance_cutoff:
            break

        c_1, c_2 = best_pair
        members_new = clusters[c_1] + clusters[c_2]
        new_id = next_id
        next_id += 1

        clusters[new_id] = members_new
        active_ids.remove(c_1)
        active_ids.remove(c_2)

        # Compute average distance from new cluster to other active clusters
        for other in active_ids:
            # Average distance between members
            total_d = sum(
                float(dist_df.loc[m1, m2])
                for m1 in members_new
                for m2 in clusters[other]
            )
            count = len(members_new) * len(clusters[other])
            avg_d = total_d / count if count > 0 else 2.0
            cluster_dist[(min(new_id, other), max(new_id, other))] = avg_d

        active_ids.add(new_id)
        implied_r = distance_to_correlation(min_dist)
        merge_history.append({
            "merged": [c_1, c_2],
            "new_id": new_id,
            "distance": round(min_dist, 4),
            "implied_correlation": round(implied_r, 4),
            "cluster_size": len(members_new),
        })

    # Partition resulting clusters
    final_clusters = [clusters[cid] for cid in active_ids if len(clusters[cid]) >= 2]
    final_singletons = [clusters[cid][0] for cid in active_ids if len(clusters[cid]) == 1]

    final_clusters.sort(key=len, reverse=True)

    formatted_clusters = []
    for idx, members in enumerate(final_clusters, start=1):
        pw_corrs = [
            float(corr_df.loc[members[i], members[j]])
            for i in range(len(members))
            for j in range(i + 1, len(members))
            if not math.isnan(float(corr_df.loc[members[i], members[j]]))
        ]
        avg_r = sum(pw_corrs) / len(pw_corrs) if pw_corrs else 0.70
        formatted_clusters.append({
            "cluster_id": idx,
            "name": f"Hierarchical Cluster #{idx}",
            "size": len(members),
            "members": sorted(members),
            "lead_stock": members[0],
            "avg_correlation": round(avg_r, 4),
            "min_correlation": round(min(pw_corrs), 4) if pw_corrs else round(avg_r, 4),
            "max_correlation": round(max(pw_corrs), 4) if pw_corrs else round(avg_r, 4),
            "description": f"Agglomerative cluster linked at distance d <= {distance_cutoff:.3f} (avg r = {avg_r:.2f})",
        })

    return {
        "distance_cutoff": distance_cutoff,
        "implied_threshold": round(distance_to_correlation(distance_cutoff), 2),
        "cluster_count": len(formatted_clusters),
        "clustered_stocks_count": sum(len(c["members"]) for c in formatted_clusters),
        "unclustered_stocks_count": len(final_singletons),
        "clusters": formatted_clusters,
        "unclustered": sorted(final_singletons),
        "merge_history": merge_history,
    }
