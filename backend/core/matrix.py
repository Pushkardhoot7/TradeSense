"""
matrix.py
---------
Correlation matrix construction and validation for TradeSense V2.

All matrix computations are implemented from scratch using pandas /
numpy arithmetic — no scipy.stats or sklearn wrappers.

Key functions
-------------
build_returns_matrix         -- align daily returns for N stocks
calculate_pearson_correlation -- Pearson r matrix from scratch
validate_matrix_properties   -- symmetry, diagonal-1, bounded checks
get_correlation_stats        -- summary statistics
extract_upper_triangle       -- non-redundant pair list
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import numpy as np

from .metrics import calculate_daily_returns


# ---------------------------------------------------------------------------
# Build aligned returns matrix
# ---------------------------------------------------------------------------

def build_returns_matrix(
    data_map: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Align daily returns for all stocks into a single DataFrame.

    Stocks with fewer data points cause rows to be NaN on those dates;
    the caller should decide whether to drop or forward-fill.

    Parameters
    ----------
    data_map:
        Mapping of ``{symbol: price_df}`` where each DataFrame has a
        ``Close`` column and a ``DatetimeIndex``.

    Returns
    -------
    pd.DataFrame
        Columns = ticker symbols, index = union of all trading dates.
        Values are simple daily returns.  Leading NaN per column is
        expected (no day-0 return).
    """
    series_dict: dict[str, pd.Series] = {}
    for symbol, df in data_map.items():
        close = df["Close"].dropna()
        series_dict[symbol] = calculate_daily_returns(close)

    if not series_dict:
        return pd.DataFrame()

    returns_df = pd.DataFrame(series_dict)
    returns_df.index.name = "Date"
    return returns_df


# ---------------------------------------------------------------------------
# Pearson correlation from scratch
# ---------------------------------------------------------------------------

def calculate_pearson_correlation(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Pearson correlation matrix from a returns DataFrame.

    Implementation uses the standard formula:

        r(X, Y) = cov(X, Y) / (std(X) * std(Y))

    where cov and std are computed over the pairwise complete observations
    (rows where both X and Y are non-NaN).

    Parameters
    ----------
    returns_df:
        DataFrame of daily returns (columns = ticker symbols).

    Returns
    -------
    pd.DataFrame
        Square correlation matrix with tickers as both index and columns.
        Diagonal values are exactly 1.0; off-diagonals are Pearson r.
    """
    tickers: list[str] = list(returns_df.columns)
    n: int = len(tickers)
    corr_matrix: list[list[float]] = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i == j:
                corr_matrix[i][j] = 1.0
                continue
            if j < i:
                # Matrix is symmetric — reuse already-computed value
                corr_matrix[i][j] = corr_matrix[j][i]
                continue

            # Pairwise complete observations
            xi: pd.Series = returns_df[tickers[i]]
            xj: pd.Series = returns_df[tickers[j]]
            valid_mask = xi.notna() & xj.notna()
            xi_clean = xi[valid_mask].values.astype(float)
            xj_clean = xj[valid_mask].values.astype(float)

            k = len(xi_clean)
            if k < 2:
                corr_matrix[i][j] = float("nan")
                continue

            mean_i = xi_clean.mean()
            mean_j = xj_clean.mean()
            diff_i = xi_clean - mean_i
            diff_j = xj_clean - mean_j

            cov = (diff_i * diff_j).sum() / (k - 1)
            std_i = math.sqrt((diff_i ** 2).sum() / (k - 1))
            std_j = math.sqrt((diff_j ** 2).sum() / (k - 1))

            if std_i == 0 or std_j == 0:
                corr_matrix[i][j] = float("nan")
            else:
                corr_matrix[i][j] = cov / (std_i * std_j)

    return pd.DataFrame(corr_matrix, index=tickers, columns=tickers)


# ---------------------------------------------------------------------------
# Matrix property validation
# ---------------------------------------------------------------------------

def validate_matrix_properties(corr_df: pd.DataFrame) -> dict[str, bool]:
    """Validate that a correlation matrix satisfies its mathematical properties.

    Checks
    ------
    1. **Diagonal = 1**: A[i, i] == 1.0 for all i.
    2. **Symmetry**: A[i, j] == A[j, i] for all i, j.
    3. **Bounded**: -1 <= A[i, j] <= 1 for all i, j.

    Parameters
    ----------
    corr_df:
        Square correlation DataFrame (tickers as index and columns).

    Returns
    -------
    dict with keys:
        * ``is_diagonal_one``  – bool
        * ``is_symmetric``     – bool
        * ``is_bounded``       – bool
        * ``all_valid``        – True iff all three checks pass
    """
    values: np.ndarray = corr_df.values.astype(float)
    n: int = values.shape[0]

    # Check 1 — diagonal
    diag_ok: bool = all(
        abs(values[i, i] - 1.0) < 1e-9
        for i in range(n)
        if not math.isnan(values[i, i])
    )

    # Check 2 — symmetry
    sym_ok: bool = True
    for i in range(n):
        for j in range(i + 1, n):
            a = values[i, j]
            b = values[j, i]
            if math.isnan(a) and math.isnan(b):
                continue
            if math.isnan(a) or math.isnan(b) or abs(a - b) > 1e-9:
                sym_ok = False
                break
        if not sym_ok:
            break

    # Check 3 — bounded [-1, 1]
    bounded_ok: bool = bool(
        np.all(
            (np.isnan(values)) | ((values >= -1.0 - 1e-9) & (values <= 1.0 + 1e-9))
        )
    )

    all_valid: bool = diag_ok and sym_ok and bounded_ok

    return {
        "is_diagonal_one": diag_ok,
        "is_symmetric": sym_ok,
        "is_bounded": bounded_ok,
        "all_valid": all_valid,
    }


# ---------------------------------------------------------------------------
# Correlation summary statistics
# ---------------------------------------------------------------------------

def get_correlation_stats(corr_df: pd.DataFrame) -> dict[str, Any]:
    """Compute summary statistics over all unique off-diagonal pairs.

    Parameters
    ----------
    corr_df:
        Square correlation DataFrame.

    Returns
    -------
    dict with keys:
        * ``max_corr``   – float (highest correlation)
        * ``min_corr``   – float (lowest correlation)
        * ``mean_corr``  – float (mean of all off-diagonal pairs)
        * ``max_pair``   – tuple[str, str] (stocks with highest correlation)
        * ``min_pair``   – tuple[str, str] (stocks with lowest correlation)
    """
    tickers: list[str] = list(corr_df.columns)
    n: int = len(tickers)

    pairs: list[tuple[float, str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            val: float = float(corr_df.iloc[i, j])
            if not math.isnan(val):
                pairs.append((val, tickers[i], tickers[j]))

    if not pairs:
        return {
            "max_corr": float("nan"),
            "min_corr": float("nan"),
            "mean_corr": float("nan"),
            "max_pair": ("N/A", "N/A"),
            "min_pair": ("N/A", "N/A"),
        }

    vals: list[float] = [p[0] for p in pairs]
    max_val: float = max(vals)
    min_val: float = min(vals)
    mean_val: float = sum(vals) / len(vals)

    max_pair = next((p[1], p[2]) for p in pairs if p[0] == max_val)
    min_pair = next((p[1], p[2]) for p in pairs if p[0] == min_val)

    return {
        "max_corr": round(max_val, 6),
        "min_corr": round(min_val, 6),
        "mean_corr": round(mean_val, 6),
        "max_pair": max_pair,
        "min_pair": min_pair,
    }


# ---------------------------------------------------------------------------
# Upper triangle extraction
# ---------------------------------------------------------------------------

def extract_upper_triangle(corr_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Extract unique off-diagonal correlation pairs from the upper triangle.

    Parameters
    ----------
    corr_df:
        Square correlation DataFrame.

    Returns
    -------
    list[dict]
        Each element: ``{'stock_a': str, 'stock_b': str, 'correlation': float}``.
        Pairs with NaN correlation are included with value ``None``.
        Sorted by descending absolute correlation.
    """
    tickers: list[str] = list(corr_df.columns)
    n: int = len(tickers)
    result: list[dict[str, Any]] = []

    for i in range(n):
        for j in range(i + 1, n):
            val: float = float(corr_df.iloc[i, j])
            result.append(
                {
                    "stock_a": tickers[i],
                    "stock_b": tickers[j],
                    "correlation": None if math.isnan(val) else round(val, 6),
                }
            )

    # Sort by descending absolute correlation (NaN last)
    result.sort(
        key=lambda x: abs(x["correlation"]) if x["correlation"] is not None else -1,
        reverse=True,
    )
    return result
