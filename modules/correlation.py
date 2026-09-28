"""
Correlation Module for TradeSense

Computes Pearson correlation matrices on daily returns, verifies formal matrix properties (Identity, Symmetry, Boundedness), and extracts unique off-diagonal stock correlation pairs.

Discrete Mathematics Concept:
A correlation matrix A is a square, symmetric matrix where A[i, j] represents the Pearson correlation coefficient between daily returns of stock i and stock j.
- Identity: A[i, i] = 1.0 for all stocks i.
- Symmetry: A[i, j] = A[j, i] for all i, j (A^T = A).
- Boundedness: -1.0 <= A[i, j] <= +1.0 for all i, j.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from modules.financial_metrics import calculate_daily_returns


def calculate_return_matrix(historical_data: Dict[str, Union[Dict[str, Any], pd.DataFrame]]) -> pd.DataFrame:
    """
    Build an aligned daily-returns matrix (dates × stocks) from historical data of selected stocks.
    Reuses calculate_daily_returns() from modules/financial_metrics.py to guarantee consistency.

    Args:
        historical_data: Mapping of stock ticker -> Dataframe or Data dict.

    Returns:
        DataFrame where index is Date and columns are stock symbols containing daily returns.
    """
    if not historical_data:
        return pd.DataFrame()

    returns_map = {}

    for symbol, obj in historical_data.items():
        if isinstance(obj, dict) and "data" in obj:
            df = obj["data"]
        elif isinstance(obj, pd.DataFrame):
            df = obj
        else:
            continue

        if df is None or df.empty:
            continue

        # Extract Price column (Adj Close with Close fallback)
        if "Price" in df.columns:
            price_series = df["Price"]
        elif "Adj Close" in df.columns:
            price_series = df["Adj Close"]
        elif "Close" in df.columns:
            price_series = df["Close"]
        else:
            continue

        daily_ret = calculate_daily_returns(price_series)
        if not daily_ret.empty:
            returns_map[symbol] = daily_ret

    if not returns_map:
        return pd.DataFrame()

    # Align dates by creating a combined DataFrame
    return_matrix = pd.DataFrame(returns_map).dropna(how="all").fillna(0.0)
    return return_matrix


def calculate_correlation_matrix(return_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Pearson correlation matrix A from daily returns.

    Args:
        return_matrix: Aligned daily-returns DataFrame (dates × stocks).

    Returns:
        Square symmetric Pearson correlation matrix DataFrame (stocks × stocks).
    """
    if return_matrix is None or return_matrix.empty or return_matrix.shape[1] == 0:
        return pd.DataFrame()

    corr_df = return_matrix.corr(method="pearson")
    
    # Safe writable numpy array copy
    values = corr_df.to_numpy(copy=True)
    values = np.nan_to_num(values, nan=0.0)
    np.fill_diagonal(values, 1.0)
    values = np.clip(values, -1.0, 1.0)

    # Re-wrap in DataFrame
    clean_corr = pd.DataFrame(values, index=corr_df.index, columns=corr_df.columns)
    return clean_corr


def validate_symmetry(correlation_matrix: pd.DataFrame, atol: float = 1e-6) -> bool:
    """
    Verify A[i, j] == A[j, i] (i.e. A == A^T) within floating-point tolerance.

    Args:
        correlation_matrix: Square correlation DataFrame.
        atol: Absolute tolerance for floating-point comparisons.

    Returns:
        True if matrix is symmetric within tolerance, False otherwise.
    """
    if correlation_matrix is None or correlation_matrix.empty:
        return False

    arr = correlation_matrix.to_numpy()
    if arr.shape[0] != arr.shape[1]:
        return False

    return bool(np.allclose(arr, arr.T, atol=atol))


def validate_matrix_properties(correlation_matrix: pd.DataFrame) -> Dict[str, bool]:
    """
    Verify all formal Discrete Mathematics properties of the correlation matrix:
    1. Identity on diagonal: A[i, i] == 1.0
    2. Symmetry: A == A^T
    3. Bounded range: -1.0 <= A[i, j] <= +1.0
    """
    if correlation_matrix is None or correlation_matrix.empty:
        return {"identity": False, "symmetry": False, "bounded": False, "valid": False}

    arr = correlation_matrix.to_numpy()
    is_square = arr.shape[0] == arr.shape[1]
    if not is_square:
        return {"identity": False, "symmetry": False, "bounded": False, "valid": False}

    diag = np.diagonal(arr)
    is_identity = bool(np.allclose(diag, 1.0, atol=1e-5))
    is_symmetric = validate_symmetry(correlation_matrix)
    is_bounded = bool(np.all(arr >= -1.0 - 1e-6) and np.all(arr <= 1.0 + 1e-6))

    return {
        "identity": is_identity,
        "symmetry": is_symmetric,
        "bounded": is_bounded,
        "valid": is_identity and is_symmetric and is_bounded,
    }


def get_correlation_pairs(correlation_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Return unique (i, j) stock pairs with their correlation value —
    i.e. upper triangle only, excluding diagonal and duplicate mirrored pairs.

    Args:
        correlation_matrix: Symmetric Pearson correlation matrix.

    Returns:
        DataFrame with columns ['Stock A', 'Stock B', 'Correlation'].
    """
    if correlation_matrix is None or correlation_matrix.empty:
        return pd.DataFrame(columns=["Stock A", "Stock B", "Correlation"])

    tickers = correlation_matrix.columns
    pairs = []

    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            val = float(correlation_matrix.iloc[i, j])
            pairs.append({
                "Stock A": tickers[i],
                "Stock B": tickers[j],
                "Correlation": round(val, 4)
            })

    df_pairs = pd.DataFrame(pairs)
    if not df_pairs.empty:
        df_pairs = df_pairs.sort_values(by="Correlation", ascending=False).reset_index(drop=True)

    return df_pairs


def calculate_correlation_summary_stats(correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate summary statistics alongside the correlation matrix per Section 5:
    - Number of stocks (N)
    - Matrix size (N × N)
    - Highest positive correlation pair & value
    - Lowest correlation pair (most negative/weakest & value)
    - Average pairwise correlation (mean of unique off-diagonal pairs)
    """
    if correlation_matrix is None or correlation_matrix.empty:
        return {
            "num_stocks": 0,
            "matrix_size": "0 × 0",
            "highest_pair": {"Stock A": "N/A", "Stock B": "N/A", "Correlation": 0.0},
            "lowest_pair": {"Stock A": "N/A", "Stock B": "N/A", "Correlation": 0.0},
            "avg_pairwise_corr": 0.0,
        }

    num_stocks = len(correlation_matrix.columns)
    matrix_size = f"{num_stocks} × {num_stocks}"
    df_pairs = get_correlation_pairs(correlation_matrix)

    if df_pairs.empty:
        return {
            "num_stocks": num_stocks,
            "matrix_size": matrix_size,
            "highest_pair": {"Stock A": "N/A", "Stock B": "N/A", "Correlation": 0.0},
            "lowest_pair": {"Stock A": "N/A", "Stock B": "N/A", "Correlation": 0.0},
            "avg_pairwise_corr": 0.0,
        }

    highest_row = df_pairs.iloc[0]
    lowest_row = df_pairs.iloc[-1]
    avg_corr = float(df_pairs["Correlation"].mean())

    return {
        "num_stocks": num_stocks,
        "matrix_size": matrix_size,
        "highest_pair": {
            "Stock A": highest_row["Stock A"],
            "Stock B": highest_row["Stock B"],
            "Correlation": round(float(highest_row["Correlation"]), 4),
        },
        "lowest_pair": {
            "Stock A": lowest_row["Stock A"],
            "Stock B": lowest_row["Stock B"],
            "Correlation": round(float(lowest_row["Correlation"]), 4),
        },
        "avg_pairwise_corr": round(avg_corr, 4),
    }


# Legacy compatibility functions
def compute_correlation_matrix(returns_df: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Legacy wrapper around calculate_correlation_matrix."""
    return calculate_correlation_matrix(returns_df)


def filter_high_correlations(corr_df: pd.DataFrame, threshold: float = 0.6) -> pd.DataFrame:
    """Legacy wrapper filtering correlation pairs by threshold."""
    df_pairs = get_correlation_pairs(corr_df)
    if df_pairs.empty:
        return pd.DataFrame(columns=["Stock A", "Stock B", "Correlation"])
    return df_pairs[df_pairs["Correlation"].abs() >= threshold].reset_index(drop=True)
