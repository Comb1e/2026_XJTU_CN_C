"""Orthogonalization of module indicators (Stage 4).

Uses Löwdin symmetric orthogonalization to decorrelate indicators while
minimally perturbing their original biological directions.

Reference: Löwdin (1950) — symmetric orthogonalization via Σ^{-1/2}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def lowdin_orthogonalize(
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Apply Löwdin symmetric orthogonalization.

    M̃ = M_centered · Σ^{-1/2}

    This is the minimal-rotation transform that makes cov(M̃) = I.

    Args:
        scores: N_cells x K_modules DataFrame

    Returns:
        (orthogonalized_scores, transform_matrix_S)
        - orthogonalized_scores: DataFrame of same shape, cov ≈ I
        - S: K x K transform matrix (Σ^{-1/2}), with diagonal dominance
          indicating biological meaning preservation.
    """
    M = scores.to_numpy(dtype=np.float64)
    N, K = M.shape

    # Center
    M_centered = M - M.mean(axis=0, keepdims=True)

    # Covariance
    cov = (M_centered.T @ M_centered) / (N - 1)

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Handle near-zero eigenvalues for numerical stability
    eigenvalues = np.maximum(eigenvalues, 1e-12)

    # Σ^{-1/2} = V Λ^{-1/2} V^T
    S = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

    # Apply transform
    M_orth = M_centered @ S

    result = pd.DataFrame(
        M_orth,
        index=scores.index,
        columns=scores.columns,
    )
    return result, S


def partial_orthogonalize(
    scores: pd.DataFrame,
    corr_threshold: float = 0.7,
) -> pd.DataFrame:
    """
    Partially orthogonalize only highly correlated module pairs.

    For module pairs with |r| > threshold, applies pairwise
    decorrelation. Modules with low correlation are left unchanged.

    Args:
        scores: N_cells x K_modules DataFrame
        corr_threshold: only decorrelate pairs with |r| above this

    Returns:
        DataFrame with same shape, reduced correlations.
    """
    M = scores.to_numpy(dtype=np.float64).copy()
    N, K = M.shape
    M_centered = M - M.mean(axis=0, keepdims=True)

    corr = np.corrcoef(M_centered.T)

    # Decorrelate each pair with |r| > threshold
    # Use a sequential approach: for each high-correlation pair,
    # replace the second variable with its residual from the first.
    processed = np.zeros(K, dtype=bool)

    for i in range(K):
        for j in range(i + 1, K):
            if abs(corr[i, j]) > corr_threshold and not processed[j]:
                # Regress column j on column i, keep residual
                x = M_centered[:, i]
                y = M_centered[:, j]
                beta = np.dot(x, y) / (np.dot(x, x) + 1e-12)
                residual = y - beta * x
                # Standardize residual
                residual_std = residual.std()
                if residual_std > 1e-12:
                    residual = residual / residual_std
                M_centered[:, j] = residual
                processed[j] = True

    result = pd.DataFrame(
        M_centered,
        index=scores.index,
        columns=scores.columns,
    )
    return result


def orthogonalize(
    scores: pd.DataFrame,
    method: str = "lowdin",
    corr_threshold: float = 0.7,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """
    Orthogonalize module indicators.

    Args:
        scores: N_cells x K_modules DataFrame
        method: "lowdin", "partial", "zca", or "none"
        corr_threshold: for partial mode only

    Returns:
        (orthogonalized_scores, transform_matrix)
        transform_matrix is None for partial/none methods.
    """
    if method == "none":
        return scores, None
    elif method == "lowdin":
        return lowdin_orthogonalize(scores)
    elif method == "zca":
        # ZCA whitening: M̃ = M_centered · Σ^{-1/2} (same as Löwdin)
        return lowdin_orthogonalize(scores)
    elif method == "partial":
        result = partial_orthogonalize(scores, corr_threshold)
        return result, None
    else:
        raise ValueError(f"Unknown orthogonalization method: {method}")
