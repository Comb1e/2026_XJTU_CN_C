"""Ensemble fusion of multiple scoring methods (Stage 3).

M_k(c) = γ_ewm * M_k^(EWM)(c) + γ_res * M_k^(RES)(c) + γ_spca * M_k^(SPCA)(c)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def fuse_ensemble(
    ewm_scores: pd.DataFrame,
    res_scores: pd.DataFrame,
    spca_scores: pd.DataFrame,
    gamma_ewm: float = 0.50,
    gamma_res: float = 0.25,
    gamma_spca: float = 0.25,
    module_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Fuse three scoring methods into ensemble indicators.

    All input DataFrames must have the same index (cell lines) and same
    number of columns (modules).

    Args:
        ewm_scores: evidence-weighted mean scores
        res_scores: rank-based enrichment scores
        spca_scores: sparse PCA projection scores
        gamma_ewm: weight for EWM
        gamma_res: weight for RES
        gamma_spca: weight for SPCA
        module_names: optional list of module display names

    Returns:
        DataFrame of shape (N_cells, K_modules) with ensemble scores.
    """
    # Validate
    assert ewm_scores.shape == res_scores.shape == spca_scores.shape, (
        f"Shape mismatch: EWM {ewm_scores.shape}, RES {res_scores.shape}, "
        f"SPCA {spca_scores.shape}"
    )
    assert abs(gamma_ewm + gamma_res + gamma_spca - 1.0) < 1e-10, (
        f"Weights must sum to 1, got {gamma_ewm + gamma_res + gamma_spca}"
    )

    # Ensure all methods are standardized before fusion
    # (RES is already ~N(0,1); standardize EWM and SPCA to same scale)
    ewm_std = (ewm_scores - ewm_scores.mean()) / (ewm_scores.std() + 1e-12)
    res_std = res_scores  # Already standard normal
    spca_std = (spca_scores - spca_scores.mean()) / (spca_scores.std() + 1e-12)

    ensemble = (
        gamma_ewm * ewm_std.to_numpy(dtype=np.float64)
        + gamma_res * res_std.to_numpy(dtype=np.float64)
        + gamma_spca * spca_std.to_numpy(dtype=np.float64)
    )

    if module_names is None:
        columns = [f"M_{i:02d}" for i in range(ewm_scores.shape[1])]
    else:
        columns = module_names

    result = pd.DataFrame(ensemble, index=ewm_scores.index, columns=columns)
    return result
