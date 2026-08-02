"""Rank-based Enrichment Score (RES) scoring method.

GSVA-like rank-based enrichment analysis. For each cell line, genes are ranked
by expression, and a running-sum statistic determines whether module genes are
coordinately up- or down-regulated relative to background.

Reference: Hänzelmann et al. (2013) GSVA: gene set variation analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _gsea_enrichment_score(
    ranked_indices: np.ndarray,
    gene_set_mask: np.ndarray,
    n_genes: int,
    n_set: int,
) -> float:
    """
    Compute GSEA-style enrichment score for one gene set and one ranking.

    Args:
        ranked_indices: indices of genes sorted by expression (descending)
        gene_set_mask: boolean array, True for genes in the set
        n_genes: total number of genes
        n_set: number of genes in the set

    Returns:
        Enrichment score (positive = set enriched at top of ranking).
    """
    if n_set == 0:
        return 0.0

    # Running sum: increment for set genes, decrement for others
    hit_increment = (n_genes - n_set)
    miss_decrement = n_set

    running_sum = 0.0
    max_es = 0.0
    min_es = 0.0

    for idx in ranked_indices:
        if gene_set_mask[idx]:
            running_sum += hit_increment
        else:
            running_sum -= miss_decrement
        if running_sum > max_es:
            max_es = running_sum
        if running_sum < min_es:
            min_es = running_sum

    # Normalize by sqrt term
    norm_factor = np.sqrt(n_set * (n_genes - n_set) / n_genes)
    if norm_factor < 1e-12:
        return 0.0

    # Use the deviation with larger absolute value
    if abs(max_es) >= abs(min_es):
        return max_es / norm_factor
    else:
        return min_es / norm_factor


def compute_res_scores(
    expression: pd.DataFrame,
    gene_module_map: dict[str, dict],
    n_modules: int = 14,
) -> pd.DataFrame:
    """
    Compute rank-based enrichment scores for each module and cell line.

    Args:
        expression: N_cells x P_genes DataFrame (z-scored)
        gene_module_map: dict gene -> {modules, ...}
        n_modules: number of modules

    Returns:
        DataFrame of shape (N_cells, n_modules) with RES scores
        (approximately standard normal distributed).
    """
    cell_ids = expression.index.tolist()
    gene_list = expression.columns.tolist()
    n_genes = len(gene_list)

    # Build boolean masks for each module
    module_masks = np.zeros((n_modules, n_genes), dtype=bool)
    module_sizes = np.zeros(n_modules, dtype=np.int32)
    for gene_idx, gene in enumerate(gene_list):
        if gene in gene_module_map:
            for mod_idx in gene_module_map[gene]["modules"]:
                module_masks[mod_idx, gene_idx] = True
                module_sizes[mod_idx] += 1

    expr_array = expression.to_numpy(dtype=np.float64)
    n_cells = len(cell_ids)

    # Compute raw ES for each cell line and module
    raw_es = np.zeros((n_cells, n_modules), dtype=np.float64)

    for cell_idx in range(n_cells):
        # Rank genes by expression (descending)
        cell_expr = expr_array[cell_idx, :]
        ranked_indices = np.argsort(-cell_expr)  # descending

        for mod_idx in range(n_modules):
            if module_sizes[mod_idx] < 3:
                continue
            raw_es[cell_idx, mod_idx] = _gsea_enrichment_score(
                ranked_indices,
                module_masks[mod_idx],
                n_genes,
                int(module_sizes[mod_idx]),
            )

    # Normal transform: rank-based to approximate standard normal
    # For each module, rank the raw ES across cell lines and apply probit
    scores = np.zeros_like(raw_es)
    for mod_idx in range(n_modules):
        col = raw_es[:, mod_idx]
        if np.allclose(col, 0):
            continue
        # Compute ranks (1-indexed, average for ties)
        from scipy.stats import rankdata
        ranks = rankdata(col, method="average")
        # Convert to quantiles in (0, 1), avoiding exactly 0 or 1
        quantiles = (ranks - 0.5) / n_cells
        quantiles = np.clip(quantiles, 1e-12, 1 - 1e-12)
        scores[:, mod_idx] = norm.ppf(quantiles)

    result = pd.DataFrame(
        scores,
        index=cell_ids,
        columns=[f"RES_{i:02d}" for i in range(n_modules)],
    )
    return result
