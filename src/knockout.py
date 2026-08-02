"""Gene knockout response computation (Stage 5).

Simulates the effect of knocking out a gene on each module indicator.
Uses closed-form solution for EWM and recomputation for RES/SPCA.

Reference: Dempster et al. (2021) Chronos — CRISPR knockout fitness effects.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

from .scoring.ewm import compute_ewm_scores, compute_ewm_knockout_delta
from .scoring.res import _gsea_enrichment_score


def compute_knockout_response(
    expression: pd.DataFrame,
    gene_module_map: dict[str, dict],
    evidence_weights: dict[str, float],
    perturbation_gene: str,
    cell_line_id: str,
    ensemble_weights: tuple[float, float, float] = (0.50, 0.25, 0.25),
    spca_loadings: dict[int, dict[str, float]] | None = None,
    n_modules: int = 14,
    zero_mode: str = "mean",
) -> np.ndarray:
    """
    Compute the change Δ_k in each module indicator when a gene is knocked out.

    Args:
        expression: N_cells x P_genes DataFrame (z-scored)
        gene_module_map: gene -> module assignments
        evidence_weights: gene -> evidence weight
        perturbation_gene: gene being knocked out
        cell_line_id: target cell line
        ensemble_weights: (γ_ewm, γ_res, γ_spca)
        spca_loadings: module_idx -> {gene -> loading}
        n_modules: number of modules
        zero_mode: "mean" (z=0) or "min" (lowest observed value)

    Returns:
        Array of shape (n_modules,) with Δ values.
    """
    gamma_ewm, gamma_res, gamma_spca = ensemble_weights

    if cell_line_id not in expression.index:
        raise ValueError(f"Cell line {cell_line_id} not found")
    if perturbation_gene not in expression.columns:
        raise ValueError(f"Gene {perturbation_gene} not found")

    gene_list = expression.columns.tolist()
    n_genes = len(gene_list)

    # ── EWM Δ (closed form) ──
    delta_ewm = compute_ewm_knockout_delta(
        expression, gene_module_map, evidence_weights,
        perturbation_gene, cell_line_id, n_modules,
    )

    # ── Build knockout expression vector ──
    cell_idx = expression.index.get_loc(cell_line_id)
    gene_idx = gene_list.index(perturbation_gene)
    expr_original = expression.to_numpy(dtype=np.float64)
    z_original = expr_original[cell_idx, gene_idx]

    # Set gene expression to knockout level
    if zero_mode == "mean":
        knock_value = 0.0  # z-score = 0 means average expression
    elif zero_mode == "min":
        knock_value = expr_original[:, gene_idx].min()
    else:
        knock_value = 0.0

    expr_knockout = expr_original.copy()
    expr_knockout[cell_idx, gene_idx] = knock_value

    # ── RES Δ (recompute) ──
    if gamma_res > 0:
        delta_res = _compute_res_delta(
            expr_original, expr_knockout, cell_idx, gene_list,
            gene_module_map, n_modules, n_genes,
        )
    else:
        delta_res = np.zeros(n_modules)

    # ── SPCA Δ (linear projection) ──
    if gamma_spca > 0 and spca_loadings is not None:
        delta_spca = _compute_spca_delta(
            spca_loadings, perturbation_gene, z_original, knock_value,
            n_modules,
        )
    else:
        delta_spca = np.zeros(n_modules)

    # ── Ensemble Δ ──
    delta = gamma_ewm * delta_ewm + gamma_res * delta_res + gamma_spca * delta_spca

    return delta


def _compute_res_delta(
    expr_original: np.ndarray,
    expr_knockout: np.ndarray,
    cell_idx: int,
    gene_list: list[str],
    gene_module_map: dict[str, dict],
    n_modules: int,
    n_genes: int,
) -> np.ndarray:
    """Compute RES Δ by recomputing enrichment scores before and after knockout."""
    # Build module masks
    module_masks = np.zeros((n_modules, n_genes), dtype=bool)
    module_sizes = np.zeros(n_modules, dtype=np.int32)
    for gi, gene in enumerate(gene_list):
        if gene in gene_module_map:
            for mod_idx in gene_module_map[gene]["modules"]:
                module_masks[mod_idx, gi] = True
                module_sizes[mod_idx] += 1

    def _compute_all_res(expr_vec: np.ndarray) -> np.ndarray:
        ranked = np.argsort(-expr_vec)  # descending
        es = np.zeros(n_modules)
        for mi in range(n_modules):
            if module_sizes[mi] >= 3:
                es[mi] = _gsea_enrichment_score(
                    ranked, module_masks[mi], n_genes, int(module_sizes[mi])
                )
        return es

    es_before = _compute_all_res(expr_original[cell_idx, :])
    es_after = _compute_all_res(expr_knockout[cell_idx, :])

    # The ES values are on the same scale, so simple difference
    return es_after - es_before


def _compute_spca_delta(
    spca_loadings: dict[int, dict[str, float]],
    perturbation_gene: str,
    z_original: float,
    knock_value: float,
    n_modules: int,
) -> np.ndarray:
    """Compute SPCA Δ using linear projection."""
    delta = np.zeros(n_modules)
    for mod_idx, loadings in spca_loadings.items():
        if perturbation_gene in loadings:
            alpha = loadings[perturbation_gene]
            delta[mod_idx] = alpha * (knock_value - z_original)
    return delta


def compute_knockout_summary(
    expression: pd.DataFrame,
    gene_module_map: dict[str, dict],
    evidence_weights: dict[str, float],
    ensemble_weights: tuple[float, float, float] = (0.50, 0.25, 0.25),
    spca_loadings: dict[int, dict[str, float]] | None = None,
    sample_genes: list[str] | None = None,
    sample_cells: list[str] | None = None,
    n_modules: int = 14,
) -> pd.DataFrame:
    """
    Compute knockout response summary for a sample of (cell_line, gene) pairs.

    This is for validation/analysis — computing ALL pairs would be too slow
    (1140 cells × 1123 genes = 1.28M computations).

    Args:
        expression: gene expression DataFrame
        gene_module_map: gene -> module map
        evidence_weights: gene -> weights
        ensemble_weights: (γ_ewm, γ_res, γ_spca)
        spca_loadings: SPCA loadings
        sample_genes: genes to sample (default: first 20)
        sample_cells: cell lines to sample (default: first 10)
        n_modules: number of modules

    Returns:
        DataFrame with columns: cell_line_id, perturbation_gene, Δ_0..Δ_11
    """
    if sample_genes is None:
        sample_genes = list(expression.columns[:20])
    if sample_cells is None:
        sample_cells = list(expression.index[:10])

    rows = []
    for cell in sample_cells:
        for gene in sample_genes:
            try:
                delta = compute_knockout_response(
                    expression, gene_module_map, evidence_weights,
                    gene, cell, ensemble_weights, spca_loadings, n_modules,
                )
                row = {"cell_line_id": cell, "perturbation_gene": gene}
                for k in range(n_modules):
                    row[f"delta_{k:02d}"] = delta[k]
                rows.append(row)
            except Exception:
                continue

    return pd.DataFrame(rows)
