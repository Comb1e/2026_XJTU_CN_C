"""Evidence-Weighted Mean (EWM) scoring method.

M_k^(EWM)(c) = Σ_{g∈G_k} w_g · z_{c,g} / Σ_{g∈G_k} w_g

where w_g is the composite evidence weight from MitoCarta3.0 scores.

Reference: Rath et al. (2021) MitoCarta3.0 — Bayesian integration of 7 features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_ewm_scores(
    expression: pd.DataFrame,
    gene_module_map: dict[str, dict],
    evidence_weights: dict[str, float],
    n_modules: int = 14,
) -> pd.DataFrame:
    """
    Compute evidence-weighted mean scores for each module and cell line.

    Args:
        expression: N_cells x P_genes DataFrame (z-scored)
        gene_module_map: dict gene -> {modules, sub_mito, ...}
        evidence_weights: dict gene -> weight
        n_modules: number of modules (default 12)

    Returns:
        DataFrame of shape (N_cells, n_modules) with EWM scores.
    """
    cell_ids = expression.index.tolist()
    gene_list = expression.columns.tolist()

    # Pre-compute per-module gene masks and weight sums
    module_genes: list[list[int]] = [[] for _ in range(n_modules)]
    module_weights: list[list[float]] = [[] for _ in range(n_modules)]

    for gene_idx, gene in enumerate(gene_list):
        if gene in gene_module_map and gene in evidence_weights:
            w = evidence_weights[gene]
            for mod_idx in gene_module_map[gene]["modules"]:
                module_genes[mod_idx].append(gene_idx)
                module_weights[mod_idx].append(w)

    # Compute weighted means
    scores = np.zeros((len(cell_ids), n_modules), dtype=np.float64)
    expr_array = expression.to_numpy(dtype=np.float64)

    for mod_idx in range(n_modules):
        gene_indices = module_genes[mod_idx]
        if not gene_indices:
            continue
        weights = np.array(module_weights[mod_idx], dtype=np.float64)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            continue
        # Weighted mean: (X · w) / Σw
        scores[:, mod_idx] = (
            expr_array[:, gene_indices] @ weights
        ) / weight_sum

    result = pd.DataFrame(
        scores,
        index=cell_ids,
        columns=[f"EWM_{i:02d}" for i in range(n_modules)],
    )
    return result


def compute_ewm_knockout_delta(
    expression: pd.DataFrame,
    gene_module_map: dict[str, dict],
    evidence_weights: dict[str, float],
    perturbation_gene: str,
    cell_line_id: str,
    n_modules: int = 14,
) -> np.ndarray:
    """
    Compute the change in EWM scores when a gene is knocked out.

    Closed-form solution (linear case):
        Δ_k = -w_g · z_{c,g} / (Σ_{j∈G_k} w_j - w_g · I(g∈G_k))

    Args:
        expression: N_cells x P_genes DataFrame
        gene_module_map: gene -> module assignments
        evidence_weights: gene -> weight
        perturbation_gene: gene symbol being knocked out
        cell_line_id: cell line identifier
        n_modules: number of modules

    Returns:
        Array of shape (n_modules,) with Δ values.
    """
    if cell_line_id not in expression.index:
        raise ValueError(f"Cell line {cell_line_id} not in expression data")
    if perturbation_gene not in expression.columns:
        raise ValueError(f"Gene {perturbation_gene} not in expression data")

    z_cg = expression.loc[cell_line_id, perturbation_gene]
    w_g = evidence_weights.get(perturbation_gene, 1.0)

    # Pre-compute total weights per module
    gene_list = expression.columns.tolist()
    module_weight_sums = np.zeros(n_modules, dtype=np.float64)
    for gene_idx, gene in enumerate(gene_list):
        if gene in gene_module_map and gene in evidence_weights:
            w = evidence_weights[gene]
            for mod_idx in gene_module_map[gene]["modules"]:
                module_weight_sums[mod_idx] += w

    # Compute Δ per module
    delta = np.zeros(n_modules, dtype=np.float64)
    gene_modules = gene_module_map.get(perturbation_gene, {}).get("modules", [])

    for mod_idx in gene_modules:
        total_w = module_weight_sums[mod_idx]
        denom = total_w - w_g
        if denom > 0:
            # Δ = -w_g * z / (Σw - w_g)
            delta[mod_idx] = -w_g * z_cg / denom

    return delta
