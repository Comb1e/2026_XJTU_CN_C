"""Sparse PCA scoring and gene set refinement (Stage 2 & Method C).

Stage 2 (optional): Refine module gene sets by keeping only co-expressed genes.
Method C: Score modules using the first sparse PC projection.

Reference: Frost (2025) EESPCA — sparse PCA for gene set curation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import SparsePCA


def refine_gene_set(
    expression: pd.DataFrame,
    gene_module_map: dict[str, dict],
    module_idx: int,
    sparsity_alpha: float = 0.5,
    sparsity_threshold: float = 0.1,
    min_genes: int = 5,
    random_state: int = 42,
) -> tuple[list[str], np.ndarray]:
    """
    Refine a module's gene set using sparse PCA.

    Keeps only genes with |loading| > threshold on the first sparse PC.

    Args:
        expression: N_cells x P_genes DataFrame
        gene_module_map: gene -> module assignments
        module_idx: which module to refine
        sparsity_alpha: ElasticNet alpha (0=Ridge, 1=Lasso)
        sparsity_threshold: minimum |loading| to keep a gene
        min_genes: minimum number of genes to retain
        random_state: random seed

    Returns:
        (refined_gene_list, loadings_array) — genes with |loading| > threshold
        and their corresponding loadings. Returns original set if refinement
        reduces below min_genes.
    """
    # Collect genes in this module
    module_genes = [
        g for g in expression.columns
        if g in gene_module_map and module_idx in gene_module_map[g]["modules"]
    ]

    if len(module_genes) < min_genes:
        return module_genes, np.ones(len(module_genes)) / np.sqrt(len(module_genes))

    # Extract sub-matrix
    X = expression[module_genes].to_numpy(dtype=np.float64)

    # Handle potential NaN
    if np.isnan(X).any():
        X = np.nan_to_num(X, nan=0.0)

    # Run sparse PCA
    n_components = 1
    spca = SparsePCA(
        n_components=n_components,
        alpha=sparsity_alpha,
        random_state=random_state,
        max_iter=500,
        ridge_alpha=0.01,
    )
    try:
        loadings = spca.fit_transform(X.T).ravel()
    except Exception:
        # Fallback: use regular PCA if sparse PCA fails
        from sklearn.decomposition import PCA
        pca = PCA(n_components=1, random_state=random_state)
        pca.fit(X)
        loadings = pca.components_.ravel()

    # Select genes with |loading| > threshold
    mask = np.abs(loadings) > sparsity_threshold
    if mask.sum() < min_genes:
        # Keep top min_genes by absolute loading
        top_indices = np.argsort(np.abs(loadings))[-min_genes:]
        mask = np.zeros(len(loadings), dtype=bool)
        mask[top_indices] = True

    refined_genes = [module_genes[i] for i in range(len(module_genes)) if mask[i]]
    refined_loadings = loadings[mask]

    # Normalize loadings to unit norm
    norm = np.sqrt(np.sum(refined_loadings ** 2))
    if norm > 0:
        refined_loadings /= norm

    return refined_genes, refined_loadings


def compute_spca_scores(
    expression: pd.DataFrame,
    gene_module_map: dict[str, dict],
    n_modules: int = 14,
    refinement_config: dict | None = None,
) -> pd.DataFrame:
    """
    Compute sparse PCA projection scores for each module and cell line.

    For each module, fits a sparse PCA on the module's gene expression
    sub-matrix and projects cell lines onto the first sparse component.

    Args:
        expression: N_cells x P_genes DataFrame
        gene_module_map: gene -> module assignments
        n_modules: number of modules
        refinement_config: optional config dict for gene set refinement

    Returns:
        DataFrame of shape (N_cells, n_modules) with SPCA scores.
        Also stores loadings in result.attrs['loadings'] dict.
    """
    cell_ids = expression.index.tolist()
    gene_list = expression.columns.tolist()

    if refinement_config is None:
        refinement_config = {"enabled": False}

    scores = np.zeros((len(cell_ids), n_modules), dtype=np.float64)
    all_loadings: dict[int, dict[str, float]] = {}

    for mod_idx in range(n_modules):
        # Get module genes
        if refinement_config.get("enabled", False):
            refined_genes, loadings = refine_gene_set(
                expression,
                gene_module_map,
                mod_idx,
                sparsity_alpha=refinement_config.get("elasticnet_alpha", 0.5),
                sparsity_threshold=refinement_config.get("sparsity_threshold", 0.1),
                min_genes=refinement_config.get("min_genes_after_refinement", 5),
            )
            gene_subset = refined_genes
            gene_loadings = loadings
        else:
            # Use all module genes
            module_genes = [
                g for g in gene_list
                if g in gene_module_map
                and mod_idx in gene_module_map[g]["modules"]
            ]
            if not module_genes:
                continue
            # Fit sparse PCA on the module genes to get loadings
            X = expression[module_genes].to_numpy(dtype=np.float64)
            if np.isnan(X).any():
                X = np.nan_to_num(X, nan=0.0)
            try:
                spca = SparsePCA(
                    n_components=1, alpha=0.5, random_state=42,
                    max_iter=500, ridge_alpha=0.01,
                )
                loadings = spca.fit_transform(X.T).ravel()
            except Exception:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=1, random_state=42)
                pca.fit(X)
                loadings = pca.components_.ravel()
            norm = np.sqrt(np.sum(loadings ** 2))
            if norm > 0:
                loadings = loadings / norm
            gene_subset = module_genes
            gene_loadings = loadings

        # Project cell lines onto the sparse PC
        X_sub = expression[gene_subset].to_numpy(dtype=np.float64)
        if np.isnan(X_sub).any():
            X_sub = np.nan_to_num(X_sub, nan=0.0)
        scores[:, mod_idx] = X_sub @ gene_loadings

        # Store loadings
        all_loadings[mod_idx] = dict(zip(gene_subset, gene_loadings.tolist()))

    result = pd.DataFrame(
        scores,
        index=cell_ids,
        columns=[f"SPCA_{i:02d}" for i in range(n_modules)],
    )
    result.attrs["loadings"] = all_loadings
    return result
