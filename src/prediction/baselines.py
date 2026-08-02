"""Baseline and collaborative features for gene dependency prediction.

Computes label-derived features with strict leakage control:
  - Gene baseline μ̂_g: shrunk mean (training) + teacher model (cold-start)
  - Cell bias β̂_c: shrunk mean (training) + RidgeCV imputation (new cells)
  - SVD latent factors: TruncatedSVD on label matrix with imputation
  - Neighbor essentiality: k-NN with self-exclusion
  - Module-level curated priors

All out-of-fold computation is handled by validation.py; this module
provides the building blocks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import cross_val_predict, KFold


# ── Empirical Bayes shrinkage ────────────────────────────────────────────────


def shrink_gene_means(
    labels: pd.DataFrame,
    global_mean: float | None = None,
    prior_weight: float = 10.0,
) -> dict[str, float]:
    """Empirical Bayes shrinkage of per-gene mean dependency toward global mean.

    μ̂_g = (n_g * x̄_g + λ * μ_global) / (n_g + λ)

    Args:
        labels: DataFrame with [perturbation_gene, label].
        global_mean: global label mean (computed from labels if None).
        prior_weight: shrinkage strength λ (higher = more shrinkage).

    Returns:
        dict gene -> shrunk mean.
    """
    if global_mean is None:
        global_mean = float(labels["label"].mean())

    gene_groups = labels.groupby("perturbation_gene")["label"]
    gene_counts = gene_groups.count()
    gene_means = gene_groups.mean()

    shrunk = {}
    for gene in gene_means.index:
        n = gene_counts[gene]
        raw_mean = gene_means[gene]
        shrunk[gene] = (n * raw_mean + prior_weight * global_mean) / (n + prior_weight)

    return shrunk


def shrink_cell_means(
    labels: pd.DataFrame,
    global_mean: float | None = None,
    prior_weight: float = 10.0,
) -> dict[str, float]:
    """Empirical Bayes shrinkage of per-cell mean dependency."""
    if global_mean is None:
        global_mean = float(labels["label"].mean())

    cell_groups = labels.groupby("cell_line_id")["label"]
    cell_counts = cell_groups.count()
    cell_means = cell_groups.mean()

    shrunk = {}
    for cell in cell_means.index:
        n = cell_counts[cell]
        raw_mean = cell_means[cell]
        shrunk[cell] = (n * raw_mean + prior_weight * global_mean) / (n + prior_weight)

    return shrunk


# ── Gene baseline teacher ───────────────────────────────────────────────────


def train_gene_baseline_teacher(
    gene_static_features: pd.DataFrame,
    gene_expr_profile_features: pd.DataFrame,
    labels: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
) -> tuple[Any, dict[str, float], pd.DataFrame]:
    """Train a teacher model to predict gene mean dependency from gene features.

    Used for cold-start gene baseline estimation. Returns out-of-fold predictions
    for training genes so they can be used as features without leakage.

    Args:
        gene_static_features: G1 features indexed by gene_symbol.
        gene_expr_profile_features: G2 features indexed by gene_symbol.
        labels: training labels DataFrame.
        n_folds: cross-validation folds.
        random_state: random seed.

    Returns:
        (fitted_model, gene_to_oof_prediction, combined_gene_features)
    """
    # Compute per-gene mean labels
    gene_means = labels.groupby("perturbation_gene")["label"].mean()
    gene_stds = labels.groupby("perturbation_gene")["label"].std()

    # Build gene feature matrix
    gene_feats = gene_static_features.join(gene_expr_profile_features, how="inner")
    common_genes = sorted(set(gene_feats.index) & set(gene_means.index))
    X = gene_feats.reindex(common_genes).to_numpy(dtype=np.float32)
    y = gene_means.reindex(common_genes).to_numpy(dtype=np.float64)

    # Fill NaN in features
    X = np.nan_to_num(X, nan=0.0)

    # Train HGBR for nonlinear gene baseline prediction
    # Conservative params for ~933 gene-level samples
    teacher = HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        min_samples_leaf=10,
        l2_regularization=1.0,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=random_state,
    )
    teacher.fit(X, y)

    # Out-of-fold predictions
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds = cross_val_predict(teacher, X, y, cv=kf)

    gene_to_oof = dict(zip(common_genes, oof_preds))

    return teacher, gene_to_oof, gene_feats


def predict_gene_baselines(
    teacher_model: Any,
    gene_feats: pd.DataFrame,
    cold_genes: list[str],
) -> dict[str, float]:
    """Predict gene baselines for cold-start genes using the teacher model."""
    X = gene_feats.reindex(cold_genes).to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    preds = teacher_model.predict(X)
    return dict(zip(cold_genes, preds))


# ── Cell bias imputation ─────────────────────────────────────────────────────


def train_cell_bias_imputer(
    cell_features: pd.DataFrame,
    labels: pd.DataFrame,
    prior_weight: float = 10.0,
) -> tuple[RidgeCV, dict[str, float]]:
    """Train a RidgeCV model to predict cell mean dependency from cell features.

    Used for new-cell-line bias imputation.

    Args:
        cell_features: G3 features indexed by cell_line_id.
        labels: training labels DataFrame.
        prior_weight: shrinkage strength.

    Returns:
        (fitted_imputer, cell_to_shrunk_bias)
    """
    cell_shrunk = shrink_cell_means(labels, prior_weight=prior_weight)
    train_cells = sorted(set(cell_features.index) & set(cell_shrunk.keys()))

    X = cell_features.reindex(train_cells).to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = np.array([cell_shrunk[c] for c in train_cells], dtype=np.float64)

    ridge = RidgeCV(alphas=np.logspace(-2, 3, 20), store_cv_results=False)
    ridge.fit(X, y)

    return ridge, cell_shrunk


def impute_cell_biases(
    imputer: RidgeCV,
    cell_features: pd.DataFrame,
    new_cells: list[str],
) -> dict[str, float]:
    """Impute cell biases for new cell lines."""
    X = cell_features.reindex(new_cells).to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)
    preds = imputer.predict(X)
    return dict(zip(new_cells, preds))


# ── SVD latent factorization ─────────────────────────────────────────────────


def compute_label_svd(
    labels: pd.DataFrame,
    k: int = 20,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.Index, pd.Index, float]:
    """Compute TruncatedSVD on the label matrix.

    Returns cell factors U, gene factors V, and the global mean for imputation.

    Args:
        labels: training labels DataFrame.
        k: number of latent dimensions.
        random_state: random seed.

    Returns:
        (U_cells, V_genes, cell_index, gene_index, global_mean)
    """
    global_mean = float(labels["label"].mean())

    # Build sparse label matrix
    cells = sorted(labels["cell_line_id"].unique())
    genes = sorted(labels["perturbation_gene"].unique())
    cell_idx = {c: i for i, c in enumerate(cells)}
    gene_idx = {g: i for i, g in enumerate(genes)}

    R = np.full((len(cells), len(genes)), global_mean, dtype=np.float32)
    for _, row in labels.iterrows():
        ci = cell_idx[row["cell_line_id"]]
        gi = gene_idx[row["perturbation_gene"]]
        R[ci, gi] = row["label"]

    # Center
    R_centered = R - global_mean

    svd = TruncatedSVD(n_components=k, random_state=random_state)
    U = svd.fit_transform(R_centered)  # cells × k
    V = svd.components_.T            # genes × k

    return (U.astype(np.float32), V.astype(np.float32),
            pd.Index(cells), pd.Index(genes), global_mean)


def impute_cell_factors(
    U: np.ndarray,
    cell_index: pd.Index,
    cell_features: pd.DataFrame,
    new_cells: list[str],
) -> np.ndarray:
    """Impute SVD cell factors for new cell lines via RidgeCV."""
    train_cells = sorted(set(cell_index) & set(cell_features.index))
    X = cell_features.reindex(train_cells).to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)

    cell_to_row = {c: i for i, c in enumerate(cell_index)}
    Y = np.array([U[cell_to_row[c]] for c in train_cells], dtype=np.float32)

    # Multi-output Ridge
    from sklearn.linear_model import Ridge
    k = U.shape[1]
    preds = np.zeros((len(new_cells), k), dtype=np.float32)
    X_new = cell_features.reindex(new_cells).to_numpy(dtype=np.float32)
    X_new = np.nan_to_num(X_new, nan=0.0)

    for j in range(k):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, Y[:, j])
        preds[:, j] = ridge.predict(X_new).astype(np.float32)

    return preds


def impute_gene_factors(
    V: np.ndarray,
    gene_index: pd.Index,
    gene_static_features: pd.DataFrame,
    gene_expr_profile_features: pd.DataFrame,
    cold_genes: list[str],
) -> np.ndarray:
    """Impute SVD gene factors for cold-start genes via Ridge."""
    gene_feats = gene_static_features.join(gene_expr_profile_features, how="inner")
    train_genes = sorted(set(gene_index) & set(gene_feats.index))

    X = gene_feats.reindex(train_genes).to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)

    gene_to_row = {g: i for i, g in enumerate(gene_index)}
    Y = np.array([V[gene_to_row[g]] for g in train_genes], dtype=np.float32)

    from sklearn.linear_model import Ridge
    k = V.shape[1]
    preds = np.zeros((len(cold_genes), k), dtype=np.float32)
    X_new = gene_feats.reindex(cold_genes).to_numpy(dtype=np.float32)
    X_new = np.nan_to_num(X_new, nan=0.0)

    for j in range(k):
        ridge = Ridge(alpha=1.0)
        ridge.fit(X, Y[:, j])
        preds[:, j] = ridge.predict(X_new).astype(np.float32)

    return preds


# ── Neighbor essentiality (k-NN) ─────────────────────────────────────────────


def compute_neighbor_essentiality(
    labels: pd.DataFrame,
    cell_indicators: pd.DataFrame,
    k: int = 20,
    exclude_self: bool = True,
) -> pd.DataFrame:
    """Compute k-NN neighbor essentiality for each (cell, gene) pair.

    For a given cell c and gene g, finds k nearest neighbor cells (by cosine
    distance over the 14 orthogonalized indicators) and averages their labels
    for gene g. Self is excluded from neighbors.

    Args:
        labels: training labels DataFrame.
        cell_indicators: N_cells × K indicators (orthogonalized).
        k: number of neighbors.
        exclude_self: whether to exclude the query cell itself.

    Returns:
        DataFrame with columns [cell_line_id, perturbation_gene, neighbor_score].
    """
    # Build per-cell label lookup
    all_cells = sorted(cell_indicators.index.tolist())
    cell_to_idx = {c: i for i, c in enumerate(all_cells)}

    # Cell × gene label matrix (mean-imputed for missing)
    all_genes = sorted(labels["perturbation_gene"].unique())
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    global_mean = float(labels["label"].mean())

    R = np.full((len(all_cells), len(all_genes)), global_mean, dtype=np.float32)
    for _, row in labels.iterrows():
        if row["cell_line_id"] in cell_to_idx and row["perturbation_gene"] in gene_to_idx:
            ci = cell_to_idx[row["cell_line_id"]]
            gi = gene_to_idx[row["perturbation_gene"]]
            R[ci, gi] = row["label"]

    # Fit k-NN on indicators
    X = cell_indicators.reindex(all_cells).to_numpy(dtype=np.float32)
    nbrs = NearestNeighbors(n_neighbors=k + (1 if exclude_self else 0), metric="cosine")
    nbrs.fit(X)
    distances, indices = nbrs.kneighbors(X)

    # Compute neighbor scores for all pairs
    rows = []
    for ci, cell in enumerate(all_cells):
        nn_indices = indices[ci]
        nn_weights = 1.0 / (distances[ci] + 1e-8)
        # Exclude self if present
        if exclude_self:
            mask = nn_indices != ci
            nn_indices = nn_indices[mask][:k]
            nn_weights = nn_weights[mask][:k]
        nn_weights /= nn_weights.sum()

        for gi, gene in enumerate(all_genes):
            score = float(np.dot(nn_weights, R[nn_indices, gi]))
            rows.append({
                "cell_line_id": cell,
                "perturbation_gene": gene,
                "neighbor_score": score,
            })

    return pd.DataFrame(rows)


def compute_neighbor_scores_for_pairs(
    pairs: pd.DataFrame,
    neighbor_df: pd.DataFrame,
) -> np.ndarray:
    """Extract neighbor scores for specific (cell, gene) pairs."""
    neighbor_map = neighbor_df.set_index(["cell_line_id", "perturbation_gene"])["neighbor_score"]
    scores = np.zeros(len(pairs), dtype=np.float32)
    for i, (_, row) in enumerate(pairs.iterrows()):
        key = (row["cell_line_id"], row["perturbation_gene"])
        if key in neighbor_map.index:
            scores[i] = neighbor_map.loc[key]
    return scores


# ── Curated module priors ────────────────────────────────────────────────────


def build_module_priors() -> dict[int, float]:
    """Build curated module-level essentiality priors from literature.

    Based on [1] Glover 2024 (cell death as CRISPR phenotype hub),
    [3] Dempster 2021 (Chronos), [4] Rath 2021 (MitoCarta3.0).

    Higher value = genes in this module are more likely to be essential
    (i.e., knockout causes stronger growth defect).
    """
    # OXPHOS and core machinery are pan-essential; signaling/transport less so
    priors = {
        0: 0.25,   # OXPHOS_CI — strong dependency
        1: 0.20,   # OXPHOS_CII_CIII
        2: 0.25,   # OXPHOS_CIV_CV — strong dependency
        3: 0.10,   # TCA_PYRUVATE — moderate
        4: 0.05,   # FAO_LIPID — tissue-conditional
        5: 0.05,   # AA_COFACTOR — tissue-conditional
        6: 0.30,   # MITO_RIBOSOME — pan-essential (translation)
        7: 0.20,   # mtDNA_RNA — strong (mtDNA maintenance)
        8: 0.10,   # PROTEIN_IMPORT — moderate
        9: 0.05,   # TRANSPORT — variable
        10: 0.05,  # REDOX_DETOX — variable
        11: 0.05,  # MITO_DYNAMICS — variable
        12: 0.20,  # CELL_DEATH — directly linked to CRISPR phenotype [1]
        13: 0.00,  # SIGNALING — mostly tissue-specific
    }
    return priors


# ── Collaborative features assembly ──────────────────────────────────────────


def build_collaborative_features(
    pairs: pd.DataFrame,
    gene_baselines: dict[str, float],
    cell_biases: dict[str, float],
    svd_dot_products: dict[tuple[str, str], float] | None = None,
    neighbor_scores: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build G5 collaborative feature DataFrame from computed baselines.

    Args:
        pairs: DataFrame with [cell_line_id, perturbation_gene].
        gene_baselines: gene → μ̂_g dict.
        cell_biases: cell → β̂_c dict.
        svd_dot_products: (cell, gene) → U_c·V_g dict (optional).
        neighbor_scores: array of neighbor scores aligned to pairs (optional).

    Returns:
        DataFrame with G5 features, aligned to pair order.
    """
    n = len(pairs)
    features = {}

    # Gene baseline
    features["g5_gene_baseline"] = np.array(
        [gene_baselines.get(g, 0.0) for g in pairs["perturbation_gene"]],
        dtype=np.float32,
    )

    # Cell bias
    features["g5_cell_bias"] = np.array(
        [cell_biases.get(c, 0.0) for c in pairs["cell_line_id"]],
        dtype=np.float32,
    )

    # SVD dot product
    if svd_dot_products is not None:
        features["g5_svd_dot"] = np.array(
            [svd_dot_products.get((c, g), 0.0)
             for c, g in zip(pairs["cell_line_id"], pairs["perturbation_gene"])],
            dtype=np.float32,
        )

    # Neighbor essentiality
    if neighbor_scores is not None:
        features["g5_neighbor_score"] = neighbor_scores

    return pd.DataFrame(features)
