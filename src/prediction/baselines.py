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


# ── LOCO (leave-one-cell-out) gene means ──────────────────────────────────────


def compute_loco_gene_means(
    labels: pd.DataFrame,
) -> tuple[dict[str, float], np.ndarray]:
    """Compute leave-one-cell-out gene means for leakage-free training features.

    For each training row (c, g): LOCO mean = (sum over all cells of g − label_{c,g}) / (n_g − 1).
    This is the honest estimate of gene g's mean dependency from all OTHER cells.

    Args:
        labels: DataFrame with [cell_line_id, perturbation_gene, label].

    Returns:
        (gene_means, loco_values):
          - gene_means: dict gene → plain mean (for test-time warm genes)
          - loco_values: array aligned to labels rows with LOCO mean per row
    """
    gene_sums = labels.groupby("perturbation_gene")["label"].transform("sum")
    gene_counts = labels.groupby("perturbation_gene")["label"].transform("count")

    # Plain gene mean
    gene_means = (gene_sums / gene_counts).to_dict()

    # LOCO: (sum − self) / (count − 1). Handle singletons (count=1 → fallback to global mean).
    global_mean = float(labels["label"].mean())
    loco = np.where(
        gene_counts > 1,
        (gene_sums.values - labels["label"].values) / (gene_counts.values - 1),
        global_mean,
    )

    # Build gene_means dict keyed by gene
    gene_mean_dict = {}
    for gene in labels["perturbation_gene"].unique():
        mask = labels["perturbation_gene"] == gene
        gene_mean_dict[gene] = float(gene_sums[mask].iloc[0] / gene_counts[mask].iloc[0])

    return gene_mean_dict, loco.astype(np.float32)


def compute_plain_gene_means(labels: pd.DataFrame) -> dict[str, float]:
    """Compute plain per-gene mean dependency (no shrinkage, no LOCO).

    For test-time warm genes: the plain mean over all training labels is the
    correct analogue of what models learn from LOCO training means.
    """
    return labels.groupby("perturbation_gene")["label"].mean().to_dict()


# ── Gene baseline teacher ───────────────────────────────────────────────────


def train_gene_baseline_teacher(
    gene_static_features: pd.DataFrame,
    gene_expr_profile_features: pd.DataFrame,
    labels: pd.DataFrame,
    n_folds: int = 5,
    random_state: int = 42,
    extra_features: pd.DataFrame | None = None,
    model_type: str = "ridge",
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
        extra_features: optional additional gene-level features (e.g., pathway
                       membership, co-expression-KNN, description keywords).
        model_type: "ridge" (linear, fast) or "xgboost" (nonlinear, higher R²).

    Returns:
        (fitted_model, gene_to_oof_prediction, combined_gene_features)
    """
    # Compute per-gene mean labels
    gene_means = labels.groupby("perturbation_gene")["label"].mean()

    # Build gene feature matrix (G1 + G2)
    gene_feats = gene_static_features.join(gene_expr_profile_features, how="inner")

    # Add learned module priors: actual per-module mean dependency from training data
    _add_learned_module_priors(gene_feats, gene_means)

    # Add extra features if provided (e.g., pathway membership)
    if extra_features is not None:
        extra_cols = [c for c in extra_features.columns if c not in gene_feats.columns]
        if extra_cols:
            extra_df = extra_features.reindex(gene_feats.index)[extra_cols].fillna(0.0)
            gene_feats = pd.concat([gene_feats, extra_df], axis=1)

    common_genes = sorted(set(gene_feats.index) & set(gene_means.index))
    X = gene_feats.reindex(common_genes).to_numpy(dtype=np.float32)
    y = gene_means.reindex(common_genes).to_numpy(dtype=np.float64)

    # Fill NaN in features
    X = np.nan_to_num(X, nan=0.0)

    if model_type == "xgboost":
        try:
            import xgboost as xgb
            teacher = xgb.XGBRegressor(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=random_state, n_jobs=-1, verbosity=0,
            )
        except ImportError:
            model_type = "ridge"

    if model_type != "xgboost":
        # Train Ridge regression for gene baseline prediction
        from sklearn.linear_model import RidgeCV
        teacher = RidgeCV(
            alphas=np.logspace(-1, 3, 20),
            store_cv_results=False,
        )

    teacher.fit(X, y)

    # Out-of-fold predictions
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds = cross_val_predict(teacher, X, y, cv=kf)

    gene_to_oof = dict(zip(common_genes, oof_preds))

    # Report teacher quality
    from sklearn.metrics import r2_score
    teacher_r2 = r2_score(y, teacher.predict(X))
    oof_r2 = r2_score(y, oof_preds)
    print(f"  Teacher ({model_type}): train R²={teacher_r2:.4f}, OOF R²={oof_r2:.4f}, "
          f"corr={np.corrcoef(y, oof_preds)[0,1]:.4f}")

    return teacher, gene_to_oof, gene_feats


def build_description_keyword_features(
    gene_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Build gene-level features from the `description` text field.

    Extracts keyword counts for biological terms that correlate with gene
    essentiality (e.g., ribosomal, subunit, assembly, signaling).
    """
    genes = gene_meta["gene_symbol"].tolist()
    descriptions = gene_meta.set_index("gene_symbol")["description"].fillna("").astype(str)

    # Keywords grouped by expected essentiality level
    keywords_high = [
        "ribosomal", "ribosome", "mitoribosom", "translation",
        "trna synthetase", "aminoacyl-trna",
    ]
    keywords_moderate = [
        "subunit", "complex i", "complex ii", "complex iii",
        "complex iv", "complex v", "cytochrome", "nadh dehydrogenase",
        "atp synthase", "oxidoreductase", "electron transfer",
        "respiratory chain", "oxphos",
    ]
    keywords_moderate2 = [
        "assembly factor", "chaperone", "protease", "peptidase",
        "import", "translocase", "tom ", "tim ", "sam ",
    ]
    keywords_variable = [
        "dehydrogenase", "transferase", "hydrolase", "isomerase",
        "ligase", "kinase", "phosphatase", "reductase", "synthase",
        "transferase", "carboxylase", "decarboxylase",
    ]
    keywords_low = [
        "carrier", "transporter", "channel", "porin",
        "signaling", "receptor", "adapter", "scaffold",
    ]

    features = {}
    desc_lower = descriptions.str.lower()

    def _count_keywords(text_series, keywords):
        counts = np.zeros(len(genes), dtype=np.float32)
        for kw in keywords:
            counts += text_series.str.contains(kw, regex=False).fillna(0).to_numpy(dtype=np.float32)
        return np.clip(counts, 0, 5)

    features["desc_high_essentiality"] = _count_keywords(desc_lower, keywords_high)
    features["desc_moderate_essentiality"] = _count_keywords(desc_lower, keywords_moderate)
    features["desc_moderate2_essentiality"] = _count_keywords(desc_lower, keywords_moderate2)
    features["desc_enzyme"] = _count_keywords(desc_lower, keywords_variable)
    features["desc_low_essentiality"] = _count_keywords(desc_lower, keywords_low)

    # Additional features
    features["desc_length"] = descriptions.str.len().fillna(0).to_numpy(dtype=np.float32)
    features["desc_has_mitochondrial"] = desc_lower.str.contains(
        "mitochondri", regex=False
    ).fillna(0).to_numpy(dtype=np.float32)

    return pd.DataFrame(features, index=genes)


def _add_learned_module_priors(
    gene_feats: pd.DataFrame,
    gene_means: pd.Series,
) -> None:
    """Add learned per-module mean dependency as gene features.

    Replaces the dead-code hand-set build_module_priors() with data-driven values.
    Computes the mean label per module from training data and assigns each gene
    the sum and max of its modules' learned priors.
    """
    # Identify module membership columns (gene_module_00 through gene_module_13)
    mod_cols = [c for c in gene_feats.columns if c.startswith("gene_module_")]
    if not mod_cols:
        return

    # Filter to genes with known labels
    common = gene_feats.index.intersection(gene_means.index)
    if len(common) < 10:
        return

    # Compute module prior: mean label of genes in each module (weighted by membership)
    module_priors = {}
    for col in mod_cols:
        in_mod = gene_feats.loc[common, col] > 0.5
        if in_mod.sum() >= 3:
            module_priors[col] = float(gene_means.loc[common[in_mod]].mean())

    # Add prior features
    prior_sum = np.zeros(len(gene_feats), dtype=np.float32)
    prior_max = np.zeros(len(gene_feats), dtype=np.float32)
    for col, prior_val in module_priors.items():
        mask = gene_feats[col].values > 0.5
        prior_sum[mask] += prior_val
        prior_max[mask] = np.maximum(prior_max[mask], prior_val)
    gene_feats["gene_learned_prior_sum"] = prior_sum
    gene_feats["gene_learned_prior_max"] = prior_max


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
    if not train_cells:
        # No training cells in features → return zeros for new cells
        k = U.shape[1]
        return np.zeros((len(new_cells), k), dtype=np.float32)

    X = cell_features.reindex(train_cells).to_numpy(dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0)

    cell_to_row = {c: i for i, c in enumerate(cell_index)}
    Y = np.array([U[cell_to_row[c]] for c in train_cells], dtype=np.float32)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)

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


def build_pw140_membership_features(
    gene_meta: pd.DataFrame,
    pathway_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Build 140-dim pathway membership one-hot features for each gene.

    Uses the provider's MitoCarta3.0 pathway annotations (leaf-level pathway
    names from the `pathways` column in gene_metadata). This is the
    fine-grained resolution the pipeline's module aggregation destroys.
    Each gene typically belongs to 1-2 of the 140 pathways.

    Args:
        gene_meta: gene_metadata DataFrame with `gene_symbol` and `pathways` columns.
        pathway_meta: pathway_metadata DataFrame with `pathway_name` column.

    Returns:
        DataFrame indexed by gene_symbol with `gene_pw_<pathway_name>` columns.
    """
    genes = gene_meta["gene_symbol"].tolist()
    pw_names = pathway_meta["pathway_name"].tolist()
    pw_to_idx = {pw: i for i, pw in enumerate(pw_names)}

    # Parse each gene's pathway annotations
    gene_pathways = {}
    for _, row in gene_meta.iterrows():
        gene = row["gene_symbol"]
        pw_raw = str(row.get("pathways", ""))
        if pw_raw and pw_raw != "nan":
            # Extract leaf pathway names from hierarchical strings
            # Format: "Category > Subcategory > LeafName" or "LeafName"
            gene_pws = set()
            for part in pw_raw.replace("|", ";").split(";"):
                part = part.strip()
                # Take the rightmost (most specific) component
                leaf = part.split(">")[-1].strip()
                # Convert to metadata key format (spaces→underscores, remove commas)
                key = leaf.replace(" ", "_").replace(",", "")
                if key in pw_to_idx:
                    gene_pws.add(key)
            gene_pathways[gene] = gene_pws

    # Build feature matrix
    n_pathways = len(pw_names)
    features = {}
    for i, pw_name in enumerate(pw_names):
        col = f"gene_pw_{pw_name[:40]}"  # Truncate long names
        features[col] = np.array(
            [1.0 if gene in gene_pathways and pw_name in gene_pathways[gene] else 0.0
             for gene in genes],
            dtype=np.float32,
        )

    return pd.DataFrame(features, index=genes)


def compute_coexpression_knn_features(
    expression: pd.DataFrame,
    labels: pd.DataFrame,
    k: int = 20,
    oof_genes: set[str] | None = None,
) -> pd.DataFrame:
    """Compute co-expression KNN gene baseline features.

    For each gene g, finds its K most expression-correlated genes and computes
    their mean label. This captures the biological prior that co-expressed genes
    tend to have similar essentiality profiles.

    When oof_genes is provided, those genes are excluded from the label pool
    (out-of-fold computation for CV). The expression correlation is always
    computed on the full expression matrix.

    Args:
        expression: N_cells × P_genes expression DataFrame (z-scored).
        labels: training labels DataFrame.
        k: number of nearest neighbors.
        oof_genes: genes to exclude from label pool (for OOF computation).

    Returns:
        DataFrame indexed by gene_symbol with columns:
          - gene_coexpr_knn_mean: mean label of K most correlated genes
          - gene_coexpr_knn_weighted: correlation-weighted mean label
    """
    gene_list = expression.columns.tolist()
    expr_array = expression.to_numpy(dtype=np.float64)

    # Compute gene × gene Pearson correlation matrix
    # Use float64 for numerical stability
    expr_centered = expr_array - expr_array.mean(axis=0, keepdims=True)
    expr_std = expr_centered.std(axis=0, ddof=0)
    expr_std[expr_std < 1e-12] = 1.0
    expr_normalized = expr_centered / expr_std
    corr_matrix = (expr_normalized.T @ expr_normalized) / (expr_array.shape[0] - 1)

    # Build gene → mean label mapping
    gene_means = labels.groupby("perturbation_gene")["label"].mean()
    gene_to_mean = gene_means.to_dict()

    # Exclude OOF genes from label pool
    if oof_genes:
        excluded = oof_genes & set(gene_to_mean.keys())
        for g in excluded:
            del gene_to_mean[g]

    labeled_genes = sorted(gene_to_mean.keys())
    if not labeled_genes:
        # Fallback: all-zero features
        return pd.DataFrame({
            "gene_coexpr_knn_mean": np.zeros(len(gene_list), dtype=np.float32),
            "gene_coexpr_knn_weighted": np.zeros(len(gene_list), dtype=np.float32),
        }, index=gene_list)

    label_gene_to_idx = {g: i for i, g in enumerate(labeled_genes)}
    label_values = np.array([gene_to_mean[g] for g in labeled_genes], dtype=np.float64)

    knn_mean = np.zeros(len(gene_list), dtype=np.float32)
    knn_weighted = np.zeros(len(gene_list), dtype=np.float32)

    for gi, gene in enumerate(gene_list):
        # Get correlation with all labeled genes
        gene_corr = np.array([
            corr_matrix[gi, gene_list.index(lg)]
            for lg in labeled_genes
        ])

        # Use absolute correlation to find most related genes (both positive
        # and negative co-expression can indicate functional relationship)
        abs_corr = np.abs(gene_corr)

        # Get top K (excluding self if gene itself is labeled)
        top_k_idx = np.argpartition(-abs_corr, min(k, len(abs_corr) - 1))[:k]
        # Sort by absolute correlation descending
        top_k_idx = top_k_idx[np.argsort(-abs_corr[top_k_idx])]

        top_corr = gene_corr[top_k_idx]
        top_labels = label_values[top_k_idx]

        # Simple mean
        knn_mean[gi] = float(np.mean(top_labels))

        # Correlation-weighted mean (use absolute correlation as weight)
        weights = np.abs(top_corr)
        weight_sum = weights.sum()
        if weight_sum > 1e-12:
            knn_weighted[gi] = float(np.dot(weights, top_labels) / weight_sum)
        else:
            knn_weighted[gi] = float(np.mean(top_labels))

    return pd.DataFrame({
        "gene_coexpr_knn_mean": knn_mean,
        "gene_coexpr_knn_weighted": knn_weighted,
    }, index=gene_list)


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


# ── Additive prediction model ────────────────────────────────────────────────


def build_gene_similarity_cf(
    train_labels: pd.DataFrame,
    gene_meta: pd.DataFrame,
    pathway_meta: pd.DataFrame,
    expression: pd.DataFrame,
    cold_genes: set[str],
    k: int = 20,
) -> dict[tuple[str, str], float]:
    """Build gene-similarity collaborative filtering predictions for cold genes.

    For each cold gene, finds K most pathway-similar warm genes and predicts
    cell-specific dependency as the weighted average of those warm genes'
    labels in each cell.

    This is the KEY innovation for cold-start genes: instead of predicting
    just the gene mean (teacher), we predict CELL-SPECIFIC values by
    transferring labels from pathway-similar warm genes.

    Args:
        train_labels: training labels DataFrame.
        gene_meta: gene metadata DataFrame.
        pathway_meta: pathway metadata DataFrame.
        expression: N_cells × P_genes expression DataFrame.
        cold_genes: set of cold gene symbols.
        k: number of similar warm genes to use.

    Returns:
        dict (cell_line_id, perturbation_gene) → predicted label.
        Only contains entries for cold genes that have warm neighbors.
    """
    from sklearn.neighbors import NearestNeighbors

    cold_genes = set(cold_genes)
    warm_genes = set(train_labels["perturbation_gene"].unique()) - cold_genes

    if not cold_genes or not warm_genes:
        return {}

    # Build PW140 gene features for similarity computation
    pw140 = build_pw140_membership_features(gene_meta, pathway_meta)
    warm_gene_list = sorted(warm_genes & set(pw140.index))
    cold_gene_list = sorted(cold_genes & set(pw140.index))

    if not warm_gene_list or not cold_gene_list:
        return {}

    pw_warm = pw140.reindex(warm_gene_list).fillna(0).to_numpy(dtype=np.float32)
    pw_cold = pw140.reindex(cold_gene_list).fillna(0).to_numpy(dtype=np.float32)

    # Fit KNN on warm gene features
    nbrs = NearestNeighbors(
        n_neighbors=min(k, len(warm_gene_list)), metric="cosine",
    )
    nbrs.fit(pw_warm)
    distances, indices = nbrs.kneighbors(pw_cold)

    # Build cold→warm similarity mapping
    cold_to_warm: dict[str, list[tuple[str, float]]] = {}
    for i, cold_gene in enumerate(cold_gene_list):
        sims = []
        for j in range(min(k, len(warm_gene_list))):
            warm_gene = warm_gene_list[indices[i, j]]
            sim = 1.0 - distances[i, j]  # cosine similarity → [0, 2]
            sims.append((warm_gene, max(sim, 1e-6)))
        total = sum(s for _, s in sims)
        if total > 0:
            cold_to_warm[cold_gene] = [(g, s / total) for g, s in sims]

    # Build cell×gene label lookup from training data
    cell_gene_label: dict[tuple[str, str], float] = {}
    for _, row in train_labels.iterrows():
        cell_gene_label[(row["cell_line_id"], row["perturbation_gene"])] = row["label"]

    # Predict cold genes in each training cell
    train_cells = sorted(train_labels["cell_line_id"].unique())
    cf_preds: dict[tuple[str, str], float] = {}

    for cold_gene in cold_gene_list:
        sims = cold_to_warm.get(cold_gene)
        if not sims:
            continue
        for cell in train_cells:
            weighted_sum = 0.0
            weight_total = 0.0
            for warm_gene, sim in sims:
                label = cell_gene_label.get((cell, warm_gene))
                if label is not None:
                    weighted_sum += sim * label
                    weight_total += sim
            if weight_total > 0:
                cf_preds[(cell, cold_gene)] = weighted_sum / weight_total

    return cf_preds


def predict_additive(
    pairs: pd.DataFrame,
    gene_baselines: dict[str, float],
    cell_biases: dict[str, float],
    svd_dot_products: dict[tuple[str, str], float] | None = None,
    svd_weight: float = 0.3,
    cf_predictions: dict[tuple[str, str], float] | None = None,
    cf_weight: float = 0.5,
    clip_range: tuple[float, float] = (-3.0, 5.0),
) -> np.ndarray:
    """Additive + CF prediction: ŷ(c,g) = base + CF_blend + γ·SVD.

    Gene mean explains 73.5% of label variance. Cell bias adds 2.0%.
    For cold genes, gene-similarity CF provides cell-specific signal.

    Blending (per pair):
        If CF prediction available: ŷ = (1-λ)·base + λ·CF + γ·SVD
        Otherwise: ŷ = base + γ·SVD
        where base = μ̂_g + β̂_c

    Args:
        pairs: DataFrame with [cell_line_id, perturbation_gene].
        gene_baselines: gene → μ̂_g dict.
        cell_biases: cell → β̂_c dict.
        svd_dot_products: (cell, gene) → U_c·V_g dict (optional).
        svd_weight: weight for SVD correction term.
        cf_predictions: (cell, gene) → CF predicted label dict (optional).
        cf_weight: blend weight for CF predictions (0=ignore, 1=CF only).
        clip_range: output clipping range.

    Returns:
        Array of predicted labels.
    """
    n = len(pairs)
    gene_arr = pairs["perturbation_gene"].to_numpy()
    cell_arr = pairs["cell_line_id"].to_numpy()

    # Build lookup arrays
    all_genes = np.unique(gene_arr)
    all_cells = np.unique(cell_arr)
    gene_to_idx = {g: i for i, g in enumerate(all_genes)}
    cell_to_idx = {c: i for i, c in enumerate(all_cells)}

    gene_vals = np.array([gene_baselines.get(g, 0.0) for g in all_genes], dtype=np.float32)
    cell_vals = np.array([cell_biases.get(c, 0.0) for c in all_cells], dtype=np.float32)

    gene_indices = np.array([gene_to_idx[g] for g in gene_arr])
    cell_indices = np.array([cell_to_idx[c] for c in cell_arr])

    preds = gene_vals[gene_indices] + cell_vals[cell_indices]

    # Blend with CF predictions for cold genes
    # CF predictions are full label predictions (weighted avg of warm gene labels),
    # so we blend them directly without re-adding cell bias.
    if cf_predictions is not None and cf_weight > 0:
        cf_vals = np.array([
            cf_predictions.get((c, g), preds[i])
            for i, (c, g) in enumerate(zip(cell_arr, gene_arr))
        ], dtype=np.float32)
        has_cf = np.array([
            (c, g) in cf_predictions
            for c, g in zip(cell_arr, gene_arr)
        ])
        # Blend: (1-λ)·base + λ·CF, where base = μ̂_g + β̂_c and CF is full label
        preds[has_cf] = (1.0 - cf_weight) * preds[has_cf] + cf_weight * cf_vals[has_cf]

    # Add SVD dot product correction
    if svd_dot_products is not None and svd_weight > 0:
        svd_vals = np.array([
            svd_dot_products.get((c, g), 0.0)
            for c, g in zip(cell_arr, gene_arr)
        ], dtype=np.float32)
        preds += svd_weight * svd_vals

    return np.clip(preds, clip_range[0], clip_range[1])
