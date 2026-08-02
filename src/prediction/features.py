"""Feature engineering for (cell_line, perturbation_gene) pairs.

Assembles ~220 features organized into five groups:
  G1 — Gene static features (~43 dims)
  G2 — Gene expression-profile features (~8 dims)
  G3 — Cell state features (~115 dims)
  G4 — Pair features (~44 dims) — the mechanistic core
  G5 — Collaborative / label-derived features (~7 dims, leakage-controlled)

All features are built as float32 chunks to bound memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ── G1: Gene static features ──────────────────────────────────────────────

# Yeast homolog mapping (from preprocess.py)
YEAST_HOMOLOG_CATS = [
    "OrthologMitoHighConf", "OrthologMitoLowConf",
    "HomologMitoHighConf", "HomologMitoLowConf",
    "Homolog", "Ortholog",
]
RICKETTSIA_HOMOLOG_CATS = ["Ortholog", "Homolog"]
MITO_DOMAIN_CATS = ["MitoDomain", "SharedDomain"]
MSMS_PURITY_CATS = ["75-100pure", "50-75pure", "25-50pure"]
SUBMITO_LOCATIONS = ["Matrix", "MIM", "MOM", "IMS", "Membrane"]


def build_gene_static_features(
    gene_meta: pd.DataFrame,
    gene_module_map: dict[str, dict],
    evidence_weights: dict[str, float],
) -> pd.DataFrame:
    """Build G1 gene static features.

    Returns DataFrame indexed by gene_symbol with feature columns.
    """
    genes = gene_meta["gene_symbol"].tolist()
    features = {}

    # Module membership (14-dim multi-label one-hot)
    n_modules = 14
    mod_matrix = np.zeros((len(genes), n_modules), dtype=np.float32)
    for i, gene in enumerate(genes):
        if gene in gene_module_map:
            for mod_idx in gene_module_map[gene].get("modules", []):
                if mod_idx < n_modules:
                    mod_matrix[i, mod_idx] = 1.0
    for k in range(n_modules):
        features[f"gene_module_{k:02d}"] = mod_matrix[:, k]

    # Sub-mitochondrial location one-hot
    sub_mito = gene_meta.set_index("gene_symbol")["sub_mito_location"]
    for loc in SUBMITO_LOCATIONS:
        col_name = f"gene_submito_{loc}"
        features[col_name] = np.array([
            1.0 if (gene in sub_mito.index and loc in str(sub_mito[gene])) else 0.0
            for gene in genes
        ], dtype=np.float32)

    # Evidence weight
    features["gene_evidence_weight"] = np.array(
        [evidence_weights.get(g, 1.0) for g in genes], dtype=np.float32
    )

    # Pathway count
    features["gene_pathway_count"] = np.array([
        len(str(gene_meta.set_index("gene_symbol").loc[g, "pathways"]).split("|"))
        if g in gene_meta.set_index("gene_symbol").index
        and pd.notna(gene_meta.set_index("gene_symbol").loc[g, "pathways"])
        else 0
        for g in genes
    ], dtype=np.float32)

    # Curated gene list flag
    curated_col = "is_curated_gene_list" if "is_curated_gene_list" in gene_meta.columns else None
    if curated_col and curated_col in gene_meta.columns:
        curated_map = gene_meta.set_index("gene_symbol")[curated_col]
        features["gene_curated"] = np.array([
            1.0 if (g in curated_map.index and curated_map[g] == 1) else 0.0
            for g in genes
        ], dtype=np.float32)
    else:
        features["gene_curated"] = np.zeros(len(genes), dtype=np.float32)

    # Yeast homolog one-hot
    yeast_col = "yeast_mito_homolog_score"
    yeast_map = gene_meta.set_index("gene_symbol").get(yeast_col, pd.Series(dtype=str))
    for cat in YEAST_HOMOLOG_CATS:
        features[f"gene_yeast_{cat}"] = np.array(
            [1.0 if (g in yeast_map.index and str(yeast_map[g]) == cat) else 0.0
             for g in genes], dtype=np.float32
        )

    # Rickettsia homolog one-hot
    rick_col = "rickettsia_homolog_score"
    rick_map = gene_meta.set_index("gene_symbol").get(rick_col, pd.Series(dtype=str))
    for cat in RICKETTSIA_HOMOLOG_CATS:
        features[f"gene_rickettsia_{cat}"] = np.array(
            [1.0 if (g in rick_map.index and str(rick_map[g]) == cat) else 0.0
             for g in genes], dtype=np.float32
        )

    # Mito domain one-hot
    md_col = "mito_domain_score"
    md_map = gene_meta.set_index("gene_symbol").get(md_col, pd.Series(dtype=str))
    for cat in MITO_DOMAIN_CATS:
        features[f"gene_mitodomain_{cat}"] = np.array(
            [1.0 if (g in md_map.index and str(md_map[g]) == cat) else 0.0
             for g in genes], dtype=np.float32
        )

    # MS/MS purity one-hot
    msms_col = "msms_score"
    msms_map = gene_meta.set_index("gene_symbol").get(msms_col, pd.Series(dtype=str))
    for cat in MSMS_PURITY_CATS:
        features[f"gene_msms_{cat}"] = np.array(
            [1.0 if (g in msms_map.index and str(msms_map[g]) == cat) else 0.0
             for g in genes], dtype=np.float32
        )

    # Numeric scores (with missing imputation)
    for col in ["targetp_score", "coexpression_gnf_n50_score"]:
        if col in gene_meta.columns:
            vals = gene_meta.set_index("gene_symbol")[col]
            arr = np.array([vals.get(g, np.nan) for g in genes], dtype=np.float32)
            mask = np.isnan(arr)
            arr[mask] = np.nanmean(arr) if not np.all(mask) else 0.0
            features[f"gene_{col}"] = arr
            features[f"gene_{col}_missing"] = mask.astype(np.float32)

    # pgc_induction_score if available
    if "pgc_induction_score" in gene_meta.columns:
        vals = gene_meta.set_index("gene_symbol")["pgc_induction_score"]
        arr = np.array([vals.get(g, np.nan) for g in genes], dtype=np.float32)
        mask = np.isnan(arr)
        arr[mask] = np.nanmean(arr) if not np.all(mask) else 0.0
        features["gene_pgc_induction"] = arr
        features["gene_pgc_induction_missing"] = mask.astype(np.float32)

    result = pd.DataFrame(features, index=genes)
    result.index.name = "gene_symbol"
    return result


# ── G2: Gene expression-profile features ────────────────────────────────────


def build_gene_expression_profile_features(
    expression: pd.DataFrame,
) -> pd.DataFrame:
    """Build G2 gene expression-profile features.

    Computes per-gene statistics across all cell lines from z-scored expression.
    """
    genes = expression.columns.tolist()
    expr = expression.to_numpy(dtype=np.float32)

    features = {}
    features["gene_expr_mean"] = expr.mean(axis=0)
    features["gene_expr_std"] = expr.std(axis=0)
    features["gene_expr_min"] = expr.min(axis=0)
    features["gene_expr_max"] = expr.max(axis=0)
    features["gene_expr_q25"] = np.percentile(expr, 25, axis=0).astype(np.float32)
    features["gene_expr_q50"] = np.percentile(expr, 50, axis=0).astype(np.float32)
    features["gene_expr_q75"] = np.percentile(expr, 75, axis=0).astype(np.float32)
    features["gene_expr_frac_positive"] = (expr > 0).mean(axis=0)

    result = pd.DataFrame(features, index=genes)
    result.index.name = "gene_symbol"
    return result


# ── G3: Cell state features ─────────────────────────────────────────────────


def build_cell_features(
    outputs_dir: str | Path,
    cell_line_ids: list[str],
    pathway_pca_components: int = 20,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build G3 cell state features from Problem 1 outputs.

    Loads the CSVs produced by the Problem 1 pipeline and assembles
    cell-level features for the requested cell lines.
    """
    outputs_dir = Path(outputs_dir)
    features = {}
    idx = pd.Index(cell_line_ids, name="cell_line_id")

    # 14 orthogonalized indicators
    indicators = pd.read_csv(outputs_dir / "cell_line_indicators.csv", index_col=0)
    for col in indicators.columns:
        features[f"cell_indicator_{col}"] = indicators.reindex(idx)[col].to_numpy(dtype=np.float32)

    # 10 derived indicators
    derived_path = outputs_dir / "derived_indicators.csv"
    if derived_path.exists():
        derived = pd.read_csv(derived_path, index_col=0)
        for col in derived.columns:
            features[f"cell_derived_{col}"] = derived.reindex(idx)[col].to_numpy(dtype=np.float32)

    # 14 lineage-conditioned indicators
    lineage_path = outputs_dir / "cell_line_indicators_lineage.csv"
    if lineage_path.exists():
        lineage = pd.read_csv(lineage_path, index_col=0)
        for col in lineage.columns:
            features[f"cell_lineage_{col}"] = lineage.reindex(idx)[col].to_numpy(dtype=np.float32)

    # Raw EWM/RES/SPCA scores
    for score_type in ["ewm", "res", "spca"]:
        score_path = outputs_dir / f"{score_type}_scores.csv"
        if score_path.exists():
            scores = pd.read_csv(score_path, index_col=0)
            for col in scores.columns:
                features[f"cell_{score_type}_{col}"] = scores.reindex(idx)[col].to_numpy(dtype=np.float32)

    # Pathway scores → PCA
    pw_path = outputs_dir / "pathway_scores_149.csv"
    if pw_path.exists():
        pw = pd.read_csv(pw_path, index_col=0)
        pw_aligned = pw.reindex(idx).to_numpy(dtype=np.float32)
        # PCA fitted on all 1,140 cells, then transform requested cells
        pca = PCA(n_components=min(pathway_pca_components, pw.shape[1]),
                  random_state=random_state)
        pca.fit(pw.to_numpy(dtype=np.float32))
        pw_pca = pca.transform(pw_aligned)
        for k in range(pw_pca.shape[1]):
            features[f"cell_pw_pca_{k:02d}"] = pw_pca[:, k].astype(np.float32)

    return pd.DataFrame(features, index=idx)


def build_lineage_onehot(
    cell_meta: pd.DataFrame,
    cell_line_ids: list[str],
) -> pd.DataFrame:
    """Build lineage one-hot features from cell metadata."""
    lineage_map = cell_meta.set_index("cell_line_id")["OncotreeLineage"]
    lineages = sorted(lineage_map.dropna().unique())
    idx = pd.Index(cell_line_ids, name="cell_line_id")
    data = {}
    for lin in lineages:
        data[f"cell_lineage_onehot_{lin}"] = np.array(
            [1.0 if lineage_map.get(c, "") == lin else 0.0 for c in cell_line_ids],
            dtype=np.float32,
        )
    return pd.DataFrame(data, index=idx)


# ── G4: Pair features ───────────────────────────────────────────────────────


def build_pair_features(
    pairs: pd.DataFrame,
    expression: pd.DataFrame,
    gene_module_map: dict[str, dict],
    evidence_weights: dict[str, float],
    ewm_scores: pd.DataFrame,
    spca_loadings: dict[int, dict[str, float]] | None = None,
    lineage_indicators: pd.DataFrame | None = None,
    n_modules: int = 14,
) -> pd.DataFrame:
    """Build G4 pair features for (cell_line, perturbation_gene) pairs.

    Fully vectorized — no Python loops over pairs.
    """
    cell_ids = pairs["cell_line_id"].tolist()
    gene_ids = pairs["perturbation_gene"].tolist()
    n_pairs = len(pairs)
    features = {}

    # Build lookup arrays
    cell_list = expression.index.tolist()
    gene_list = expression.columns.tolist()
    cell_to_idx = {c: i for i, c in enumerate(cell_list)}
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}

    # Map pair (cell, gene) → matrix indices
    cell_indices = np.array([cell_to_idx.get(c, -1) for c in cell_ids], dtype=np.intp)
    gene_indices = np.array([gene_to_idx.get(g, -1) for g in gene_ids], dtype=np.intp)
    valid_mask = (cell_indices >= 0) & (gene_indices >= 0)

    expr_array = expression.to_numpy(dtype=np.float32)

    # 1. z_{c,g}: expression z-score via numpy advanced indexing
    z_cg = np.zeros(n_pairs, dtype=np.float32)
    vc = cell_indices[valid_mask]
    vg = gene_indices[valid_mask]
    z_cg[valid_mask] = expr_array[vc, vg]
    features["pair_z_cg"] = z_cg

    # 2. Expression percentile: fraction of genes with lower expression in same cell
    expr_percentile = np.zeros(n_pairs, dtype=np.float32)
    z_vals = z_cg[valid_mask]
    # For each valid pair, count how many columns in that cell's row are < z_val
    cell_rows = expr_array[vc]  # (n_valid, n_genes)
    expr_percentile[valid_mask] = (cell_rows < z_vals[:, np.newaxis]).mean(axis=1)
    features["pair_expr_percentile"] = expr_percentile

    # 3–5: Pre-compute gene×module membership matrix and weights
    gene_module_mask = np.zeros((len(gene_list), n_modules), dtype=bool)
    gene_weights = np.ones(len(gene_list), dtype=np.float64)
    for gi, gene in enumerate(gene_list):
        if gene in gene_module_map:
            for mod_idx in gene_module_map[gene].get("modules", []):
                if mod_idx < n_modules:
                    gene_module_mask[gi, mod_idx] = True
        gene_weights[gi] = evidence_weights.get(gene, 1.0)

    module_weight_sums = (gene_module_mask.astype(np.float64) * gene_weights[:, np.newaxis]).sum(axis=0)

    # For each valid pair, get its gene's module mask and weight
    pair_gene_mask = gene_module_mask[vg]  # (n_valid, 14) bool
    pair_gene_w = gene_weights[vg]         # (n_valid,) float
    pair_z = z_cg[valid_mask]              # (n_valid,) float

    # 3. Δ_EWM vectorized for all 14 modules
    for k in range(n_modules):
        delta_k = np.zeros(n_pairs, dtype=np.float32)
        total_w = module_weight_sums[k]
        in_module = pair_gene_mask[:, k]  # shape (n_valid,) bool
        if in_module.any():
            w_g_k = pair_gene_w[in_module]          # weights of genes in module
            z_g_k = pair_z[in_module]                # z-scores for those pairs
            denom = total_w - w_g_k
            valid_denom = denom > 0
            dk_vals = np.zeros(in_module.sum(), dtype=np.float32)
            dk_vals[valid_denom] = (-w_g_k[valid_denom] * z_g_k[valid_denom] / denom[valid_denom]).astype(np.float32)
            # Place results back: for valid pairs, scatter into in_module positions
            valid_positions = np.where(valid_mask)[0]
            delta_k[valid_positions[in_module]] = dk_vals
        features[f"pair_delta_ewm_{k:02d}"] = delta_k

    # 4. Δ_SPCA vectorized
    if spca_loadings is not None:
        gene_loading_vec = np.zeros(len(gene_list), dtype=np.float32)
        for k in range(n_modules):
            delta_k = np.zeros(n_pairs, dtype=np.float32)
            loadings = spca_loadings.get(k, {})
            for gi, gene in enumerate(gene_list):
                gene_loading_vec[gi] = loadings.get(gene, 0.0)
            pair_loadings = gene_loading_vec[vg]    # shape (n_valid,)
            has_loading = pair_loadings != 0
            if has_loading.any():
                dk_vals = (-pair_loadings[has_loading] * pair_z[has_loading]).astype(np.float32)
                valid_positions = np.where(valid_mask)[0]
                delta_k[valid_positions[has_loading]] = dk_vals
            features[f"pair_delta_spca_{k:02d}"] = delta_k
    else:
        for k in range(n_modules):
            features[f"pair_delta_spca_{k:02d}"] = np.zeros(n_pairs, dtype=np.float32)

    # 5. Module-match: cell's EWM score if gene belongs to module k
    ewm_array = ewm_scores.to_numpy(dtype=np.float32)
    ewm_cell_rows = ewm_array[vc]  # (n_valid, 14)
    valid_positions = np.where(valid_mask)[0]
    for k in range(n_modules):
        mm_k = np.zeros(n_pairs, dtype=np.float32)
        in_module = pair_gene_mask[:, k]
        if in_module.any():
            mm_k[valid_positions[in_module]] = ewm_cell_rows[in_module, k]
        features[f"pair_module_match_{k:02d}"] = mm_k

    # 6. Lineage × module interaction
    if lineage_indicators is not None:
        lin_array = lineage_indicators.to_numpy(dtype=np.float32)
        lin_cell_to_idx = {c: i for i, c in enumerate(lineage_indicators.index)}
        lin_cell_indices = np.array([lin_cell_to_idx.get(c, -1) for c in cell_ids], dtype=np.intp)
        lin_valid = lin_cell_indices >= 0
        lin_interact = np.zeros(n_pairs, dtype=np.float32)
        # Use combined valid mask
        combined_valid = valid_mask & lin_valid
        if combined_valid.any():
            comb_positions = np.where(combined_valid)[0]
            vl = lin_cell_indices[combined_valid]  # indices into lin_array
            vm = pair_gene_mask[lin_valid[valid_mask]]  # module mask for these rows
            lin_rows = lin_array[vl]  # (n_cv, 14)
            n_gene_mods = vm.sum(axis=1).astype(np.float32)
            has_mods = n_gene_mods > 0
            if has_mods.any():
                lin_interact[comb_positions[has_mods]] = (
                    (lin_rows[has_mods] * vm[has_mods]).sum(axis=1) / n_gene_mods[has_mods]
                ).astype(np.float32)
        features["pair_lineage_module_interact"] = lin_interact
    else:
        features["pair_lineage_module_interact"] = np.zeros(n_pairs, dtype=np.float32)

    return pd.DataFrame(features)


# ── Full feature table assembly ─────────────────────────────────────────────


def assemble_feature_table(
    pairs: pd.DataFrame,
    gene_static_features: pd.DataFrame,
    gene_expr_profile_features: pd.DataFrame,
    cell_features: pd.DataFrame,
    lineage_onehot: pd.DataFrame,
    pair_features: pd.DataFrame,
    collaborative_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assemble the complete feature table for (cell, gene) pairs.

    Merges G1-G5 features by joining on gene_symbol (G1, G2) and
    cell_line_id (G3), with G4/G5 already aligned to pair order.
    """
    n_pairs = len(pairs)
    cell_ids = pairs["cell_line_id"].tolist()
    gene_ids = pairs["perturbation_gene"].tolist()

    # G1: Gene static features (join by gene)
    g1_arr = gene_static_features.reindex(gene_ids).to_numpy(dtype=np.float32)
    g1_cols = list(gene_static_features.columns)

    # G2: Gene expression profile features (join by gene)
    g2_arr = gene_expr_profile_features.reindex(gene_ids).to_numpy(dtype=np.float32)
    g2_cols = list(gene_expr_profile_features.columns)

    # G3: Cell features (join by cell)
    cell_feats = pd.concat([cell_features, lineage_onehot], axis=1)
    cell_arr = cell_feats.reindex(cell_ids).to_numpy(dtype=np.float32)
    cell_cols = list(cell_feats.columns)

    # G4: Pair features (already aligned)
    pair_arr = pair_features.to_numpy(dtype=np.float32)
    pair_cols = list(pair_features.columns)

    # Assemble with proper group prefixes
    arrays = []
    all_cols = []
    arrays.append(g1_arr)
    all_cols.extend([f"g1_{c}" for c in g1_cols])
    arrays.append(g2_arr)
    all_cols.extend([f"g2_{c}" for c in g2_cols])
    arrays.append(cell_arr)
    all_cols.extend([f"g3_{c}" for c in cell_cols])
    arrays.append(pair_arr)
    all_cols.extend([f"g4_{c}" for c in pair_cols])

    # G5: Collaborative features (aligned)
    if collaborative_features is not None and len(collaborative_features) > 0:
        g5_arr = collaborative_features.to_numpy(dtype=np.float32)
        arrays.append(g5_arr)
        all_cols.extend([f"g5_{c}" for c in collaborative_features.columns])

    # Stack horizontally
    X = np.column_stack(arrays) if len(arrays) > 1 else arrays[0]

    result = pd.DataFrame(X, columns=all_cols, dtype=np.float32)
    result.insert(0, "cell_line_id", cell_ids)
    result.insert(1, "perturbation_gene", gene_ids)

    return result


def build_all_features(
    pairs: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the complete feature table from config and pairs DataFrame.

    This is the main entry point for feature assembly. Reads all Problem 1
    outputs, builds G1-G5, and returns the full feature matrix plus metadata.

    Args:
        pairs: DataFrame with columns [cell_line_id, perturbation_gene].
        config: full configuration dict (from config.yaml).

    Returns:
        (features_df, metadata) where metadata contains feature group info,
        gene_module_map, evidence_weights, etc.
    """
    pred_cfg = config.get("prediction", {})
    feature_cfg = pred_cfg.get("features", {})
    data_dir = Path(config["paths"]["data_dir"])
    outputs_dir = Path(config["paths"]["output_dir"])
    metadata_dir = Path(config["paths"]["metadata_dir"])

    # Load gene metadata
    gene_meta = pd.read_csv(metadata_dir / "gene_metadata.csv")

    # Build gene-module map and evidence weights (reuse Problem 1 code)
    from ..preprocess import build_gene_module_map, compute_evidence_weights
    pathway_meta = pd.read_csv(metadata_dir / "pathway_metadata.csv")
    gene_module_map = build_gene_module_map(gene_meta, pathway_meta)
    evidence_weights = compute_evidence_weights(gene_meta, config)

    # Load expression
    features_dir = Path(config["paths"]["features_dir"])
    expression = pd.read_csv(
        features_dir / "cell_expression_zscore.csv", index_col=0
    )

    # Load EWM scores (needed for module-match and cell features)
    ewm_scores = pd.read_csv(outputs_dir / "ewm_scores.csv", index_col=0)

    # Load SPCA loadings if available
    spca_loadings = None
    spca_path = outputs_dir / "spca_scores.csv"
    if spca_path.exists():
        from ..scoring.spca import compute_spca_scores
        # Recompute to get loadings
        spca_result = compute_spca_scores(expression, gene_module_map, n_modules=14)
        spca_loadings = spca_result.attrs.get("loadings", {})

    # Load lineage indicators
    lineage_indicators = None
    lin_path = outputs_dir / "cell_line_indicators_lineage.csv"
    if lin_path.exists():
        lineage_indicators = pd.read_csv(lin_path, index_col=0)

    print("  Building G1: gene static features...")
    g1 = build_gene_static_features(gene_meta, gene_module_map, evidence_weights)

    print("  Building G2: gene expression profile features...")
    g2 = build_gene_expression_profile_features(expression)

    unique_cells = sorted(pairs["cell_line_id"].unique())
    print(f"  Building G3: cell state features for {len(unique_cells)} cells...")
    g3 = build_cell_features(
        outputs_dir, unique_cells,
        pathway_pca_components=feature_cfg.get("pathway_pca_components", 20),
    )

    cell_meta = pd.read_csv(metadata_dir / "cell_line_metadata.csv")
    g3_lineage = build_lineage_onehot(cell_meta, unique_cells)

    print("  Building G4: pair features...")
    g4 = build_pair_features(
        pairs, expression, gene_module_map, evidence_weights,
        ewm_scores, spca_loadings, lineage_indicators, n_modules=14,
    )

    print("  Assembling feature table...")
    X = assemble_feature_table(pairs, g1, g2, g3, g3_lineage, g4)

    # Check no NaN
    nan_count = int(X.isna().sum().sum())
    if nan_count > 0:
        # Find columns with NaN and fill with 0
        nan_cols = [c for c in X.columns if X[c].isna().any()]
        print(f"  Warning: {nan_count} NaN values in {len(nan_cols)} columns, filling with 0")
        X[nan_cols] = X[nan_cols].fillna(0.0)

    metadata = {
        "gene_module_map": gene_module_map,
        "evidence_weights": evidence_weights,
        "feature_groups": {"G1": len(g1.columns), "G2": len(g2.columns),
                           "G3_cell": len(g3.columns), "G3_lineage": len(g3_lineage.columns),
                           "G4": len(g4.columns)},
        "expression": expression,
        "ewm_scores": ewm_scores,
        "spca_loadings": spca_loadings,
        "lineage_indicators": lineage_indicators,
        "cell_meta": cell_meta,
        "gene_meta": gene_meta,
        "pathway_meta": pathway_meta,
    }

    return X, metadata
