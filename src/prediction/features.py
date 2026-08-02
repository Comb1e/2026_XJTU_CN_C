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

    # Pre-compute module EWM scores per cell for Δ_EWM baseline term
    # M_k(c) = Σ_j w_j · z_{c,j} / Σ_j w_j  (weighted mean of module k genes in cell c)
    weighted_expr = expr_array * gene_weights[np.newaxis, :]  # (n_cells, n_genes)
    module_cell_scores = np.zeros((expr_array.shape[0], n_modules), dtype=np.float64)
    for k in range(n_modules):
        mask_k = gene_module_mask[:, k]
        module_cell_scores[:, k] = weighted_expr[:, mask_k].sum(axis=1) / max(module_weight_sums[k], 1e-12)

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
            # Fix: Δ_k = -w_g · (z_{c,g} - M_k(c)) / (Σw - w_g)
            # Previously missing the M_k module baseline term (median relative error 41%)
            M_k_c = module_cell_scores[vc[in_module], k]  # module baseline per pair
            dk_vals[valid_denom] = (
                -w_g_k[valid_denom] * (z_g_k[valid_denom] - M_k_c[valid_denom]) / denom[valid_denom]
            ).astype(np.float32)
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


def build_pathway_match_features(
    pairs: pd.DataFrame,
    expression: pd.DataFrame,
    gene_module_map: dict[str, dict],
    pathway_scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build pathway-level match features for (cell, gene) pairs.

    For each pair, looks up the gene's annotated MitoCarta pathways and
    computes the cell's mean/max score over those pathways.
    This is the 149-resolution analogue of the 14-dim module_match.

    Args:
        pairs: DataFrame with [cell_line_id, perturbation_gene].
        expression: N_cells × P_genes expression DataFrame (for cell index).
        gene_module_map: gene → module assignments dict with pathways_raw info.
        pathway_scores: N_cells × 149 pathways DataFrame (z-scored).
    """
    if pathway_scores is None:
        return pd.DataFrame(index=range(len(pairs)))

    cell_ids = pairs["cell_line_id"].tolist()
    gene_ids = pairs["perturbation_gene"].tolist()
    n_pairs = len(pairs)
    features = {}

    # Build gene→pathways mapping from pathways_raw field
    def _leaf_to_pw_key(leaf: str) -> str:
        """Convert leaf pathway name to metadata key format."""
        return leaf.replace(" ", "_").replace(",", "").replace("(", "").replace(")", "")

    gene_to_pathways: dict[str, list[str]] = {}
    for gene, info in gene_module_map.items():
        pw_raw = info.get("pathways_raw", "")
        if not pw_raw:
            continue
        pw_list = []
        for entry in pw_raw.split("|"):
            entry = entry.strip()
            if not entry or entry == "0" or entry.isdigit():
                continue
            parts = [p.strip() for p in entry.split(">")]
            leaf = parts[-1]
            if leaf and leaf != "0" and not leaf.isdigit():
                pw_key = _leaf_to_pw_key(leaf)
                pw_list.append(pw_key)
        if pw_list:
            gene_to_pathways[gene] = pw_list

    # Build pathway name → column index mapping
    pw_names = pathway_scores.columns.tolist()
    pw_to_idx = {name: i for i, name in enumerate(pw_names)}

    # Map cells to rows
    cell_idx_map = {c: i for i, c in enumerate(pathway_scores.index)}
    pw_array = pathway_scores.to_numpy(dtype=np.float32)

    pw_mean = np.zeros(n_pairs, dtype=np.float32)
    pw_max = np.zeros(n_pairs, dtype=np.float32)
    pw_count = np.zeros(n_pairs, dtype=np.float32)

    for i, (cell, gene) in enumerate(zip(cell_ids, gene_ids)):
        if cell not in cell_idx_map or gene not in gene_to_pathways:
            continue
        ci = cell_idx_map[cell]
        pw_indices = [pw_to_idx[pw] for pw in gene_to_pathways[gene]
                      if pw in pw_to_idx]
        if not pw_indices:
            continue
        scores = pw_array[ci, pw_indices]
        pw_mean[i] = float(np.mean(scores))
        pw_max[i] = float(np.max(scores))
        pw_count[i] = float(len(pw_indices))

    features["pair_pw_match_mean"] = pw_mean
    features["pair_pw_match_max"] = pw_max
    features["pair_pw_count"] = pw_count

    return pd.DataFrame(features, dtype=np.float32)


def _compute_res_delta_features(
    pairs: pd.DataFrame,
    expression: pd.DataFrame,
    gene_module_map: dict[str, dict],
    n_modules: int = 14,
) -> pd.DataFrame:
    """Compute Δ_RES knockout delta features using GSEA prefix-sum trick.

    For each (cell, gene) pair, computes the change in RES enrichment score
    per module when the gene is removed from the ranking (expression→0).

    Uses O(1) prefix-suffix max arrays per cell for speed.
    """
    cell_ids = pairs["cell_line_id"].tolist()
    gene_ids = pairs["perturbation_gene"].tolist()
    n_pairs = len(pairs)

    gene_list = expression.columns.tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    n_genes = len(gene_list)

    # Build module masks
    module_masks = np.zeros((n_modules, n_genes), dtype=bool)
    module_sizes = np.zeros(n_modules, dtype=np.int32)
    for gi, gene in enumerate(gene_list):
        if gene in gene_module_map:
            for mod_idx in gene_module_map[gene].get("modules", []):
                if mod_idx < n_modules:
                    module_masks[mod_idx, gi] = True
                    module_sizes[mod_idx] += 1

    expr_array = expression.to_numpy(dtype=np.float64)
    cell_to_row = {c: i for i, c in enumerate(expression.index)}

    # Pre-allocate output
    delta_res = np.zeros((n_pairs, n_modules), dtype=np.float32)

    # Process each unique cell once (cache RES state)
    cell_cache: dict[str, tuple] = {}
    unique_cells = sorted(set(cell_ids))

    for cell in unique_cells:
        if cell not in cell_to_row:
            continue
        ci = cell_to_row[cell]
        cell_expr = expr_array[ci]
        ranked = np.argsort(-cell_expr)  # descending
        rank_pos = np.zeros(n_genes, dtype=np.int32)
        rank_pos[ranked] = np.arange(n_genes, dtype=np.int32)

        # For each module, pre-compute running sums and ES
        mod_data = {}
        for k in range(n_modules):
            n_set = int(module_sizes[k])
            if n_set < 3:
                mod_data[k] = None
                continue
            mask = module_masks[k]

            # Running sum: hit_increment = n_genes - n_set, miss_decrement = n_set
            hit_inc = n_genes - n_set
            miss_dec = n_set

            # Build running sum at each position
            increments = np.where(mask[ranked], hit_inc, -miss_dec).astype(np.float64)
            running_sum = np.zeros(n_genes + 1, dtype=np.float64)
            running_sum[1:] = np.cumsum(increments)

            # Prefix max (inclusive at each position: max of RS[0..t])
            prefix_max = np.maximum.accumulate(running_sum)
            # Suffix max: max of RS[t..n_genes] starting from each position
            suffix_max = np.zeros(n_genes + 1, dtype=np.float64)
            suffix_max[-1] = running_sum[-1]
            for t in range(n_genes - 1, -1, -1):
                suffix_max[t] = max(running_sum[t], suffix_max[t + 1])

            # Original ES (max or min deviation, whichever is larger in abs)
            norm_factor = np.sqrt(n_set * (n_genes - n_set) / n_genes)
            if norm_factor < 1e-12:
                mod_data[k] = None
                continue
            max_rs = float(prefix_max[-1])
            min_rs = float(np.minimum.accumulate(running_sum)[-1])
            if abs(max_rs) >= abs(min_rs):
                orig_es = max_rs / norm_factor
                use_max = True
            else:
                orig_es = min_rs / norm_factor
                use_max = False

            mod_data[k] = (running_sum, prefix_max, suffix_max, increments,
                          orig_es, norm_factor, use_max)

        cell_cache[cell] = (rank_pos, mod_data)

    # Compute delta for each pair
    for i, (cell, gene) in enumerate(zip(cell_ids, gene_ids)):
        if cell not in cell_cache or gene not in gene_to_idx:
            continue
        rank_pos, mod_data = cell_cache[cell]
        gi = gene_to_idx[gene]
        p = int(rank_pos[gi])

        for k in range(n_modules):
            md = mod_data.get(k)
            if md is None:
                continue
            running_sum, prefix_max, suffix_max, increments, orig_es, norm, use_max = md

            # Remove gene at position p: RS'[t] = RS[t] for t<=p, RS[t]-inc for t>p
            inc_p = increments[p]
            # New max after removal
            new_max_before_p = prefix_max[p]  # max of RS[0..p]
            new_max_after_p = suffix_max[p + 1] - inc_p if p < n_genes else -1e18
            if use_max:
                new_es = max(new_max_before_p, new_max_after_p) / norm
            else:
                # For min-based ES, we need prefix_min and suffix_min similarly
                prefix_min = np.minimum.accumulate(running_sum)
                suffix_min = np.zeros(n_genes + 1, dtype=np.float64)
                suffix_min[-1] = running_sum[-1]
                for t in range(n_genes - 1, -1, -1):
                    suffix_min[t] = min(running_sum[t], suffix_min[t + 1])
                new_min_before_p = prefix_min[p]
                new_min_after_p = suffix_min[p + 1] - inc_p if p < n_genes else 1e18
                new_es = min(new_min_before_p, new_min_after_p) / norm

            delta_res[i, k] = float(new_es - orig_es)

    result = pd.DataFrame(
        {f"pair_delta_res_{k:02d}": delta_res[:, k] for k in range(n_modules)},
        dtype=np.float32,
    )
    return result


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

    # Optional Δ_RES features (GSEA knockout delta via prefix-sum trick)
    if feature_cfg.get("include_res_delta", False):
        print("  Building Δ_RES knockout delta features...")
        g4_res = _compute_res_delta_features(
            pairs, expression, gene_module_map, n_modules=14,
        )
        g4 = pd.concat([g4, g4_res], axis=1)

    # Pathway-level match features (149-resolution module match)
    if feature_cfg.get("include_pathway_match", True):
        print("  Building pathway-level match features...")
        pw_scores_path = features_dir / "cell_pathway_scores_zscore.csv"
        if pw_scores_path.exists():
            pw_scores = pd.read_csv(pw_scores_path, index_col=0)
            g4_pw = build_pathway_match_features(
                pairs, expression, gene_module_map, pw_scores,
            )
            g4 = pd.concat([g4, g4_pw], axis=1)

    # Cross features: gene×cell state interactions
    if feature_cfg.get("include_cross_features", True):
        print("  Building cross features (gene×cell interactions)...")
        indicators_df = pd.read_csv(outputs_dir / "cell_line_indicators.csv", index_col=0)
        pair_z = g4["pair_z_cg"].to_numpy(dtype=np.float32)
        pair_expr_pct = g4["pair_expr_percentile"].to_numpy(dtype=np.float32)
        cross_feats = {}
        # Top indicator interactions with z_cg
        ind_cols = indicators_df.columns[:14]
        ind_arr = indicators_df.reindex(pairs["cell_line_id"]).to_numpy(dtype=np.float32)
        for k in range(14):
            cross_feats[f"pair_z_x_ind_{k:02d}"] = (pair_z * ind_arr[:, k]).astype(np.float32)
        # Expression percentile × evidence weight
        gene_ew = np.array([evidence_weights.get(g, 1.0) for g in pairs["perturbation_gene"]], dtype=np.float32)
        cross_feats["pair_expr_pct_x_ew"] = (pair_expr_pct * gene_ew).astype(np.float32)
        g4_cross = pd.DataFrame(cross_feats, dtype=np.float32)
        g4 = pd.concat([g4, g4_cross], axis=1)

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
