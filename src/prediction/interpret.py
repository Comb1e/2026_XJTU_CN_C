"""Model interpretability and explainability outputs for Problem 2.

All outputs are derived from interpretable white-box models:
  - Feature importance from ElasticNet coefficients, PCA-Ridge back-mapping
  - Factor Analysis gene loadings per latent factor
  - Gene baseline decomposition via RidgeCV teacher
  - Context-specificity analysis (lineage × module)
  - Cell-level module dependency profiles
  - Case studies
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

from .features import build_all_features
from .baselines import (
    shrink_gene_means, shrink_cell_means,
    build_collaborative_features,
    train_gene_baseline_teacher,
)
from .whitebox import (
    FactorAnalysisModel, SparseElasticNetModel, PCARidgeModel,
    SplineGAMModel, RidgeBlend,
)


def generate_all_interpretations(
    pred_result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate all interpretability outputs from white-box models.

    Args:
        pred_result: output from run_prediction().
        config: configuration dict.

    Returns:
        dict with paths to all generated files.
    """
    output_dir = Path(config["paths"]["output_dir"]) / "prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = pred_result["models"]
    components = models.get("components", {})
    blend = models.get("blend")

    # Reload data for analysis
    print("  Loading data for interpretation...")
    data_dir = Path(config["paths"]["data_dir"])
    labels = pd.read_csv(data_dir / "labels" / "gene_dependency.csv")
    cell_meta = pd.read_csv(data_dir / "metadata" / "cell_line_metadata.csv")

    # Build features
    print("  Building features...")
    X, meta = build_all_features(
        labels[["cell_line_id", "perturbation_gene"]], config,
    )
    gene_bl = shrink_gene_means(labels)
    cell_bl = shrink_cell_means(labels)
    g5 = build_collaborative_features(
        labels[["cell_line_id", "perturbation_gene"]], gene_bl, cell_bl,
    )
    for col in g5.columns:
        X[col] = g5[col].values if col not in X.columns else X[col]

    results = {}

    # 1. Component blend weights (key interpretability output)
    print("  1. Component blend weights...")
    results["blend_weights"] = _blend_weights(blend, output_dir)

    # 2. ElasticNet feature importance
    enet_model = components.get("elasticnet", {}).get("model")
    if enet_model is not None:
        print("  2. ElasticNet feature importance...")
        results["elasticnet_importance"] = _elasticnet_importance(enet_model, output_dir)

    # 3. PCA-Ridge feature importance
    pca_model = components.get("pca_ridge", {}).get("model")
    if pca_model is not None:
        print("  3. PCA-Ridge feature importance...")
        results["pca_ridge_importance"] = _pca_ridge_importance(pca_model, output_dir)

    # 4. Factor Analysis gene loadings
    fa_model = components.get("factor_analysis", {}).get("model")
    if fa_model is not None:
        print("  4. Factor Analysis gene loadings...")
        results["fa_loadings"] = _fa_loadings(fa_model, output_dir)

    # 5. Spline-GAM partial dependence
    spline_model = components.get("spline_gam", {}).get("model")
    if spline_model is not None:
        print("  5. Spline-GAM partial dependence...")
        results["spline_gam"] = _spline_partial_dependence(spline_model, output_dir)

    # 6. Gene baseline decomposition
    print("  6. Gene baseline decomposition...")
    results["gene_baseline"] = _gene_baseline_decomposition(meta, labels, config, output_dir)

    # 7. Context-specificity analysis
    print("  7. Context-specificity analysis...")
    results["context"] = _context_specificity(pred_result, labels, cell_meta,
                                               meta, output_dir)

    # 8. Cell module dependency profiles
    print("  8. Cell module dependency profiles...")
    results["cell_module"] = _cell_module_profiles(pred_result, labels, meta, output_dir)

    # 9. Case studies
    print("  9. Case studies...")
    results["case_studies"] = _case_studies(pred_result, labels, meta, output_dir)

    return results


# ── White-box component interpretability ────────────────────────────────────

def _blend_weights(blend: RidgeBlend | None, output_dir: Path) -> str:
    """Export RidgeCV blend weights showing component contributions."""
    if blend is None:
        return ""
    weights = blend.get_component_weights()
    df = pd.DataFrame(weights, columns=["component", "weight"])
    df["abs_weight"] = np.abs(df["weight"])
    df = df.sort_values("abs_weight", ascending=False)
    path = output_dir / "blend_weights.csv"
    df.to_csv(path, index=False)
    print(f"    Blend alpha: {blend.alpha_:.4f}")
    for _, row in df.iterrows():
        print(f"    {row['component']}: {row['weight']:.4f}")
    return str(path)


def _elasticnet_importance(
    enet_model: SparseElasticNetModel,
    output_dir: Path,
) -> str:
    """Export ElasticNet coefficients — directly interpretable feature weights."""
    top = enet_model.get_top_features(top_n=50)
    df = pd.DataFrame(top, columns=["feature", "coefficient"])
    df["abs_coefficient"] = np.abs(df["coefficient"])
    print(f"    {enet_model.n_nonzero_} nonzero / {len(enet_model.feature_names_)} features")
    path = output_dir / "elasticnet_importance.csv"
    df.to_csv(path, index=False)
    return str(path)


def _pca_ridge_importance(
    pca_model: PCARidgeModel,
    output_dir: Path,
) -> str:
    """Export PCA-Ridge back-mapped feature importance."""
    top = pca_model.get_top_features(top_n=50)
    df = pd.DataFrame(top, columns=["feature", "importance"])
    print(f"    {pca_model.n_components} components, "
          f"cumulative var={pca_model.explained_variance_ratio_.sum():.3f}")
    path = output_dir / "pca_ridge_importance.csv"
    df.to_csv(path, index=False)
    return str(path)


def _fa_loadings(
    fa_model: FactorAnalysisModel,
    output_dir: Path,
) -> str:
    """Export Factor Analysis gene loadings — per-factor top genes."""
    if fa_model.gene_loadings_ is None:
        return ""
    K = fa_model.gene_loadings_.shape[1]
    rows = []
    for k in range(min(K, 10)):  # Top 10 factors
        top_genes = fa_model.get_top_genes_per_factor(k, top_n=20)
        for rank, (gene, loading) in enumerate(top_genes):
            rows.append({
                "factor": k,
                "rank": rank + 1,
                "gene": gene,
                "loading": loading,
            })
    df = pd.DataFrame(rows)
    path = output_dir / "fa_gene_loadings.csv"
    df.to_csv(path, index=False)
    return str(path)


def _spline_partial_dependence(
    spline_model: SplineGAMModel,
    output_dir: Path,
) -> str:
    """Export Spline-GAM partial dependence curves for each selected feature."""
    rows = []
    for feat_idx in spline_model.selected_features_:
        name = (spline_model.feature_names_[feat_idx]
                if feat_idx < len(spline_model.feature_names_)
                else f"feat_{feat_idx}")
        x_grid, y_vals = spline_model.get_partial_dependence(feat_idx, n_points=50)
        for x, y in zip(x_grid, y_vals):
            rows.append({"feature": name, "x": float(x), "effect": float(y)})
    df = pd.DataFrame(rows)
    path = output_dir / "spline_partial_dependence.csv"
    df.to_csv(path, index=False)
    return str(path)


# ── Gene baseline decomposition ──────────────────────────────────────────────

def _gene_baseline_decomposition(
    meta: dict,
    labels: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> str:
    """Decompose gene baseline predictions into feature contributions."""
    gene_meta = meta.get("gene_meta")
    if gene_meta is None:
        gene_meta = pd.read_csv(
            Path(config["paths"]["metadata_dir"]) / "gene_metadata.csv"
        )

    from .features import build_gene_static_features, build_gene_expression_profile_features

    g1 = build_gene_static_features(
        gene_meta, meta["gene_module_map"], meta["evidence_weights"],
    )
    g2 = build_gene_expression_profile_features(meta["expression"])
    gene_feats = g1.join(g2, how="inner")

    teacher, oof_preds, _ = train_gene_baseline_teacher(g1, g2, labels)

    # For each cold gene, show predicted baseline
    train_genes = set(labels["perturbation_gene"].unique())
    all_genes = set(gene_feats.index)
    cold_genes = sorted(all_genes - train_genes)

    rows = []
    for gene in cold_genes[:50]:  # Top 50 cold genes
        if gene not in gene_feats.index:
            continue
        x = gene_feats.loc[gene].to_numpy(dtype=np.float32).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0)
        pred = teacher.predict(x)[0]

        # Module memberships
        modules = meta["gene_module_map"].get(gene, {}).get("modules", [])
        rows.append({
            "gene": gene,
            "predicted_baseline": pred,
            "n_modules": len(modules),
        })

    df = pd.DataFrame(rows)
    path = output_dir / "gene_baseline_decomposition.csv"
    df.to_csv(path, index=False)
    return str(path)


# ── Context-specificity analysis ────────────────────────────────────────────

def _context_specificity(
    pred_result: dict,
    labels: pd.DataFrame,
    cell_meta: pd.DataFrame,
    meta: dict,
    output_dir: Path,
) -> str:
    """Analyze lineage-specific dependency patterns (ref [2][5])."""
    submission = pred_result["submission"]
    lineage_map = cell_meta.set_index("cell_line_id")["OncotreeLineage"]

    df = submission.merge(
        labels[["cell_line_id", "perturbation_gene", "label"]].rename(
            columns={"label": "true_label"}),
        on=["cell_line_id", "perturbation_gene"], how="left",
    )
    df["lineage"] = df["cell_line_id"].map(lineage_map)

    gmm = meta.get("gene_module_map", {})
    n_modules = 14

    rows = []
    for lineage in sorted(df["lineage"].dropna().unique()):
        lin_df = df[df["lineage"] == lineage]
        for mod in range(n_modules):
            mod_genes = {g for g, info in gmm.items()
                        if mod in info.get("modules", [])}
            mod_mask = lin_df["perturbation_gene"].isin(mod_genes)
            if mod_mask.sum() > 0:
                mean_pred = lin_df.loc[mod_mask, "label"].mean()
                rows.append({
                    "lineage": lineage,
                    "module": mod,
                    "module_name": _module_name(mod),
                    "mean_dependency": mean_pred,
                    "n_pairs": int(mod_mask.sum()),
                })

    df_out = pd.DataFrame(rows)
    path = output_dir / "lineage_module_dependency.csv"
    df_out.to_csv(path, index=False)

    # Context-specificity index per module
    if len(df_out) > 0:
        csi = df_out.groupby("module")["mean_dependency"].agg(["std", "mean"])
        csi["csi"] = csi["std"] / (csi["mean"].abs() + 1e-8)
        csi_path = output_dir / "context_specificity_index.csv"
        csi.to_csv(csi_path)

    return str(path)


def _cell_module_profiles(
    pred_result: dict,
    labels: pd.DataFrame,
    meta: dict,
    output_dir: Path,
) -> str:
    """Compute per-cell module dependency profiles."""
    submission = pred_result["submission"]
    gmm = meta.get("gene_module_map", {})
    n_modules = 14

    rows = []
    for cell in submission["cell_line_id"].unique()[:100]:
        cell_df = submission[submission["cell_line_id"] == cell]
        for mod in range(n_modules):
            mod_genes = {g for g, info in gmm.items()
                        if mod in info.get("modules", [])}
            mod_mask = cell_df["perturbation_gene"].isin(mod_genes)
            if mod_mask.sum() > 0:
                mean_pred = cell_df.loc[mod_mask, "label"].mean()
                rows.append({
                    "cell_line_id": cell,
                    "module": mod,
                    "mean_predicted_dependency": mean_pred,
                })

    df = pd.DataFrame(rows)
    path = output_dir / "cell_module_dependency.csv"
    df.to_csv(path, index=False)
    return str(path)


def _case_studies(
    pred_result: dict,
    labels: pd.DataFrame,
    meta: dict,
    output_dir: Path,
) -> str:
    """Generate case studies for representative cell lines."""
    submission = pred_result["submission"]

    example_cells = submission["cell_line_id"].unique()[:3]

    rows = []
    for cell in example_cells:
        cell_df = submission[submission["cell_line_id"] == cell].copy()
        cell_df = cell_df.sort_values("label", ascending=False)
        top15 = cell_df.head(15)
        for i, (_, row) in enumerate(top15.iterrows()):
            gene = row["perturbation_gene"]
            modules = meta.get("gene_module_map", {}).get(gene, {}).get("modules", [])
            rows.append({
                "cell_line_id": cell,
                "rank": i + 1,
                "perturbation_gene": gene,
                "predicted_dependency": row["label"],
                "gene_modules": "|".join(str(m) for m in modules),
                "n_modules": len(modules),
            })

    df = pd.DataFrame(rows)
    path = output_dir / "case_studies.csv"
    df.to_csv(path, index=False)
    return str(path)


def _module_name(mod_idx: int) -> str:
    """Get module name from index."""
    names = [
        "OXPHOS_CI", "OXPHOS_CII_CIII", "OXPHOS_CIV_CV", "TCA_PYRUVATE",
        "FAO_LIPID", "AA_COFACTOR", "MITO_RIBOSOME", "mtDNA_RNA",
        "PROTEIN_IMPORT", "TRANSPORT", "REDOX_DETOX", "MITO_DYNAMICS",
        "CELL_DEATH", "SIGNALING",
    ]
    return names[mod_idx] if mod_idx < len(names) else f"Module_{mod_idx}"
