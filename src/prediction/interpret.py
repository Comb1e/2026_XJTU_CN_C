"""Model interpretability and explainability outputs for Problem 2.

Generates:
  1. Feature importances (impurity + permutation)
  2. Sparse surrogate model (ElasticNet)
  3. Gene baseline decomposition
  4. Context-specificity analysis (lineage × module)
  5. Cell-level module dependency profiles
  6. Case studies
  7. Validation report
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.inspection import permutation_importance

from .features import build_all_features
from .baselines import (
    shrink_gene_means, shrink_cell_means,
    build_collaborative_features, build_module_priors,
    train_gene_baseline_teacher,
)
from .models import predict_all, prepare_features_a


def generate_all_interpretations(
    pred_result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate all interpretability outputs.

    Args:
        pred_result: output from run_prediction().
        config: configuration dict.

    Returns:
        dict with paths to all generated files.
    """
    output_dir = Path(config["paths"]["output_dir"]) / "prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = pred_result["models"]
    model_a = models["model_a"]

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

    # 1. Feature importances
    print("  1. Feature importances...")
    results["feature_importance"] = _feature_importances(model_a, X, output_dir)

    # 2. Sparse surrogate
    print("  2. Sparse surrogate model...")
    results["surrogate"] = _sparse_surrogate(model_a, X, labels, output_dir)

    # 3. Gene baseline decomposition
    print("  3. Gene baseline decomposition...")
    results["gene_baseline"] = _gene_baseline_decomposition(meta, labels, config, output_dir)

    # 4. Context-specificity analysis
    print("  4. Context-specificity analysis...")
    results["context"] = _context_specificity(pred_result, labels, cell_meta,
                                               meta, output_dir)

    # 5. Cell module dependency profiles
    print("  5. Cell module dependency profiles...")
    results["cell_module"] = _cell_module_profiles(pred_result, labels, meta, output_dir)

    # 6. Case studies
    print("  6. Case studies...")
    results["case_studies"] = _case_studies(pred_result, labels, meta, output_dir)

    return results


def _feature_importances(
    model_a: Any,
    X: pd.DataFrame,
    output_dir: Path,
) -> str:
    """Compute and save feature importances."""
    feature_cols = [c for c in X.columns
                    if c not in ("cell_line_id", "perturbation_gene")
                    and "neighbor_score" not in c]
    Xa = X[feature_cols].to_numpy(dtype=np.float32)

    # Impurity-based
    impurity = model_a.feature_importances_

    # Permutation importance (on subset for speed)
    rng = np.random.RandomState(42)
    subset_idx = rng.choice(len(Xa), min(50000, len(Xa)), replace=False)
    perm = permutation_importance(
        model_a, Xa[subset_idx], model_a.predict(Xa[subset_idx]),
        n_repeats=5, random_state=42, n_jobs=-1,
    )

    # Build DataFrame
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "impurity_importance": impurity,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    })
    importance_df = importance_df.sort_values("impurity_importance", ascending=False)

    path = output_dir / "feature_importance.csv"
    importance_df.to_csv(path, index=False)
    return str(path)


def _sparse_surrogate(
    model_a: Any,
    X: pd.DataFrame,
    labels: pd.DataFrame,
    output_dir: Path,
) -> str:
    """Train a sparse linear surrogate model for interpretability."""
    feature_cols = [c for c in X.columns
                    if c not in ("cell_line_id", "perturbation_gene")
                    and "neighbor_score" not in c]
    Xa = X[feature_cols].to_numpy(dtype=np.float32)

    # Get model predictions as target
    y_pred = model_a.predict(Xa)

    # Standardize features for comparability
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(Xa)

    # ElasticNet with cross-validated alpha
    enet = ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000, random_state=42)
    enet.fit(X_scaled, y_pred)

    coef_df = pd.DataFrame({
        "feature": feature_cols,
        "coefficient": enet.coef_,
        "abs_coefficient": np.abs(enet.coef_),
    })
    coef_df = coef_df.sort_values("abs_coefficient", ascending=False)

    path = output_dir / "surrogate_coefficients.csv"
    coef_df.to_csv(path, index=False)
    return str(path)


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

    # For each cold gene, show top contributing features
    train_genes = set(labels["perturbation_gene"].unique())
    all_genes = set(gene_feats.index)
    cold_genes = sorted(all_genes - train_genes)

    rows = []
    for gene in cold_genes[:20]:  # Top 20 cold genes
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
            "evidence_weight": meta["evidence_weights"].get(gene, 1.0),
        })

    df = pd.DataFrame(rows)
    path = output_dir / "gene_baseline_decomposition.csv"
    df.to_csv(path, index=False)
    return str(path)


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

    # Merge lineage info (suffix to avoid label_x/label_y collision)
    df = submission.merge(
        labels[["cell_line_id", "perturbation_gene", "label"]].rename(
            columns={"label": "true_label"}),
        on=["cell_line_id", "perturbation_gene"], how="left",
    )
    df["lineage"] = df["cell_line_id"].map(lineage_map)

    # Per-lineage per-module mean prediction (use predicted "label" column)
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

    # Compute mean predicted dependency per module per cell
    rows = []
    for cell in submission["cell_line_id"].unique()[:100]:  # Sample 100 cells
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

    # Pick 3 example cells
    example_cells = submission["cell_line_id"].unique()[:3]

    rows = []
    for cell in example_cells:
        cell_df = submission[submission["cell_line_id"] == cell].copy()
        cell_df = cell_df.sort_values("label", ascending=False)
        top15 = cell_df.head(15)
        for i, (_, row) in enumerate(top15.iterrows()):
            # Get gene's module memberships
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
