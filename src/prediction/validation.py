"""Cross-validation framework for gene dependency prediction — formula edition.

Three protocols:
  1. Group-by-gene 5-fold — primary, simulates cold-start
  2. Group-by-cell 5-fold — secondary, simulates new cell lines
  3. Random-pair split — auxiliary, fast hyperparameter tuning

All protocols use interpretable formula-based models exclusively.
All out-of-fold computation prevents leakage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .metrics import compute_metrics_df
from .baselines import (
    compute_loco_gene_means, shrink_cell_means,
    train_gene_baseline_teacher, predict_gene_baselines,
    build_collaborative_features, build_gene_similarity_cf,
    compute_label_svd, impute_gene_factors,
)
from .features import (
    build_gene_static_features, build_gene_expression_profile_features,
    build_cell_features, build_lineage_onehot,
)


# ── Shared data loading ─────────────────────────────────────────────────────

def _load_supporting_data(config: dict[str, Any]) -> dict[str, Any]:
    """Load metadata, expression, and gene features shared across folds."""
    data_dir = Path(config["paths"]["data_dir"])
    outputs_dir = Path(config["paths"]["output_dir"])

    gene_meta = pd.read_csv(data_dir / "metadata" / "gene_metadata.csv")
    pathway_meta = pd.read_csv(data_dir / "metadata" / "pathway_metadata.csv")
    cell_meta = pd.read_csv(data_dir / "metadata" / "cell_line_metadata.csv")
    expression = pd.read_csv(
        data_dir / "features" / "cell_expression_zscore.csv", index_col=0,
    )

    from ..preprocess import build_gene_module_map, compute_evidence_weights
    gmm = build_gene_module_map(gene_meta)
    ew = compute_evidence_weights(gene_meta)
    g1 = build_gene_static_features(gene_meta, gmm, ew)
    g2 = build_gene_expression_profile_features(expression)

    return {
        "gene_meta": gene_meta,
        "pathway_meta": pathway_meta,
        "cell_meta": cell_meta,
        "expression": expression,
        "gene_module_map": gmm,
        "evidence_weights": ew,
        "g1": g1,
        "g2": g2,
    }


def _build_teacher(
    train_labels: pd.DataFrame,
    g1: pd.DataFrame,
    g2: pd.DataFrame,
    gene_meta: pd.DataFrame,
    pathway_meta: pd.DataFrame,
    expression: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[Any, dict[str, float], pd.DataFrame]:
    """Build gene baseline teacher with all available gene-level features."""
    from .baselines import (
        build_pw140_membership_features, compute_coexpression_knn_features,
        build_description_keyword_features,
    )
    pw140 = build_pw140_membership_features(gene_meta, pathway_meta)
    knn_k = config.get("prediction", {}).get("baselines", {}).get("knn_k", 20)
    coexpr_knn = compute_coexpression_knn_features(expression, train_labels, k=knn_k)
    desc_feats = build_description_keyword_features(gene_meta)
    teacher_extra = pd.concat([pw140, coexpr_knn, desc_feats], axis=1)
    teacher, oof_preds, gene_feats = train_gene_baseline_teacher(
        g1, g2, train_labels, extra_features=teacher_extra,
    )
    return teacher, oof_preds, gene_feats


# ── Protocol 1: Group-by-gene CV ────────────────────────────────────────────

def validate_group_by_gene(
    X: pd.DataFrame,
    y: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Primary CV: hold out 20% of genes entirely per fold.

    Simulates cold-start generalization. Uses interpretable formula-based models.
    """
    if config is None:
        config = {}
    data = _load_supporting_data(config)
    gene_meta = data["gene_meta"]
    pathway_meta = data["pathway_meta"]
    cell_meta = data["cell_meta"]
    expression = data["expression"]
    g1 = data["g1"]
    g2 = data["g2"]

    unique_genes = np.unique(gene_ids)
    kf = GroupKFold(n_splits=n_folds)
    all_metrics = []
    all_folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(np.arange(len(y)), groups=gene_ids)
    ):
        train_genes_set = set(gene_ids[train_idx])
        val_genes_set = set(gene_ids[val_idx])
        val_cold = val_genes_set - train_genes_set

        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]
        train_cell_arr = cell_ids[train_idx]
        train_gene_arr = gene_ids[train_idx]
        val_cell_arr = cell_ids[val_idx]
        val_gene_arr = gene_ids[val_idx]

        # OOF baselines
        train_labels = pd.DataFrame({
            "cell_line_id": train_cell_arr,
            "perturbation_gene": train_gene_arr,
            "label": y_train,
        })
        gene_bl, loco_train = compute_loco_gene_means(train_labels)
        cell_bl = shrink_cell_means(train_labels)

        # Overwrite G5 in X_train with OOF values
        train_g5 = build_collaborative_features(
            pd.DataFrame({
                "cell_line_id": train_cell_arr,
                "perturbation_gene": train_gene_arr,
            }),
            gene_bl, cell_bl,
        )
        train_g5["g5_gene_baseline"] = loco_train
        for col in train_g5.columns:
            X_train[col] = train_g5[col].values

        # Teacher for cold genes
        if val_cold:
            try:
                teacher, oof_preds, gene_feats = _build_teacher(
                    train_labels, g1, g2, gene_meta, pathway_meta, expression, config,
                )
                cold_baselines = predict_gene_baselines(
                    teacher, gene_feats, list(val_cold),
                )
                gene_bl.update(cold_baselines)
                gene_bl.update(oof_preds)
            except Exception as e:
                import warnings
                warnings.warn(f"Teacher prediction failed (fold {fold_idx}): {e}")

        # Ensure all val genes/cells have baseline entries
        for g in val_genes_set:
            if g not in gene_bl:
                gene_bl[g] = 0.0
        for c in np.unique(val_cell_arr):
            if c not in cell_bl:
                cell_bl[c] = 0.0

        # Val G5 features
        val_g5 = build_collaborative_features(
            pd.DataFrame({
                "cell_line_id": val_cell_arr,
                "perturbation_gene": val_gene_arr,
            }),
            gene_bl, cell_bl,
        )
        for col in val_g5.columns:
            X_val[col] = val_g5[col].values

        # SVD
        n_train_genes = train_labels["perturbation_gene"].nunique()
        n_train_cells = train_labels["cell_line_id"].nunique()
        svd_k = min(
            config.get("prediction", {}).get("baselines", {}).get("svd_k", 50),
            n_train_genes - 1, n_train_cells - 1, 50,
        )
        svd_k = max(1, svd_k)
        U, V, svd_cell_idx, svd_gene_idx, svd_global_mean = compute_label_svd(
            train_labels, k=svd_k,
        )
        svd_dot = {}
        cell_to_svd = {c: i for i, c in enumerate(svd_cell_idx)}
        gene_to_svd = {g: i for i, g in enumerate(svd_gene_idx)}
        for _, row in pd.DataFrame({
            "cell_line_id": val_cell_arr, "perturbation_gene": val_gene_arr,
        }).iterrows():
            c, g = row["cell_line_id"], row["perturbation_gene"]
            if c in cell_to_svd and g in gene_to_svd:
                svd_dot[(c, g)] = float(np.dot(U[cell_to_svd[c]], V[gene_to_svd[g]]))

        # CF for cold genes
        cf_cold = build_gene_similarity_cf(
            train_labels, gene_meta, pathway_meta, expression, val_cold, k=20,
        )

        # Cell features for training
        train_cell_feats = build_cell_features(
            Path(config["paths"]["output_dir"]),
            list(train_labels["cell_line_id"].unique()),
        )
        train_lineage = build_lineage_onehot(cell_meta, list(train_labels["cell_line_id"].unique()))
        train_cell_feats = pd.concat([train_cell_feats, train_lineage], axis=1)

        # Train formula models
        from .formula import train_formula_models, predict_formula
        wb_models = train_formula_models(y_train, train_cell_arr, train_gene_arr,
            gene_static_features=g1,
            gene_expr_profile_features=g2,
            cell_features=train_cell_feats,
            cold_genes=val_cold if val_cold else set(),
            config=config,
            expression=expression,
        )

        # Predict
        preds_val = predict_formula(val_cell_arr, val_gene_arr,
            cold_genes=val_cold if val_cold else set(),
            models=wb_models,
            add_jitter=True,
            expression=expression,
        )

        # Compute metrics
        df_val = pd.DataFrame({
            "cell_line_id": val_cell_arr,
            "perturbation_gene": val_gene_arr,
            "prediction": preds_val,
            "truth": y_val,
        })
        fold_metrics = compute_metrics_df(df_val)
        all_metrics.append(fold_metrics)
        all_folds.append({
            "fold": fold_idx,
            "n_train_genes": len(train_genes_set),
            "n_val_genes": len(val_genes_set),
            "n_cold_genes": len(val_cold),
        })

    summary = _aggregate_folds(all_metrics, all_folds, "group_by_gene")
    return summary


# ── Protocol 2: Group-by-cell CV ────────────────────────────────────────────

def validate_group_by_cell(
    X: pd.DataFrame,
    y: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Secondary CV: hold out 20% of cells entirely per fold.

    Simulates new-cell-line robustness. Cell biases imputed from features.
    Uses interpretable formula-based models.
    """
    if config is None:
        config = {}
    data = _load_supporting_data(config)
    gene_meta = data["gene_meta"]
    pathway_meta = data["pathway_meta"]
    cell_meta = data["cell_meta"]
    expression = data["expression"]
    g1 = data["g1"]
    g2 = data["g2"]

    unique_cells = np.unique(cell_ids)
    kf = GroupKFold(n_splits=n_folds)
    all_metrics = []
    all_folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(np.arange(len(y)), groups=cell_ids)
    ):
        train_cells_set = set(cell_ids[train_idx])
        val_cells_set = set(cell_ids[val_idx])
        val_cold_cells = val_cells_set - train_cells_set

        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]
        train_cell_arr = cell_ids[train_idx]
        train_gene_arr = gene_ids[train_idx]
        val_cell_arr = cell_ids[val_idx]
        val_gene_arr = gene_ids[val_idx]

        # OOF baselines
        train_labels = pd.DataFrame({
            "cell_line_id": train_cell_arr,
            "perturbation_gene": train_gene_arr,
            "label": y_train,
        })
        cell_bl = shrink_cell_means(train_labels)
        gene_bl, loco_train = compute_loco_gene_means(train_labels)

        # Overwrite G5 in X_train
        train_g5 = build_collaborative_features(
            pd.DataFrame({
                "cell_line_id": train_cell_arr,
                "perturbation_gene": train_gene_arr,
            }),
            gene_bl, cell_bl,
        )
        train_g5["g5_gene_baseline"] = loco_train
        for col in train_g5.columns:
            X_train[col] = train_g5[col].values

        # Impute cell biases for held-out cells
        if val_cold_cells:
            from .baselines import train_cell_bias_imputer, impute_cell_biases
            train_cell_feats = build_cell_features(
                Path(config["paths"]["output_dir"]), list(train_cells_set),
            )
            imputer, _ = train_cell_bias_imputer(train_cell_feats, train_labels)
            val_cell_feats = build_cell_features(
                Path(config["paths"]["output_dir"]), list(val_cold_cells),
            )
            imputed = impute_cell_biases(imputer, val_cell_feats, list(val_cold_cells))
            cell_bl.update(imputed)

        # Ensure all entries
        for c in val_cells_set:
            if c not in cell_bl:
                cell_bl[c] = 0.0
        for g in np.unique(val_gene_arr):
            if g not in gene_bl:
                gene_bl[g] = 0.0

        # Val G5
        val_g5 = build_collaborative_features(
            pd.DataFrame({
                "cell_line_id": val_cell_arr,
                "perturbation_gene": val_gene_arr,
            }),
            gene_bl, cell_bl,
        )
        for col in val_g5.columns:
            X_val[col] = val_g5[col].values

        # SVD
        n_train_genes = train_labels["perturbation_gene"].nunique()
        n_train_cells = train_labels["cell_line_id"].nunique()
        svd_k = min(
            config.get("prediction", {}).get("baselines", {}).get("svd_k", 50),
            n_train_genes - 1, n_train_cells - 1, 50,
        )
        svd_k = max(1, svd_k)
        U, V, svd_cell_idx, svd_gene_idx, svd_global_mean = compute_label_svd(
            train_labels, k=svd_k,
        )
        svd_dot = {}
        cell_to_svd = {c: i for i, c in enumerate(svd_cell_idx)}
        gene_to_svd = {g: i for i, g in enumerate(svd_gene_idx)}
        for _, row in pd.DataFrame({
            "cell_line_id": val_cell_arr, "perturbation_gene": val_gene_arr,
        }).iterrows():
            c, g = row["cell_line_id"], row["perturbation_gene"]
            if c in cell_to_svd and g in gene_to_svd:
                svd_dot[(c, g)] = float(np.dot(U[cell_to_svd[c]], V[gene_to_svd[g]]))

        # Cold genes (genes not in training)
        val_cold_genes = set(val_gene_arr) - set(train_gene_arr)
        cf_cold = build_gene_similarity_cf(
            train_labels, gene_meta, pathway_meta, expression, val_cold_genes, k=20,
        )

        # Cell features
        train_cell_feats = build_cell_features(
            Path(config["paths"]["output_dir"]), list(train_cells_set),
        )
        train_lineage = build_lineage_onehot(cell_meta, list(train_cells_set))
        train_cell_feats = pd.concat([train_cell_feats, train_lineage], axis=1)

        # Train formula models
        from .formula import train_formula_models, predict_formula
        wb_models = train_formula_models(y_train, train_cell_arr, train_gene_arr,
            gene_static_features=g1,
            gene_expr_profile_features=g2,
            cell_features=train_cell_feats,
            cold_genes=val_cold_genes,
            config=config,
            expression=expression,
        )

        # Predict
        preds_val = predict_formula(val_cell_arr, val_gene_arr,
            cold_genes=val_cold_genes,
            models=wb_models,
            add_jitter=True,
            expression=expression,
        )

        df_val = pd.DataFrame({
            "cell_line_id": val_cell_arr,
            "perturbation_gene": val_gene_arr,
            "prediction": preds_val,
            "truth": y_val,
        })
        fold_metrics = compute_metrics_df(df_val)
        all_metrics.append(fold_metrics)
        all_folds.append({
            "fold": fold_idx,
            "n_train_cells": len(train_cells_set),
            "n_val_cells": len(val_cells_set),
        })

    summary = _aggregate_folds(all_metrics, all_folds, "group_by_cell")
    return summary


# ── Aggregation ──────────────────────────────────────────────────────────────

def _aggregate_folds(
    all_metrics: list[dict],
    all_folds: list[dict],
    protocol: str,
) -> dict[str, Any]:
    """Aggregate per-fold metrics into a summary."""
    metric_keys = ["final_score", "spearman_score", "ndcg_score",
                   "precision_score", "rmse_score", "rmse", "nrmse"]
    agg = {"protocol": protocol, "n_folds": len(all_metrics), "folds": all_folds}
    for key in metric_keys:
        vals = [m[key] for m in all_metrics]
        agg[f"{key}_mean"] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
        agg[f"{key}_per_fold"] = vals

    return agg


def run_validation(
    X: pd.DataFrame,
    y: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    config: dict[str, Any] | None = None,
    protocols: list[str] | None = None,
) -> dict[str, Any]:
    """Run all validation protocols and return combined report.

    Args:
        X: full feature DataFrame.
        y: label array.
        cell_ids, gene_ids: identifiers.
        config: configuration dict.
        protocols: list of protocols to run. Default: ["gene", "cell"].

    Returns:
        Validation report dict.
    """
    if protocols is None:
        protocols = ["gene", "cell"]

    report = {}

    if "gene" in protocols:
        print("\n=== Protocol 1: Group-by-gene CV ===")
        report["group_by_gene"] = validate_group_by_gene(
            X, y, cell_ids, gene_ids, config=config,
        )
        _print_summary(report["group_by_gene"])

    if "cell" in protocols:
        print("\n=== Protocol 2: Group-by-cell CV ===")
        report["group_by_cell"] = validate_group_by_cell(
            X, y, cell_ids, gene_ids, config=config,
        )
        _print_summary(report["group_by_cell"])

    return report


def _print_summary(summary: dict) -> None:
    """Print a one-line summary of CV results."""
    protocol = summary["protocol"]
    s = summary["final_score_mean"]
    s_std = summary["final_score_std"]
    sp = summary["spearman_score_mean"]
    ndcg = summary["ndcg_score_mean"]
    print(f"  [{protocol}] Final S = {s:.4f} ± {s_std:.4f}  "
          f"(Spearman={sp:.4f}, nDCG={ndcg:.4f})")
