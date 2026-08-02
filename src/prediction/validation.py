"""Cross-validation framework for gene dependency prediction.

Three protocols:
  1. Group-by-gene 5-fold — primary, simulates cold-start
  2. Group-by-cell 5-fold — secondary, simulates new cell lines
  3. Random-pair split — auxiliary, fast hyperparameter tuning

All protocols enforce strict leakage control:
  - G5 teacher features computed out-of-fold
  - Neighbor essentiality excludes held-out data
  - Metrics computed only on held-out pairs per cell
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from .metrics import compute_metrics_df
from .models import (
    train_models, predict_all, prepare_features_a, prepare_features_b,
    blend_ranks, calibrate_rmse, apply_calibration,
)
from .baselines import (
    compute_loco_gene_means, shrink_gene_means, shrink_cell_means,
    train_gene_baseline_teacher, predict_gene_baselines,
    train_cell_bias_imputer, impute_cell_biases,
    build_collaborative_features,
)
from .features import (
    build_gene_static_features, build_gene_expression_profile_features,
)


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

    Simulates cold-start generalization. Gene baselines for held-out genes
    come only from training-fold data (teacher model).
    """
    unique_genes = np.unique(gene_ids)
    kf = GroupKFold(n_splits=n_folds)
    all_metrics = []
    all_folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(np.arange(len(y)), groups=gene_ids)
    ):
        train_genes = set(gene_ids[train_idx])
        val_genes = set(gene_ids[val_idx])

        # Train models on train fold
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Recompute G5 features from fold-train labels only (no leakage)
        train_labels = pd.DataFrame({
            "cell_line_id": cell_ids[train_idx],
            "perturbation_gene": gene_ids[train_idx],
            "label": y_train,
        })
        gene_bl, loco_train = compute_loco_gene_means(train_labels)
        cell_bl = shrink_cell_means(train_labels)

        # Overwrite G5 in X_train with fold-train values (LOCO for gene baseline)
        train_g5 = build_collaborative_features(
            pd.DataFrame({
                "cell_line_id": cell_ids[train_idx],
                "perturbation_gene": gene_ids[train_idx],
            }),
            gene_bl, cell_bl,
        )
        # Override with LOCO values for honest training
        train_g5["g5_gene_baseline"] = loco_train
        for col in train_g5.columns:
            X_train[col] = train_g5[col].values

        # For val genes (simulating cold-start): teacher prediction
        val_cold = val_genes - train_genes
        if val_cold and config is not None:
            try:
                # Build teacher from fold-train data using metadata
                from .features import (
                    build_gene_static_features, build_gene_expression_profile_features,
                )
                from .baselines import (
                    build_pw140_membership_features, compute_coexpression_knn_features,
                    build_description_keyword_features,
                )
                metadata_dir = Path(config["paths"]["metadata_dir"])
                gene_meta = pd.read_csv(metadata_dir / "gene_metadata.csv")
                path_meta = pd.read_csv(metadata_dir / "pathway_metadata.csv")
                features_dir = Path(config["paths"]["features_dir"])
                expression = pd.read_csv(features_dir / "cell_expression_zscore.csv", index_col=0)
                from ..preprocess import build_gene_module_map, compute_evidence_weights
                gmm = build_gene_module_map(gene_meta, path_meta)
                ew = compute_evidence_weights(gene_meta, config)
                g1 = build_gene_static_features(gene_meta, gmm, ew)
                g2 = build_gene_expression_profile_features(expression)
                # Build extra gene-level features for teacher (OOF: from fold-train only)
                pw140 = build_pw140_membership_features(gene_meta, path_meta)
                knn_k = config.get("prediction", {}).get("baselines", {}).get("knn_k", 20)
                coexpr_knn = compute_coexpression_knn_features(
                    expression, train_labels, k=knn_k,
                )
                desc_feats = build_description_keyword_features(gene_meta)
                teacher_extra = pd.concat([pw140, coexpr_knn, desc_feats], axis=1)
                teacher, oof_preds, gene_feats = train_gene_baseline_teacher(
                    g1, g2, train_labels, n_folds=min(5, len(train_genes)),
                    extra_features=teacher_extra,
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
        for g in val_genes:
            if g not in gene_bl:
                gene_bl[g] = 0.0
        for c in np.unique(cell_ids[val_idx]):
            if c not in cell_bl:
                cell_bl[c] = 0.0

        X_val_g5 = build_collaborative_features(
            pd.DataFrame({
                "cell_line_id": cell_ids[val_idx],
                "perturbation_gene": gene_ids[val_idx],
            }),
            gene_bl, cell_bl,
        )
        for col in X_val_g5.columns:
            X_val[col] = X_val_g5[col].values

        # Train on fold
        fold_models = train_models(
            X_train, y_train, cell_ids[train_idx], gene_ids[train_idx],
            cold_genes=val_cold if val_cold else set(),
            config=config,
        )

        # Predict
        preds_val = predict_all(
            X_val, cell_ids[val_idx], gene_ids[val_idx], fold_models,
        )

        # Compute metrics
        df_val = pd.DataFrame({
            "cell_line_id": cell_ids[val_idx],
            "perturbation_gene": gene_ids[val_idx],
            "prediction": preds_val,
            "truth": y_val,
        })
        fold_metrics = compute_metrics_df(df_val)
        all_metrics.append(fold_metrics)
        all_folds.append({
            "fold": fold_idx,
            "n_train_genes": len(train_genes),
            "n_val_genes": len(val_genes),
            "n_cold_genes": len(val_genes - train_genes),
        })

    # Aggregate
    summary = _aggregate_folds(all_metrics, all_folds, "group_by_gene")
    return summary


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

    Simulates new-cell-line robustness. Cell biases for held-out cells
    are imputed from cell features.
    """
    unique_cells = np.unique(cell_ids)
    kf = GroupKFold(n_splits=n_folds)
    all_metrics = []
    all_folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        kf.split(np.arange(len(y)), groups=cell_ids)
    ):
        train_cells = set(cell_ids[train_idx])
        val_cells = set(cell_ids[val_idx])

        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Out-of-fold cell biases and gene baselines
        train_labels = pd.DataFrame({
            "cell_line_id": cell_ids[train_idx],
            "perturbation_gene": gene_ids[train_idx],
            "label": y_train,
        })
        cell_bl = shrink_cell_means(train_labels)
        gene_bl = shrink_gene_means(train_labels)

        # Overwrite G5 in X_train with fold-train values (no leakage)
        train_g5 = build_collaborative_features(
            pd.DataFrame({
                "cell_line_id": cell_ids[train_idx],
                "perturbation_gene": gene_ids[train_idx],
            }),
            gene_bl, cell_bl,
        )
        for col in train_g5.columns:
            X_train[col] = train_g5[col].values

        # Impute cell biases for held-out cells
        val_cold_cells = val_cells - train_cells
        if val_cold_cells and config is not None:
            try:
                from .features import build_cell_features
                from .baselines import train_cell_bias_imputer, impute_cell_biases
                # Build cell features for TRAINING cells (to fit the imputer)
                train_cell_feats = build_cell_features(
                    Path(config["paths"]["output_dir"]),
                    list(train_cells),
                )
                imputer, _ = train_cell_bias_imputer(train_cell_feats, train_labels)
                # Build cell features for VAL cells (to impute)
                val_cell_feats = build_cell_features(
                    Path(config["paths"]["output_dir"]),
                    list(val_cold_cells),
                )
                imputed = impute_cell_biases(imputer, val_cell_feats, list(val_cold_cells))
                cell_bl.update(imputed)
            except Exception as e:
                import warnings
                warnings.warn(f"Cell bias imputation failed: {e}")

        # Ensure all val cells have entries
        for c in val_cells:
            if c not in cell_bl:
                cell_bl[c] = 0.0
        for g in np.unique(gene_ids[val_idx]):
            if g not in gene_bl:
                gene_bl[g] = 0.0

        X_val_g5 = build_collaborative_features(
            pd.DataFrame({
                "cell_line_id": cell_ids[val_idx],
                "perturbation_gene": gene_ids[val_idx],
            }),
            gene_bl, cell_bl,
        )
        for col in X_val_g5.columns:
            X_val[col] = X_val_g5[col].values

        fold_models = train_models(
            X_train, y_train, cell_ids[train_idx], gene_ids[train_idx],
            config=config,
        )

        preds_val = predict_all(
            X_val, cell_ids[val_idx], gene_ids[val_idx], fold_models,
        )

        df_val = pd.DataFrame({
            "cell_line_id": cell_ids[val_idx],
            "perturbation_gene": gene_ids[val_idx],
            "prediction": preds_val,
            "truth": y_val,
        })
        fold_metrics = compute_metrics_df(df_val)
        all_metrics.append(fold_metrics)
        all_folds.append({
            "fold": fold_idx,
            "n_train_cells": len(train_cells),
            "n_val_cells": len(val_cells),
        })

    summary = _aggregate_folds(all_metrics, all_folds, "group_by_cell")
    return summary


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
