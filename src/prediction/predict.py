"""End-to-end prediction pipeline for Problem 2.

Produces a submission CSV in the exact format of sample_submission_gene.csv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import build_all_features
from .baselines import (
    shrink_gene_means, shrink_cell_means,
    train_gene_baseline_teacher, predict_gene_baselines,
    train_cell_bias_imputer, impute_cell_biases,
    compute_label_svd, build_module_priors,
    compute_neighbor_essentiality, compute_neighbor_scores_for_pairs,
    impute_gene_factors, impute_cell_factors,
    build_collaborative_features,
)
from .models import train_models, predict_all
from .metrics import compute_metrics_df


def run_prediction(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full prediction pipeline: train models and generate submission.

    Args:
        config: full configuration dict.

    Returns:
        dict with keys: submission_df, models, metrics (if validation labels exist)
    """
    pred_cfg = config.get("prediction", {})
    data_dir = Path(config["paths"]["data_dir"])
    outputs_dir = Path(config["paths"]["output_dir"])
    submission_dir = Path(config["paths"]["submission_dir"])

    # ── Load labels ──
    print("Loading training labels...")
    labels = pd.read_csv(data_dir / "labels" / "gene_dependency.csv")
    print(f"  {len(labels):,} training pairs")

    # ── Load submission template ──
    print("Loading submission template...")
    submission = pd.read_csv(submission_dir / "sample_submission_gene.csv")
    print(f"  {len(submission):,} test pairs")

    # Identify cold-start genes
    train_genes = set(labels["perturbation_gene"].unique())
    test_genes = set(submission["perturbation_gene"].unique())
    cold_genes = test_genes - train_genes
    print(f"  {len(cold_genes)} cold-start genes (no training labels)")

    # ── Build features for training pairs ──
    print("\nBuilding training features...")
    X_train, meta = build_all_features(
        labels[["cell_line_id", "perturbation_gene"]], config,
    )

    # ── Compute G5 collaborative features ──
    print("\nComputing collaborative features (G5)...")
    baseline_cfg = pred_cfg.get("baselines", {})

    gene_bl = shrink_gene_means(labels)
    cell_bl = shrink_cell_means(labels)

    # SVD latent factorization
    svd_k = baseline_cfg.get("svd_k", 20)
    U, V, svd_cell_idx, svd_gene_idx, svd_global_mean = compute_label_svd(
        labels, k=svd_k,
    )
    svd_dot = {}
    for ci, cell in enumerate(svd_cell_idx):
        for gi, gene in enumerate(svd_gene_idx):
            svd_dot[(cell, gene)] = float(np.dot(U[ci], V[gi]))

    # Neighbor essentiality (k-NN)
    knn_k = baseline_cfg.get("knn_k", 20)
    indicators_path = Path(config["paths"]["output_dir"]) / "cell_line_indicators.csv"
    cell_indicators = pd.read_csv(indicators_path, index_col=0)
    print("  Computing k-NN neighbor essentiality...")
    neighbor_df = compute_neighbor_essentiality(labels, cell_indicators, k=knn_k)
    from .baselines import compute_neighbor_scores_for_pairs
    train_neighbor = compute_neighbor_scores_for_pairs(
        labels[["cell_line_id", "perturbation_gene"]], neighbor_df,
    )

    g5_train = build_collaborative_features(
        labels[["cell_line_id", "perturbation_gene"]],
        gene_bl, cell_bl,
        svd_dot_products=svd_dot,
        neighbor_scores=train_neighbor,
    )

    # Add G5 columns to X_train
    for col in g5_train.columns:
        if col not in X_train.columns:
            X_train[col] = g5_train[col].values
        else:
            X_train[col] = g5_train[col].values

    # ── Train models ──
    print("\nTraining models...")
    y_train = labels["label"].to_numpy(dtype=np.float64)
    train_cells = labels["cell_line_id"].to_numpy()
    train_genes_arr = labels["perturbation_gene"].to_numpy()

    models = train_models(
        X_train, y_train, train_cells, train_genes_arr,
        cold_genes=cold_genes, config=config,
    )

    # ── Build features for test pairs ──
    print("\nBuilding test features...")
    X_test, _ = build_all_features(submission, config)

    # Compute gene baselines for cold genes via teacher
    g1 = meta.get("gene_static_features")
    g2 = meta.get("gene_expr_profile_features")
    if g1 is None:
        from .features import build_gene_static_features, build_gene_expression_profile_features
        g1 = build_gene_static_features(
            meta["gene_meta"], meta["gene_module_map"], meta["evidence_weights"],
        )
        g2 = build_gene_expression_profile_features(meta["expression"])

    teacher, oof_preds, gene_feats = train_gene_baseline_teacher(g1, g2, labels)
    # Use teacher model to predict baselines for cold-start genes
    if cold_genes:
        cold_baselines = predict_gene_baselines(teacher, gene_feats, list(cold_genes))
        gene_bl.update(cold_baselines)
    # Backfill training genes with OOF predictions (avoids leakage)
    for gene in oof_preds:
        gene_bl[gene] = oof_preds[gene]

    # SVD dot products for test pairs (impute for cold genes / new cells)
    test_cold_genes_list = [g for g in submission["perturbation_gene"].unique()
                            if g not in set(svd_gene_idx)]
    if test_cold_genes_list:
        V_cold = impute_gene_factors(V, svd_gene_idx, g1, g2, test_cold_genes_list)
        V_extended = np.vstack([V, V_cold]) if len(V) > 0 else V_cold
        gene_idx_extended = list(svd_gene_idx) + test_cold_genes_list
    else:
        V_extended = V
        gene_idx_extended = list(svd_gene_idx)

    test_new_cells = [c for c in submission["cell_line_id"].unique()
                      if c not in set(svd_cell_idx)]
    if test_new_cells:
        from .features import build_cell_features
        test_cell_feats = build_cell_features(
            Path(config["paths"]["output_dir"]), test_new_cells,
        )
        U_new = impute_cell_factors(U, svd_cell_idx, test_cell_feats, test_new_cells)
        U_extended = np.vstack([U, U_new]) if len(U) > 0 else U_new
        cell_idx_extended = list(svd_cell_idx) + test_new_cells
    else:
        U_extended = U
        cell_idx_extended = list(svd_cell_idx)

    svd_dot_test = {}
    cell_to_row = {c: i for i, c in enumerate(cell_idx_extended)}
    gene_to_col = {g: i for i, g in enumerate(gene_idx_extended)}
    for _, row in submission.iterrows():
        c, g = row["cell_line_id"], row["perturbation_gene"]
        if c in cell_to_row and g in gene_to_col:
            svd_dot_test[(c, g)] = float(np.dot(
                U_extended[cell_to_row[c]], V_extended[gene_to_col[g]]
            ))
        else:
            svd_dot_test[(c, g)] = 0.0

    # Neighbor scores for test pairs
    test_neighbor = compute_neighbor_scores_for_pairs(
        submission[["cell_line_id", "perturbation_gene"]], neighbor_df,
    )

    test_g5 = build_collaborative_features(
        submission[["cell_line_id", "perturbation_gene"]],
        gene_bl, cell_bl,
        svd_dot_products=svd_dot_test,
        neighbor_scores=test_neighbor,
    )
    for col in test_g5.columns:
        if col not in X_test.columns:
            X_test[col] = test_g5[col].values
        else:
            X_test[col] = test_g5[col].values

    # ── Predict ──
    print("\nGenerating predictions...")
    test_preds = predict_all(
        X_test,
        submission["cell_line_id"].to_numpy(),
        submission["perturbation_gene"].to_numpy(),
        models,
        add_jitter=True,
    )

    # ── Build submission ──
    submission_df = submission.copy()
    submission_df["label"] = test_preds

    print(f"\n  Predictions: mean={test_preds.mean():.4f}, "
          f"std={test_preds.std():.4f}, "
          f"range=[{test_preds.min():.4f}, {test_preds.max():.4f}]")

    # ── Save ──
    pred_output_dir = outputs_dir / "prediction"
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = pred_output_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\n  Submission saved to: {submission_path}")

    # ── Training set metrics (for reference) ──
    print("\nComputing training metrics (for reference)...")
    train_preds = predict_all(
        X_train, train_cells, train_genes_arr, models, add_jitter=False,
    )
    train_metrics_df = pd.DataFrame({
        "cell_line_id": train_cells,
        "perturbation_gene": train_genes_arr,
        "prediction": train_preds,
        "truth": y_train,
    })
    train_metrics = compute_metrics_df(train_metrics_df)

    # ── Validation report ──
    report = {
        "config_fingerprint": str(hash(json.dumps(pred_cfg, sort_keys=True, default=str))),
        "n_train_pairs": len(labels),
        "n_test_pairs": len(submission),
        "n_cold_genes": len(cold_genes),
        "cold_genes": sorted(cold_genes),
        "train_metrics": {k: float(v) if isinstance(v, (np.floating, float)) else v
                          for k, v in train_metrics.items()},
        "model_params": {
            "blend_alpha_a": models["blend_alpha_a"],
            "blend_alpha_b": models["blend_alpha_b"],
            "blend_alpha_c": models["blend_alpha_c"],
        },
    }
    report_path = pred_output_dir / "prediction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Report saved to: {report_path}")

    return {
        "submission": submission_df,
        "models": models,
        "train_metrics": train_metrics,
        "report": report,
    }
