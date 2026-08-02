"""Lightweight submission generator using only the additive model.

Skips expensive G4 pair feature building — only computes gene baselines,
cell biases, SVD factors, and gene-similarity CF for the additive model.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def generate_submission(config: dict) -> pd.DataFrame:
    """Generate submission.csv using the additive + CF + SVD model."""
    pred_cfg = config.get("prediction", {})
    data_dir = Path(config["paths"]["data_dir"])
    outputs_dir = Path(config["paths"]["output_dir"])

    # ── Load data ──
    print("Loading data...")
    labels = pd.read_csv(data_dir / "labels" / "gene_dependency.csv")
    submission = pd.read_csv(data_dir / "submission" / "sample_submission_gene.csv")
    gene_meta = pd.read_csv(data_dir / "metadata" / "gene_metadata.csv")
    pathway_meta = pd.read_csv(data_dir / "metadata" / "pathway_metadata.csv")
    expression = pd.read_csv(
        data_dir / "features" / "cell_expression_zscore.csv", index_col=0,
    )
    indicators = pd.read_csv(outputs_dir / "cell_line_indicators.csv", index_col=0)

    train_genes = set(labels["perturbation_gene"].unique())
    cold_genes = set(submission["perturbation_gene"].unique()) - train_genes
    print(f"  {len(labels):,} training pairs, {len(cold_genes)} cold genes")

    # ── Gene baselines ──
    from src.prediction.baselines import (
        compute_loco_gene_means, shrink_cell_means,
        train_gene_baseline_teacher, predict_gene_baselines,
        build_pw140_membership_features, compute_coexpression_knn_features,
        build_description_keyword_features,
        compute_label_svd, impute_gene_factors, impute_cell_factors,
        train_cell_bias_imputer, impute_cell_biases,
        build_gene_similarity_cf, predict_additive,
    )
    from src.prediction.features import (
        build_gene_static_features, build_gene_expression_profile_features,
        build_cell_features,
    )
    from src.preprocess import build_gene_module_map, compute_evidence_weights

    print("\nComputing gene baselines...")
    gene_bl, _ = compute_loco_gene_means(labels)
    cell_bl = shrink_cell_means(labels)
    print(f"  Warm genes: {len(gene_bl)}, Cells: {len(cell_bl)}")

    # ── Cell bias imputation for new test cells ──
    train_cell_feats = build_cell_features(
        outputs_dir, list(labels["cell_line_id"].unique()),
    )
    cell_bias_imputer, cell_bl = train_cell_bias_imputer(train_cell_feats, labels)
    test_new_cells = [
        c for c in submission["cell_line_id"].unique()
        if c not in set(labels["cell_line_id"].unique())
    ]
    if test_new_cells:
        test_cell_feats = build_cell_features(outputs_dir, test_new_cells)
        new_cell_biases = impute_cell_biases(
            cell_bias_imputer, test_cell_feats, test_new_cells,
        )
        cell_bl.update(new_cell_biases)
        print(f"  Imputed cell biases for {len(test_new_cells)} new cells")

    # ── Teacher for cold genes ──
    print("\nTraining gene baseline teacher...")
    gmm = build_gene_module_map(gene_meta)
    ew = compute_evidence_weights(gene_meta)
    g1 = build_gene_static_features(gene_meta, gmm, ew)
    g2 = build_gene_expression_profile_features(expression)
    pw140 = build_pw140_membership_features(gene_meta, pathway_meta)
    knn_k = pred_cfg.get("baselines", {}).get("knn_k", 20)
    coexpr_knn = compute_coexpression_knn_features(expression, labels, k=knn_k)
    desc_feats = build_description_keyword_features(gene_meta)
    teacher_extra = pd.concat([pw140, coexpr_knn, desc_feats], axis=1)

    teacher, _, gene_feats = train_gene_baseline_teacher(
        g1, g2, labels, extra_features=teacher_extra,
    )
    if cold_genes:
        cold_bl = predict_gene_baselines(teacher, gene_feats, list(cold_genes))
        gene_bl.update(cold_bl)
        print(f"  Teacher predictions for {len(cold_genes)} cold genes")

    # ── SVD factorization ──
    print("\nComputing SVD factorization...")
    svd_k = pred_cfg.get("baselines", {}).get("svd_k", 20)
    U, V, svd_cell_idx, svd_gene_idx, svd_global_mean = compute_label_svd(
        labels, k=svd_k,
    )
    # Impute cold gene factors
    if cold_genes:
        cold_list = list(cold_genes & set(g1.index))
        V_cold = impute_gene_factors(V, svd_gene_idx, g1, g2, cold_list)
        V_ext = np.vstack([V, V_cold]) if len(V) > 0 else V_cold
        gene_idx_ext = list(svd_gene_idx) + cold_list
    else:
        V_ext = V
        gene_idx_ext = list(svd_gene_idx)
    # Impute new cell factors (need train+test features for training the imputer)
    if test_new_cells:
        all_cells = list(labels["cell_line_id"].unique()) + test_new_cells
        all_cell_feats = build_cell_features(outputs_dir, all_cells)
        tc_list = [c for c in test_new_cells if c in all_cell_feats.index]
        if tc_list:
            U_new = impute_cell_factors(U, svd_cell_idx, all_cell_feats, tc_list)
            U_ext = np.vstack([U, U_new]) if len(U) > 0 else U_new
            cell_idx_ext = list(svd_cell_idx) + tc_list
        else:
            U_ext = U
            cell_idx_ext = list(svd_cell_idx)
    else:
        U_ext = U
        cell_idx_ext = list(svd_cell_idx)

    # Build SVD lookup
    gene_to_col = {g: i for i, g in enumerate(gene_idx_ext)}
    cell_to_row = {c: i for i, c in enumerate(cell_idx_ext)}
    svd_dot = {}
    for _, row in submission.iterrows():
        c, g = row["cell_line_id"], row["perturbation_gene"]
        if c in cell_to_row and g in gene_to_col:
            svd_dot[(c, g)] = float(np.dot(
                U_ext[cell_to_row[c]], V_ext[gene_to_col[g]]
            ))
        else:
            svd_dot[(c, g)] = 0.0
    print(f"  SVD factors: {U_ext.shape[1]} dims")

    # ── Gene-similarity CF for cold genes ──
    print("\nComputing gene-similarity CF...")
    cf_cold = build_gene_similarity_cf(
        labels, gene_meta, pathway_meta, expression, cold_genes, k=20,
    )
    print(f"  CF predictions for {len(cf_cold):,} pairs")

    # ── Predict ──
    print("\nGenerating predictions...")
    test_preds = predict_additive(
        submission[["cell_line_id", "perturbation_gene"]],
        gene_bl, cell_bl,
        svd_dot_products=svd_dot, svd_weight=0.5,
        cf_predictions=cf_cold, cf_weight=0.2,
    )
    print(f"  Predictions: mean={test_preds.mean():.4f}, "
          f"std={test_preds.std():.4f}, "
          f"range=[{test_preds.min():.4f}, {test_preds.max():.4f}]")

    # ── Save ──
    submission_df = submission.copy()
    submission_df["label"] = test_preds
    pred_output_dir = outputs_dir / "prediction"
    pred_output_dir.mkdir(parents=True, exist_ok=True)
    submission_path = pred_output_dir / "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")

    return submission_df


if __name__ == "__main__":
    import sys; sys.path.insert(0, str(Path(__file__).resolve().parent))
    from src.utils import load_config
    generate_submission(load_config())
