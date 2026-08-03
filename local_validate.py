"""Local validation script for Problem 2 — gene dependency prediction.

Evaluates model performance using the official 数据文件/calculate_metric.py script.
Supports three protocols:
  --protocol cold  : group-by-gene split (all val genes are cold-start)
  --protocol warm  : random pair holdout within each gene (matrix completion)
  --protocol real  : mimics the real test — ~15% genes cold, rest warm with
                     ~20% pair holdout (50/50 cold/warm pair split)

Usage:
    python local_validate.py                        # Realistic (default)
    python local_validate.py --protocol cold        # Cold-start only
    python local_validate.py --protocol warm        # Warm-pair only
    python local_validate.py --n-genes 200          # Quick test
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.utils import load_config


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_teacher(train_labels, gene_meta, pathway_meta, expression, config):
    """Build gene baseline teacher. Returns (teacher_model, gene_features_df)."""
    from src.prediction.features import (
        build_gene_static_features, build_gene_expression_profile_features,
    )
    from src.prediction.baselines import (
        build_pw140_membership_features, compute_coexpression_knn_features,
        build_description_keyword_features, train_gene_baseline_teacher,
    )
    from src.preprocess import build_gene_module_map, compute_evidence_weights

    gene_module_map = build_gene_module_map(gene_meta)
    evidence_weights = compute_evidence_weights(gene_meta)
    g1 = build_gene_static_features(gene_meta, gene_module_map, evidence_weights)
    g2 = build_gene_expression_profile_features(expression)
    pw140 = build_pw140_membership_features(gene_meta, pathway_meta)
    knn_k = config.get("prediction", {}).get("baselines", {}).get("knn_k", 20)
    coexpr_knn = compute_coexpression_knn_features(expression, train_labels, k=knn_k)
    desc_feats = build_description_keyword_features(gene_meta)
    teacher_extra = pd.concat([pw140, coexpr_knn, desc_feats], axis=1)
    teacher, oof_preds, gene_feats = train_gene_baseline_teacher(
        g1, g2, train_labels, extra_features=teacher_extra,
    )
    return teacher, gene_feats


def _build_features(pairs, train_labels, gene_bl, cell_bl, config):
    """Build feature table and add G5 collaborative features."""
    from src.prediction.features import build_all_features
    from src.prediction.baselines import build_collaborative_features

    X, meta = build_all_features(pairs[["cell_line_id", "perturbation_gene"]], config)
    g5 = build_collaborative_features(
        pairs[["cell_line_id", "perturbation_gene"]], gene_bl, cell_bl,
    )
    for col in g5.columns:
        X[col] = g5[col].values
    return X, meta


def _score(submission_df, answer_df, data_dir, label=""):
    """Score predictions using internal metrics."""
    from src.prediction.metrics import compute_metrics_df
    df = pd.DataFrame({
        "cell_line_id": answer_df["cell_line_id"].values,
        "perturbation_gene": answer_df["perturbation_gene"].values,
        "prediction": submission_df["label"].values,
        "truth": answer_df["label"].values,
    })
    metrics = compute_metrics_df(df)
    prefix = f"[{label}] " if label else ""
    print(f"  {prefix}S={metrics['final_score']:.4f}, "
          f"Spearman={metrics['spearman_score']:.4f}, "
          f"NDCG={metrics['ndcg_score']:.4f}, "
          f"Precision={metrics['precision_score']:.4f}, "
          f"RMSE={metrics['rmse_score']:.4f}")
    return metrics


def _run_official_scoring(submission_path, answer_path, data_dir):
    """Run the official calculate_metric.py script."""
    script_path = data_dir / "calculate_metric.py"
    if not script_path.exists():
        print(f"ERROR: Official script not found at {script_path}")
        return
    cmd = [sys.executable, str(script_path), str(submission_path), str(answer_path)]
    result = subprocess.run(cmd, capture_output=True, text=True,
                           cwd=str(Path(__file__).parent))
    if result.returncode == 0:
        print(result.stdout)
    else:
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol: Cold-start (group-by-gene split)
# ═══════════════════════════════════════════════════════════════════════════════

def run_cold(labels, config, data_dir, output_dir, args):
    """All val genes are held-out entirely (cold-start)."""
    from src.prediction.baselines import (
        compute_loco_gene_means, shrink_cell_means,
        predict_gene_baselines,
    )
    # Split by gene
    unique_genes = labels["perturbation_gene"].unique()
    rng = np.random.RandomState(args.random_state)
    shuffled = rng.permutation(unique_genes)
    n_val = int(len(shuffled) * args.test_size)
    val_genes = set(shuffled[:n_val])
    train_genes = set(shuffled[n_val:])

    train_labels = labels[labels["perturbation_gene"].isin(train_genes)].copy()
    val_labels = labels[labels["perturbation_gene"].isin(val_genes)].copy()
    cold_genes = val_genes - train_genes

    print(f"\nTrain: {len(train_labels):,} pairs, {len(train_genes)} genes")
    print(f"Val:   {len(val_labels):,} pairs, {len(val_genes)} genes")
    print(f"Cold genes: {len(cold_genes)}")

    # Baselines
    gene_bl, loco_train = compute_loco_gene_means(train_labels)
    cell_bl = shrink_cell_means(train_labels)

    # Teacher for cold gene baselines
    gene_meta = pd.read_csv(data_dir / "metadata" / "gene_metadata.csv")
    pathway_meta = pd.read_csv(data_dir / "metadata" / "pathway_metadata.csv")
    cell_meta = pd.read_csv(data_dir / "metadata" / "cell_line_metadata.csv")
    expression = pd.read_csv(data_dir / "features" / "cell_expression_zscore.csv", index_col=0)
    teacher, gene_feats = _build_teacher(
        train_labels, gene_meta, pathway_meta, expression, config,
    )
    if cold_genes:
        cold_bl = predict_gene_baselines(teacher, gene_feats, list(cold_genes))
        gene_bl.update(cold_bl)

    # Features
    print("\nBuilding features...")
    X_train, _ = _build_features(train_labels, train_labels, gene_bl, cell_bl, config)
    X_train["g5_gene_baseline"] = loco_train  # LOCO override for honest training

    # Train formula models
    print("\nTraining formula-based models...")
    y_train = train_labels["label"].to_numpy(dtype=np.float64)
    from src.prediction.features import build_cell_features, build_lineage_onehot
    train_cell_feats = build_cell_features(
        Path(config["paths"]["output_dir"]),
        list(train_labels["cell_line_id"].unique()),
    )
    train_lineage = build_lineage_onehot(cell_meta, list(train_labels["cell_line_id"].unique()))
    train_cell_feats = pd.concat([train_cell_feats, train_lineage], axis=1)

    from src.preprocess import build_gene_module_map, compute_evidence_weights
    gmm = build_gene_module_map(gene_meta)
    ew = compute_evidence_weights(gene_meta)
    from src.prediction.features import build_gene_static_features, build_gene_expression_profile_features
    g1 = build_gene_static_features(gene_meta, gmm, ew)
    g2 = build_gene_expression_profile_features(expression)

    from src.prediction.formula import train_formula_models, predict_formula
    fm_models = train_formula_models(y_train,
        train_labels["cell_line_id"].to_numpy(),
        train_labels["perturbation_gene"].to_numpy(),
        
        gene_static_features=g1,
        gene_expr_profile_features=g2,
        cell_features=train_cell_feats,
        
        
        cold_genes=cold_genes,
        config=config,
    )

    # Val features
    for cell in val_labels["cell_line_id"].unique():
        if cell not in cell_bl:
            cell_bl[cell] = 0.0
    X_val, _ = _build_features(val_labels, train_labels, gene_bl, cell_bl, config)

    # Predict
    print("\nPredicting...")
    from src.prediction.formula import predict_formula as _predict_formula
    fm_preds = _predict_formula(val_labels["cell_line_id"].to_numpy(),
        val_labels["perturbation_gene"].to_numpy(),
        
        
        cold_genes=cold_genes,
        models=fm_models,
        add_jitter=True,
    )

    # Score
    print("\n--- Results ---")
    _score(pd.DataFrame({"label": fm_preds}), val_labels, data_dir, "formula")

    # Save
    submission_df = val_labels[["cell_line_id", "perturbation_gene"]].copy()
    submission_df["label"] = fm_preds
    submission_df.to_csv(output_dir / "val_submission.csv", index=False)
    val_labels[["cell_line_id", "perturbation_gene", "label"]].to_csv(
        output_dir / "val_answer.csv", index=False,
    )
    print(f"\nSaved: {output_dir / 'val_submission.csv'}")
    print(f"Saved: {output_dir / 'val_answer.csv'}")
    _run_official_scoring(output_dir / "val_submission.csv", output_dir / "val_answer.csv", data_dir)

    return {"formula": fm_preds, "val_labels": val_labels}


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol: Warm-pair (random pair holdout within each gene)
# ═══════════════════════════════════════════════════════════════════════════════

def run_warm(labels, config, data_dir, output_dir, args):
    """Hold out random cell-gene pairs; all genes appear in both train and val."""
    from src.prediction.baselines import (
        compute_loco_gene_means, shrink_cell_means,
        predict_gene_baselines,
    )

    rng = np.random.RandomState(args.random_state)

    # Stratified holdout: within each cell, hold out test_size fraction of pairs
    train_rows, val_rows = [], []
    for cell in labels["cell_line_id"].unique():
        cell_data = labels[labels["cell_line_id"] == cell]
        n = len(cell_data)
        n_val = max(1, int(n * args.test_size))
        val_idx = rng.choice(n, n_val, replace=False)
        val_mask = np.zeros(n, dtype=bool); val_mask[val_idx] = True
        train_rows.append(cell_data[~val_mask])
        val_rows.append(cell_data[val_mask])

    train_labels = pd.concat(train_rows).reset_index(drop=True)
    val_labels = pd.concat(val_rows).reset_index(drop=True)
    cold_genes = set(val_labels["perturbation_gene"].unique()) - set(train_labels["perturbation_gene"].unique())

    print(f"\nTrain: {len(train_labels):,} pairs, {train_labels['perturbation_gene'].nunique()} genes")
    print(f"Val:   {len(val_labels):,} pairs, {val_labels['perturbation_gene'].nunique()} genes")
    print(f"Cold genes (no train pairs): {len(cold_genes)}")

    # Baselines
    gene_bl, loco_train = compute_loco_gene_means(train_labels)
    cell_bl = shrink_cell_means(train_labels)

    # Teacher for any cold genes
    gene_meta = pd.read_csv(data_dir / "metadata" / "gene_metadata.csv")
    pathway_meta = pd.read_csv(data_dir / "metadata" / "pathway_metadata.csv")
    cell_meta = pd.read_csv(data_dir / "metadata" / "cell_line_metadata.csv")
    expression = pd.read_csv(data_dir / "features" / "cell_expression_zscore.csv", index_col=0)
    if cold_genes:
        teacher, gene_feats = _build_teacher(
            train_labels, gene_meta, pathway_meta, expression, config,
        )
        cold_bl = predict_gene_baselines(teacher, gene_feats, list(cold_genes))
        gene_bl.update(cold_bl)

    # Features
    print("\nBuilding features...")
    X_train, _ = _build_features(train_labels, train_labels, gene_bl, cell_bl, config)
    X_train["g5_gene_baseline"] = loco_train

    # Train formula models
    print("\nTraining formula-based models...")
    y_train = train_labels["label"].to_numpy(dtype=np.float64)
    from src.prediction.features import build_cell_features as _bcf, build_lineage_onehot as _blo
    train_cell_feats = _bcf(
        Path(config["paths"]["output_dir"]),
        list(train_labels["cell_line_id"].unique()),
    )
    train_lineage = _blo(cell_meta, list(train_labels["cell_line_id"].unique()))
    train_cell_feats = pd.concat([train_cell_feats, train_lineage], axis=1)

    from src.preprocess import build_gene_module_map as _bgmm, compute_evidence_weights as _cew
    gmm = _bgmm(gene_meta)
    ew = _cew(gene_meta)
    from src.prediction.features import build_gene_static_features as _bgsf, build_gene_expression_profile_features as _bgepf
    g1 = _bgsf(gene_meta, gmm, ew)
    g2 = _bgepf(expression)

    from src.prediction.formula import train_formula_models as _train_fm
    fm_models = _train_fm(
        y_train,
        train_labels["cell_line_id"].to_numpy(),
        train_labels["perturbation_gene"].to_numpy(),
        
        gene_static_features=g1,
        gene_expr_profile_features=g2,
        cell_features=train_cell_feats,
        
        
        cold_genes=cold_genes,
        config=config,
    )

    # Val
    for cell in val_labels["cell_line_id"].unique():
        if cell not in cell_bl:
            cell_bl[cell] = 0.0
    X_val, _ = _build_features(val_labels, train_labels, gene_bl, cell_bl, config)

    # Predict
    print("\nPredicting...")
    from src.prediction.formula import predict_formula as _predict_formula
    fm_preds = _predict_formula(val_labels["cell_line_id"].to_numpy(),
        val_labels["perturbation_gene"].to_numpy(),
        
        
        cold_genes=cold_genes,
        models=fm_models,
        add_jitter=True,
    )

    # Score per regime
    print("\n--- Results ---")
    _score(pd.DataFrame({"label": fm_preds}), val_labels, data_dir, "formula-ALL")

    warm_mask = ~val_labels["perturbation_gene"].isin(cold_genes)
    cold_mask = val_labels["perturbation_gene"].isin(cold_genes)
    n_cold_pairs = cold_mask.sum()
    print(f"\n  Cold pairs: {n_cold_pairs} ({100*n_cold_pairs/len(val_labels):.1f}%)")

    if warm_mask.any():
        print("\n--- Warm-pair regime ---")
        _score(pd.DataFrame({"label": fm_preds[warm_mask.values]}),
               val_labels[warm_mask.values].reset_index(drop=True), data_dir, "FM-WARM")
    if cold_mask.any():
        print("\n--- Cold-pair regime ---")
        _score(pd.DataFrame({"label": fm_preds[cold_mask.values]}),
               val_labels[cold_mask.values].reset_index(drop=True), data_dir, "FM-COLD")

    # Save — use formula-based predictions for official scoring
    submission_df = val_labels[["cell_line_id", "perturbation_gene"]].copy()
    submission_df["label"] = fm_preds
    submission_df.to_csv(output_dir / "val_submission.csv", index=False)
    val_labels[["cell_line_id", "perturbation_gene", "label"]].to_csv(
        output_dir / "val_answer.csv", index=False,
    )
    print(f"\nSaved: {output_dir / 'val_submission.csv'}")
    _run_official_scoring(output_dir / "val_submission.csv", output_dir / "val_answer.csv", data_dir)

    return {"formula": fm_preds, "val_labels": val_labels}


# ═══════════════════════════════════════════════════════════════════════════════
# Protocol: Realistic (mimics the real test — ~50/50 cold/warm pair split)
# ═══════════════════════════════════════════════════════════════════════════════

def run_real(labels, config, data_dir, output_dir, args):
    """Mimics the real test: ~15% genes cold, rest warm with ~20% pair holdout."""
    from src.prediction.baselines import (
        compute_loco_gene_means, shrink_cell_means,
        predict_gene_baselines,
    )

    rng = np.random.RandomState(args.random_state)
    unique_genes = labels["perturbation_gene"].unique()
    shuffled = rng.permutation(unique_genes)

    # Hold out ~15% genes as cold (matching real test: 165/1098 ≈ 15%)
    n_cold_genes = max(1, int(len(shuffled) * 0.15))
    cold_genes = set(shuffled[:n_cold_genes])
    warm_genes = set(shuffled[n_cold_genes:])

    # For warm genes: hold out random pairs per cell
    # For cold genes: all pairs go to val
    train_rows, val_rows = [], []
    for cell in labels["cell_line_id"].unique():
        cell_data = labels[labels["cell_line_id"] == cell]
        cell_cold = cell_data[cell_data["perturbation_gene"].isin(cold_genes)]
        cell_warm = cell_data[cell_data["perturbation_gene"].isin(warm_genes)]

        val_rows.append(cell_cold)  # All cold pairs → val

        if len(cell_warm) > 0:
            n_val_warm = max(1, int(len(cell_warm) * args.test_size))
            val_idx = rng.choice(len(cell_warm), n_val_warm, replace=False)
            val_mask = np.zeros(len(cell_warm), dtype=bool); val_mask[val_idx] = True
            train_rows.append(cell_warm[~val_mask])
            val_rows.append(cell_warm[val_mask])
        else:
            train_rows.append(cell_warm)

    train_labels = pd.concat(train_rows).reset_index(drop=True)
    val_labels = pd.concat(val_rows).reset_index(drop=True)
    val_cold_genes = set(val_labels["perturbation_gene"].unique()) - set(train_labels["perturbation_gene"].unique())

    n_cold_pairs = len(val_labels[val_labels["perturbation_gene"].isin(cold_genes)])
    n_warm_pairs = len(val_labels[~val_labels["perturbation_gene"].isin(cold_genes)])

    print(f"\nTrain: {len(train_labels):,} pairs, {train_labels['perturbation_gene'].nunique()} genes")
    print(f"Val:   {len(val_labels):,} pairs, {val_labels['perturbation_gene'].nunique()} genes")
    print(f"  Cold pairs: {n_cold_pairs:,} ({100*n_cold_pairs/len(val_labels):.1f}%)")
    print(f"  Warm pairs: {n_warm_pairs:,} ({100*n_warm_pairs/len(val_labels):.1f}%)")
    print(f"  Cold genes: {len(val_cold_genes)}")

    # Baselines
    gene_bl, loco_train = compute_loco_gene_means(train_labels)
    cell_bl = shrink_cell_means(train_labels)

    # Teacher
    gene_meta = pd.read_csv(data_dir / "metadata" / "gene_metadata.csv")
    pathway_meta = pd.read_csv(data_dir / "metadata" / "pathway_metadata.csv")
    cell_meta = pd.read_csv(data_dir / "metadata" / "cell_line_metadata.csv")
    expression = pd.read_csv(data_dir / "features" / "cell_expression_zscore.csv", index_col=0)
    if val_cold_genes:
        teacher, gene_feats = _build_teacher(
            train_labels, gene_meta, pathway_meta, expression, config,
        )
        cold_bl = predict_gene_baselines(teacher, gene_feats, list(val_cold_genes))
        gene_bl.update(cold_bl)

    # Features
    print("\nBuilding features...")
    X_train, _ = _build_features(train_labels, train_labels, gene_bl, cell_bl, config)
    X_train["g5_gene_baseline"] = loco_train

    # Train formula models
    print("\nTraining formula-based models...")
    y_train = train_labels["label"].to_numpy(dtype=np.float64)
    from src.prediction.features import build_cell_features as _bcf, build_lineage_onehot as _blo
    train_cell_feats = _bcf(
        Path(config["paths"]["output_dir"]),
        list(train_labels["cell_line_id"].unique()),
    )
    train_lineage = _blo(cell_meta, list(train_labels["cell_line_id"].unique()))
    train_cell_feats = pd.concat([train_cell_feats, train_lineage], axis=1)

    from src.preprocess import build_gene_module_map as _bgmm, compute_evidence_weights as _cew
    gmm = _bgmm(gene_meta)
    ew = _cew(gene_meta)
    from src.prediction.features import build_gene_static_features as _bgsf, build_gene_expression_profile_features as _bgepf
    g1 = _bgsf(gene_meta, gmm, ew)
    g2 = _bgepf(expression)

    from src.prediction.formula import train_formula_models as _train_fm
    fm_models = _train_fm(
        y_train,
        train_labels["cell_line_id"].to_numpy(),
        train_labels["perturbation_gene"].to_numpy(),
        
        gene_static_features=g1,
        gene_expr_profile_features=g2,
        cell_features=train_cell_feats,
        
        
        cold_genes=val_cold_genes,
        config=config,
    )

    # Val
    for cell in val_labels["cell_line_id"].unique():
        if cell not in cell_bl:
            cell_bl[cell] = 0.0
    X_val, _ = _build_features(val_labels, train_labels, gene_bl, cell_bl, config)

    # Predict
    print("\nPredicting...")
    from src.prediction.formula import predict_formula as _predict_formula
    fm_preds = _predict_formula(val_labels["cell_line_id"].to_numpy(),
        val_labels["perturbation_gene"].to_numpy(),
        
        
        cold_genes=cold_genes,
        models=fm_models,
        add_jitter=True,
    )

    # Score — overall and per-regime
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print("\n--- Overall ---")
    _score(pd.DataFrame({"label": fm_preds}), val_labels, data_dir, "formula")

    warm_mask = ~val_labels["perturbation_gene"].isin(cold_genes)
    cold_mask = val_labels["perturbation_gene"].isin(cold_genes)

    if warm_mask.any():
        print("\n--- Warm regime (matrix completion) ---")
        _score(pd.DataFrame({"label": fm_preds[warm_mask.values]}),
               val_labels[warm_mask.values].reset_index(drop=True), data_dir, "FM-WARM")
    if cold_mask.any():
        print("\n--- Cold regime (cold-start) ---")
        _score(pd.DataFrame({"label": fm_preds[cold_mask.values]}),
               val_labels[cold_mask.values].reset_index(drop=True), data_dir, "FM-COLD")

    # Save and run official scoring
    submission_df = val_labels[["cell_line_id", "perturbation_gene"]].copy()
    submission_df["label"] = fm_preds
    submission_df.to_csv(output_dir / "val_submission.csv", index=False)
    val_labels[["cell_line_id", "perturbation_gene", "label"]].to_csv(
        output_dir / "val_answer.csv", index=False,
    )
    print(f"\nSaved: {output_dir / 'val_submission.csv'}")

    print("\n" + "=" * 60)
    print("OFFICIAL SCORING SCRIPT")
    print("=" * 60)
    _run_official_scoring(output_dir / "val_submission.csv", output_dir / "val_answer.csv", data_dir)

    return {"formula": fm_preds, "val_labels": val_labels}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Local validation for Problem 2")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction to hold out (default: 0.2)")
    parser.add_argument("--n-genes", type=int, default=0,
                        help="Limit to first N genes (0 = all)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--protocol", type=str, default="real",
                        choices=["cold", "warm", "real"],
                        help="Validation protocol (default: real)")
    args = parser.parse_args()

    config = load_config()
    data_dir = Path(config["paths"]["data_dir"])
    output_dir = Path(config["paths"]["output_dir"]) / "prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"Local Validation — protocol={args.protocol}, test_size={args.test_size}")
    print("=" * 60)

    labels = pd.read_csv(data_dir / "labels" / "gene_dependency.csv")
    if args.n_genes > 0:
        all_genes = sorted(labels["perturbation_gene"].unique())[:args.n_genes]
        labels = labels[labels["perturbation_gene"].isin(all_genes)]

    print(f"\nTotal: {len(labels):,} pairs, "
          f"{labels['cell_line_id'].nunique()} cells, "
          f"{labels['perturbation_gene'].nunique()} genes")

    if args.protocol == "cold":
        run_cold(labels, config, data_dir, output_dir, args)
    elif args.protocol == "warm":
        run_warm(labels, config, data_dir, output_dir, args)
    else:
        run_real(labels, config, data_dir, output_dir, args)


if __name__ == "__main__":
    main()
