"""Local validation script as described in 数据文件/README.md.

Splits gene_dependency.csv by gene (80/20 group-by-gene), trains models
on the training split, predicts on the validation split, and scores
using the official calculate_metric.py script.

Usage:
    python local_validate.py                    # Default: 80/20 split
    python local_validate.py --test-size 0.1    # 90/10 split
    python local_validate.py --n-genes 100      # Quick test with only 100 genes
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Local validation for Problem 2")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of genes to hold out (default: 0.2)")
    parser.add_argument("--n-genes", type=int, default=0,
                        help="Limit to first N genes for quick testing (0 = all)")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    config = load_config()

    data_dir = Path(config["paths"]["data_dir"])
    output_dir = Path(config["paths"]["output_dir"]) / "prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──
    print("=" * 60)
    print("Local Validation")
    print("=" * 60)

    labels = pd.read_csv(data_dir / "labels" / "gene_dependency.csv")

    if args.n_genes > 0:
        all_genes = sorted(labels["perturbation_gene"].unique())[:args.n_genes]
        labels = labels[labels["perturbation_gene"].isin(all_genes)]

    print(f"\nTotal: {len(labels):,} pairs, "
          f"{labels['cell_line_id'].nunique()} cells, "
          f"{labels['perturbation_gene'].nunique()} genes")

    # ── Split by gene ──
    unique_genes = labels["perturbation_gene"].unique()
    rng = np.random.RandomState(args.random_state)
    shuffled = rng.permutation(unique_genes)
    n_val = int(len(shuffled) * args.test_size)
    val_genes = set(shuffled[:n_val])
    train_genes = set(shuffled[n_val:])

    train_labels = labels[labels["perturbation_gene"].isin(train_genes)].copy()
    val_labels = labels[labels["perturbation_gene"].isin(val_genes)].copy()

    print(f"Train: {len(train_labels):,} pairs, {len(train_genes)} genes "
          f"({len(train_labels['cell_line_id'].unique())} cells)")
    print(f"Val:   {len(val_labels):,} pairs, {len(val_genes)} genes "
          f"({len(val_labels['cell_line_id'].unique())} cells)")
    print(f"Cold-start genes (in val but not train): {len(val_genes - train_genes)}")

    # ── Build features ──
    print("\nBuilding training features...")
    from src.prediction.features import build_all_features
    from src.prediction.baselines import (
        shrink_gene_means, shrink_cell_means,
        build_collaborative_features,
        train_gene_baseline_teacher, predict_gene_baselines,
    )

    X_train, meta = build_all_features(
        train_labels[["cell_line_id", "perturbation_gene"]], config,
    )
    gene_bl = shrink_gene_means(train_labels)
    cell_bl = shrink_cell_means(train_labels)
    g5_train = build_collaborative_features(
        train_labels[["cell_line_id", "perturbation_gene"]], gene_bl, cell_bl,
    )
    for col in g5_train.columns:
        X_train[col] = g5_train[col].values

    # ── Train models ──
    print("\nTraining models...")
    from src.prediction.models import train_models

    y_train = train_labels["label"].to_numpy(dtype=np.float64)
    train_cells = train_labels["cell_line_id"].to_numpy()
    train_genes_arr = train_labels["perturbation_gene"].to_numpy()

    cold_genes_in_val = val_genes - train_genes
    models = train_models(
        X_train, y_train, train_cells, train_genes_arr,
        cold_genes=cold_genes_in_val, config=config,
    )

    # ── Build val features ──
    print("\nBuilding validation features...")
    X_val, _ = build_all_features(
        val_labels[["cell_line_id", "perturbation_gene"]], config,
    )

    # Predict gene baselines for cold genes
    g1 = meta.get("gene_static_features")
    g2 = meta.get("gene_expr_profile_features")
    if g1 is None:
        from src.prediction.features import (
            build_gene_static_features, build_gene_expression_profile_features,
        )
        g1 = build_gene_static_features(
            meta["gene_meta"], meta["gene_module_map"], meta["evidence_weights"],
        )
        g2 = build_gene_expression_profile_features(meta["expression"])

    teacher, oof_preds, gene_feats = train_gene_baseline_teacher(g1, g2, train_labels)
    # Backfill training genes with OOF predictions
    for gene, val in oof_preds.items():
        gene_bl[gene] = val
    # Predict baselines for cold-start genes via teacher model
    if cold_genes_in_val:
        cold_baselines = predict_gene_baselines(teacher, gene_feats, list(cold_genes_in_val))
        gene_bl.update(cold_baselines)
    for cell in val_labels["cell_line_id"].unique():
        if cell not in cell_bl:
            cell_bl[cell] = 0.0

    g5_val = build_collaborative_features(
        val_labels[["cell_line_id", "perturbation_gene"]], gene_bl, cell_bl,
    )
    for col in g5_val.columns:
        X_val[col] = g5_val[col].values

    # ── Predict ──
    print("\nGenerating predictions...")
    from src.prediction.models import predict_all

    val_preds = predict_all(
        X_val,
        val_labels["cell_line_id"].to_numpy(),
        val_labels["perturbation_gene"].to_numpy(),
        models,
        add_jitter=True,
    )

    # ── Save submission and answer files ──
    submission_path = output_dir / "val_submission.csv"
    answer_path = output_dir / "val_answer.csv"

    submission_df = val_labels[["cell_line_id", "perturbation_gene"]].copy()
    submission_df["label"] = val_preds
    submission_df.to_csv(submission_path, index=False)

    answer_df = val_labels[["cell_line_id", "perturbation_gene", "label"]].copy()
    answer_df.to_csv(answer_path, index=False)

    print(f"\nSaved: {submission_path}")
    print(f"Saved: {answer_path}")

    # ── Run official scoring script ──
    print("\n" + "=" * 60)
    print("Running official scoring script...")
    print("=" * 60)

    script_path = data_dir / "calculate_metric.py"
    if not script_path.exists():
        print(f"ERROR: Official script not found at {script_path}")
        sys.exit(1)

    # Copy the script's approach — run via subprocess
    cmd = [
        sys.executable, str(script_path),
        str(submission_path), str(answer_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).parent))

    if result.returncode != 0:
        print("STDERR:", result.stderr)
        print("STDOUT:", result.stdout)
        sys.exit(result.returncode)

    print(result.stdout)

    # Also compute with our internal metrics for comparison
    print("\n" + "=" * 60)
    print("Internal metrics (for verification):")
    print("=" * 60)
    from src.prediction.metrics import compute_metrics_df, format_metric_report
    df_val = pd.DataFrame({
        "cell_line_id": val_labels["cell_line_id"].values,
        "perturbation_gene": val_labels["perturbation_gene"].values,
        "prediction": val_preds,
        "truth": val_labels["label"].values,
    })
    our_metrics = compute_metrics_df(df_val)
    print(format_metric_report(our_metrics))


if __name__ == "__main__":
    main()
