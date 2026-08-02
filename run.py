"""Entry point for the mitochondrial gene dependency modeling project.

Usage:
    python run.py                          # Run Problem 1 pipeline (default)
    python run.py --task predict           # Run Problem 2 prediction
    python run.py --task validate          # Run Problem 2 cross-validation
    python run.py --task interpret         # Generate interpretability outputs
    python run.py --config config.yaml     # Specify config file
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mitochondrial gene dependency modeling — Problems 1 & 2."
    )
    parser.add_argument(
        "--task", "-t",
        type=str,
        default="indicators",
        choices=["indicators", "predict", "validate", "interpret"],
        help="Task to run: indicators (Problem 1), predict (Problem 2), "
             "validate (CV), interpret (explainability)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.task == "indicators":
        _run_indicators(config)
    elif args.task == "predict":
        _run_prediction(config)
    elif args.task == "validate":
        _run_validation(config)
    elif args.task == "interpret":
        _run_interpret(config)


def _run_indicators(config: dict) -> None:
    """Problem 1: Compute mitochondrial expression module indicators."""
    from src.indicators import run_pipeline, export_outputs

    print("=" * 60)
    print("Problem 1: Mitochondrial Expression Module Indicators")
    print("=" * 60)
    results = run_pipeline(config)

    print("\n" + "=" * 60)
    print("Exporting outputs...")
    export_outputs(results, config)

    indicators = results["indicators"]
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Output indicators: {indicators.shape[0]} cell lines × {indicators.shape[1]} modules")
    print(f"  Output directory: {config['paths']['output_dir']}/")
    print("=" * 60)


def _run_prediction(config: dict) -> None:
    """Problem 2: Train models and generate submission."""
    from src.prediction.predict import run_prediction

    print("=" * 60)
    print("Problem 2: Gene Dependency Prediction")
    print("=" * 60)
    result = run_prediction(config)

    print("\n" + "=" * 60)
    print("Prediction complete!")
    train_m = result["train_metrics"]
    print(f"  Training S: {train_m['final_score']:.4f}")
    print(f"  Submission: outputs/prediction/submission.csv")
    print("=" * 60)


def _run_validation(config: dict) -> None:
    """Problem 2: Run cross-validation."""
    from src.prediction.features import build_all_features
    from src.prediction.baselines import (
        shrink_gene_means, shrink_cell_means,
        build_collaborative_features,
    )
    from src.prediction.validation import run_validation

    print("=" * 60)
    print("Problem 2: Cross-Validation")
    print("=" * 60)

    # Load labels and build features
    import pandas as pd
    from pathlib import Path

    data_dir = Path(config["paths"]["data_dir"])
    labels = pd.read_csv(data_dir / "labels" / "gene_dependency.csv")

    print("Building features...")
    X, meta = build_all_features(
        labels[["cell_line_id", "perturbation_gene"]], config,
    )

    # G5 features
    gene_bl = shrink_gene_means(labels)
    cell_bl = shrink_cell_means(labels)
    g5 = build_collaborative_features(
        labels[["cell_line_id", "perturbation_gene"]], gene_bl, cell_bl,
    )
    for col in g5.columns:
        if col not in X.columns:
            X[col] = g5[col].values

    y = labels["label"].to_numpy(dtype=np.float64)
    cell_ids = labels["cell_line_id"].to_numpy()
    gene_ids = labels["perturbation_gene"].to_numpy()

    report = run_validation(X, y, cell_ids, gene_ids, config=config)

    # Save report
    import json
    output_dir = Path(config["paths"]["output_dir"]) / "prediction"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\nValidation report saved to: {output_dir / 'validation_report.json'}")
    print("=" * 60)


def _run_interpret(config: dict) -> None:
    """Problem 2: Generate interpretability outputs."""
    from src.prediction.predict import run_prediction
    from src.prediction.interpret import generate_all_interpretations

    print("=" * 60)
    print("Problem 2: Model Interpretability")
    print("=" * 60)

    # First train models
    print("Training models...")
    pred_result = run_prediction(config)

    # Then generate interpretations
    print("\nGenerating interpretations...")
    interp_result = generate_all_interpretations(pred_result, config)

    print("\n" + "=" * 60)
    print("Interpretability outputs saved to: outputs/prediction/")
    print("=" * 60)


if __name__ == "__main__":
    main()
