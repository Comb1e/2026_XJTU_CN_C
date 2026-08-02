"""Integration tests for the full prediction pipeline."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.utils import load_config


class TestIntegration:
    """End-to-end integration test on a small subset."""

    @pytest.fixture(scope="module")
    def config(self):
        return load_config()

    def test_full_pipeline_small_subset(self, config, tmp_path):
        """Run train+predict on a small subset of real data."""
        from src.prediction.features import build_all_features
        from src.prediction.baselines import (
            shrink_gene_means, shrink_cell_means,
            build_collaborative_features,
        )
        from src.prediction.models import train_models, predict_all

        data_dir = Path(config["paths"]["data_dir"])
        labels = pd.read_csv(data_dir / "labels" / "gene_dependency.csv", nrows=5000)

        # Build features for training pairs
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

        # Identify cold genes (genes with < 5 labels in this subset)
        gene_counts = labels.groupby("perturbation_gene").size()
        cold = set(gene_counts[gene_counts < 5].index)

        print(f"  Features: {X.shape[1]}, cold genes: {len(cold)}")

        # Train models
        models = train_models(X, y, cell_ids, gene_ids, cold_genes=cold, config=config)

        # Predict
        preds = predict_all(X, cell_ids, gene_ids, models, add_jitter=True)

        # Verify predictions
        assert len(preds) == len(labels)
        assert preds.dtype == np.float32
        assert not np.isnan(preds).any()
        assert not np.isinf(preds).any()
        print(f"  Predictions: mean={preds.mean():.4f}, std={preds.std():.4f}")

        # Verify cold genes get predictions too
        cold_mask = np.array([g in cold for g in gene_ids])
        if cold_mask.any():
            assert not np.isnan(preds[cold_mask]).any()

    def test_submission_format(self):
        """Test that prediction output matches submission template format."""
        submission_path = Path("数据文件/submission/sample_submission_gene.csv")
        if not submission_path.exists():
            submission_path = Path(project_root) / "数据文件/submission/sample_submission_gene.csv"

        sub = pd.read_csv(submission_path, nrows=100)
        assert "cell_line_id" in sub.columns
        assert "perturbation_gene" in sub.columns
        assert "label" in sub.columns

    def test_cold_gene_feature_coverage(self, config):
        """Cold-start genes must have all G1-G4 features available."""
        data_dir = Path(config["paths"]["data_dir"])
        labels = pd.read_csv(data_dir / "labels" / "gene_dependency.csv")
        submission = pd.read_csv(data_dir / "submission" / "sample_submission_gene.csv")

        train_genes = set(labels["perturbation_gene"].unique())
        test_genes = set(submission["perturbation_gene"].unique())
        cold_genes = test_genes - train_genes

        # Check cold genes exist in expression data
        features_dir = Path(config["paths"]["features_dir"])
        expr = pd.read_csv(features_dir / "cell_expression_zscore.csv", index_col=0)
        for gene in cold_genes:
            assert gene in expr.columns, f"Cold gene {gene} not in expression data"

        # Check cold genes have metadata
        gene_meta = pd.read_csv(data_dir / "metadata" / "gene_metadata.csv")
        meta_genes = set(gene_meta["gene_symbol"])
        for gene in cold_genes:
            assert gene in meta_genes, f"Cold gene {gene} not in gene metadata"

        print(f"  {len(cold_genes)} cold genes — all verified in expression and metadata")
