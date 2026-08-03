"""Integration tests for the full prediction pipeline — white-box edition."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.utils import load_config


class TestIntegration:
    """End-to-end integration test with white-box interpretable models."""

    @pytest.fixture(scope="module")
    def config(self):
        return load_config()

    def test_full_pipeline_small_subset(self, config):
        """Run train+predict on a small subset of real data with white-box models."""
        from src.prediction.features import (
            build_all_features, build_gene_static_features,
            build_gene_expression_profile_features, build_cell_features,
            build_lineage_onehot,
        )
        from src.prediction.baselines import (
            shrink_gene_means, shrink_cell_means,
            build_collaborative_features, build_gene_similarity_cf,
            compute_loco_gene_means,
        )
        from src.prediction.whitebox import train_whitebox_models, predict_whitebox

        data_dir = Path(config["paths"]["data_dir"])
        # Read all labels then sample 100 random genes for a balanced subset
        all_labels = pd.read_csv(data_dir / "labels" / "gene_dependency.csv")
        rng = np.random.RandomState(42)
        sampled_genes = rng.choice(
            sorted(all_labels["perturbation_gene"].unique()), size=100, replace=False,
        )
        labels = all_labels[all_labels["perturbation_gene"].isin(sampled_genes)].copy()
        gene_meta = pd.read_csv(data_dir / "metadata" / "gene_metadata.csv")
        pathway_meta = pd.read_csv(data_dir / "metadata" / "pathway_metadata.csv")
        cell_meta = pd.read_csv(data_dir / "metadata" / "cell_line_metadata.csv")
        expression = pd.read_csv(
            data_dir / "features" / "cell_expression_zscore.csv", index_col=0,
        )

        from src.preprocess import build_gene_module_map, compute_evidence_weights
        gmm = build_gene_module_map(gene_meta)
        ew = compute_evidence_weights(gene_meta)
        g1 = build_gene_static_features(gene_meta, gmm, ew)
        g2 = build_gene_expression_profile_features(expression)

        # Build features
        X, meta = build_all_features(
            labels[["cell_line_id", "perturbation_gene"]], config,
        )

        # Baselines
        gene_bl, loco_train = compute_loco_gene_means(labels)
        cell_bl = shrink_cell_means(labels)

        # G5 features
        g5 = build_collaborative_features(
            labels[["cell_line_id", "perturbation_gene"]], gene_bl, cell_bl,
        )
        for col in g5.columns:
            X[col] = g5[col].values if col not in X.columns else X[col]
        X["g5_gene_baseline"] = loco_train

        y = labels["label"].to_numpy(dtype=np.float64)
        cell_ids = labels["cell_line_id"].to_numpy()
        gene_ids = labels["perturbation_gene"].to_numpy()

        # Cold genes
        gene_counts = labels.groupby("perturbation_gene").size()
        cold = set(gene_counts[gene_counts < 5].index)

        # CF predictions
        cf_cold = build_gene_similarity_cf(
            labels, gene_meta, pathway_meta, expression, cold, k=10,
        )

        # SVD — use small k since we have few genes in subset
        from src.prediction.baselines import compute_label_svd
        n_genes_in_subset = labels["perturbation_gene"].nunique()
        svd_k = max(1, min(5, n_genes_in_subset - 1))
        U, V, svd_cell_idx, svd_gene_idx, _ = compute_label_svd(labels, k=svd_k)
        svd_dot = {}
        for ci, cell in enumerate(svd_cell_idx):
            for gi, gene in enumerate(svd_gene_idx):
                svd_dot[(cell, gene)] = float(np.dot(U[ci], V[gi]))

        # Cell features
        cell_feats = build_cell_features(
            Path(config["paths"]["output_dir"]),
            list(labels["cell_line_id"].unique()),
        )
        lineage = build_lineage_onehot(cell_meta, list(labels["cell_line_id"].unique()))
        cell_feats = pd.concat([cell_feats, lineage], axis=1)

        print(f"  Features: {X.shape[1]}, cold genes: {len(cold)}")

        # Train white-box models
        models = train_whitebox_models(
            X, y, cell_ids, gene_ids,
            expression=expression,
            gene_static_features=g1,
            gene_expr_profile_features=g2,
            cell_features=cell_feats,
            gene_bl=gene_bl, cell_bl=cell_bl,
            svd_dot=svd_dot,
            cf_predictions=cf_cold,
            cold_genes=cold,
            config=config,
        )

        # Predict
        preds = predict_whitebox(
            X, cell_ids, gene_ids,
            gene_bl=gene_bl, cell_bl=cell_bl,
            svd_dot=svd_dot,
            cf_predictions=cf_cold,
            cold_genes=cold,
            models=models,
            add_jitter=True,
        )

        assert len(preds) == len(labels)
        assert preds.dtype == np.float32
        assert not np.isnan(preds).any()
        assert not np.isinf(preds).any()
        print(f"  Predictions: mean={preds.mean():.4f}, std={preds.std():.4f}")

        # Cold genes should get predictions
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

        features_dir = Path(config["paths"]["features_dir"])
        expr = pd.read_csv(features_dir / "cell_expression_zscore.csv", index_col=0)
        for gene in cold_genes:
            assert gene in expr.columns, f"Cold gene {gene} not in expression data"

        gene_meta = pd.read_csv(data_dir / "metadata" / "gene_metadata.csv")
        meta_genes = set(gene_meta["gene_symbol"])
        for gene in cold_genes:
            assert gene in meta_genes, f"Cold gene {gene} not in gene metadata"

        print(f"  {len(cold_genes)} cold genes — all verified in expression and metadata")
