"""Tests for src/prediction/validation.py."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.prediction.validation import validate_group_by_gene, validate_group_by_cell
from src.prediction.baselines import shrink_gene_means, shrink_cell_means
from src.prediction.features import (
    build_gene_static_features, build_gene_expression_profile_features,
    build_cell_features, build_lineage_onehot, build_pair_features,
    assemble_feature_table,
)
from src.preprocess import build_gene_module_map, compute_evidence_weights
from src.utils import load_config


@pytest.fixture(scope="module")
def config():
    return load_config()


class TestGroupByGeneCV:
    def test_folds_have_zero_gene_overlap(self):
        """Each fold's validation genes must not appear in training."""
        rng = np.random.RandomState(42)
        genes = np.array([f"G{i}" for i in range(20)])
        gene_ids = np.concatenate([genes] * 10)  # 200 rows
        from sklearn.model_selection import GroupKFold
        kf = GroupKFold(n_splits=5)
        for train_idx, val_idx in kf.split(np.arange(len(gene_ids)), groups=gene_ids):
            train_genes = set(gene_ids[train_idx])
            val_genes = set(gene_ids[val_idx])
            assert len(train_genes & val_genes) == 0, \
                f"Overlap: {train_genes & val_genes}"

    def test_validate_group_by_gene_runs(self, config):
        """Smoke test: validation runs without error on small data."""
        rng = np.random.RandomState(42)
        n = 200
        X_df = pd.DataFrame({
            "cell_line_id": [f"C{i % 10}" for i in range(n)],
            "perturbation_gene": [f"G{i % 20}" for i in range(n)],
            "pair_z_cg": rng.randn(n).astype(np.float32),
            "g1_gene_module_00": rng.randn(n).astype(np.float32),
            "g3_cell_indicator_0": rng.randn(n).astype(np.float32),
            "g4_pair_delta_ewm_00": rng.randn(n).astype(np.float32),
            "g5_gene_baseline": rng.randn(n).astype(np.float32),
            "g5_cell_bias": rng.randn(n).astype(np.float32),
        })
        y = (0.5 * X_df["pair_z_cg"].to_numpy() +
             0.3 * rng.randn(n)).astype(np.float64)
        cells = X_df["cell_line_id"].to_numpy()
        genes = X_df["perturbation_gene"].to_numpy()

        # Inject metadata path
        cfg = dict(config)
        cfg["paths"] = dict(config["paths"])

        summary = validate_group_by_gene(X_df, y, cells, genes, n_folds=3, config=cfg)
        assert "final_score_mean" in summary
        assert "final_score_std" in summary
        assert summary["n_folds"] == 3


class TestGroupByCellCV:
    def test_folds_have_zero_cell_overlap(self):
        """Each fold's validation cells must not appear in training."""
        rng = np.random.RandomState(42)
        cells = np.array([f"C{i}" for i in range(10)])
        cell_ids = np.repeat(cells, 20)  # 200 rows
        from sklearn.model_selection import GroupKFold
        kf = GroupKFold(n_splits=5)
        for train_idx, val_idx in kf.split(np.arange(len(cell_ids)), groups=cell_ids):
            train_cells = set(cell_ids[train_idx])
            val_cells = set(cell_ids[val_idx])
            assert len(train_cells & val_cells) == 0

    def test_validate_group_by_cell_runs(self, config):
        """Smoke test: cell-group CV runs without error."""
        rng = np.random.RandomState(42)
        n = 200
        X_df = pd.DataFrame({
            "cell_line_id": [f"C{i % 10}" for i in range(n)],
            "perturbation_gene": [f"G{i % 20}" for i in range(n)],
            "pair_z_cg": rng.randn(n).astype(np.float32),
            "g1_gene_module_00": rng.randn(n).astype(np.float32),
            "g3_cell_indicator_0": rng.randn(n).astype(np.float32),
            "g4_pair_delta_ewm_00": rng.randn(n).astype(np.float32),
            "g5_gene_baseline": rng.randn(n).astype(np.float32),
            "g5_cell_bias": rng.randn(n).astype(np.float32),
        })
        y = (0.5 * X_df["pair_z_cg"].to_numpy() +
             0.3 * rng.randn(n)).astype(np.float64)
        cells = X_df["cell_line_id"].to_numpy()
        genes = X_df["perturbation_gene"].to_numpy()

        cfg = dict(config)
        cfg["paths"] = dict(config["paths"])

        summary = validate_group_by_cell(X_df, y, cells, genes, n_folds=3, config=cfg)
        assert "final_score_mean" in summary
        assert summary["n_folds"] == 3


class TestLeakageAssertions:
    def test_shrink_gene_means_no_future_info(self):
        """Out-of-fold gene means must not use validation labels."""
        labels = pd.DataFrame({
            "perturbation_gene": ["G1", "G1", "G2", "G2", "G3"],
            "cell_line_id": ["C1", "C2", "C1", "C2", "C1"],
            "label": [1.0, 2.0, 0.5, 0.3, -0.2],
        })
        train_genes = {"G1", "G2"}
        train_labels = labels[labels["perturbation_gene"].isin(train_genes)]
        shrunk = shrink_gene_means(train_labels)
        # G3 should NOT be in shrunk
        assert "G3" not in shrunk
