"""Tests for src/prediction/validation.py CV framework."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.prediction.baselines import shrink_gene_means, shrink_cell_means
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
        """Integration test: group-by-gene CV runs with real data (small subset)."""
        from src.prediction.validation import validate_group_by_gene
        from src.prediction.features import build_all_features
        from src.prediction.baselines import (
            compute_loco_gene_means, build_collaborative_features,
        )

        data_dir = Path(config["paths"]["data_dir"])
        # Use a small random gene subset for speed
        all_labels = pd.read_csv(data_dir / "labels" / "gene_dependency.csv")
        rng = np.random.RandomState(42)
        sampled_genes = rng.choice(
            sorted(all_labels["perturbation_gene"].unique()), size=30, replace=False,
        )
        labels = all_labels[all_labels["perturbation_gene"].isin(sampled_genes)].copy()

        # Build features
        X, meta = build_all_features(
            labels[["cell_line_id", "perturbation_gene"]], config,
        )
        gene_bl, loco_train = compute_loco_gene_means(labels)
        cell_bl = shrink_cell_means(labels)
        g5 = build_collaborative_features(
            labels[["cell_line_id", "perturbation_gene"]], gene_bl, cell_bl,
        )
        for col in g5.columns:
            X[col] = g5[col].values if col not in X.columns else X[col]
        X["g5_gene_baseline"] = loco_train

        y = labels["label"].to_numpy(dtype=np.float64)
        cell_ids = labels["cell_line_id"].to_numpy()
        gene_ids = labels["perturbation_gene"].to_numpy()

        summary = validate_group_by_gene(X, y, cell_ids, gene_ids, n_folds=3, config=config)
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
        """Integration test: group-by-cell CV runs with real data (small subset)."""
        from src.prediction.validation import validate_group_by_cell
        from src.prediction.features import build_all_features
        from src.prediction.baselines import (
            compute_loco_gene_means, build_collaborative_features,
        )

        data_dir = Path(config["paths"]["data_dir"])
        all_labels = pd.read_csv(data_dir / "labels" / "gene_dependency.csv")
        rng = np.random.RandomState(42)
        sampled_genes = rng.choice(
            sorted(all_labels["perturbation_gene"].unique()), size=30, replace=False,
        )
        labels = all_labels[all_labels["perturbation_gene"].isin(sampled_genes)].copy()

        X, meta = build_all_features(
            labels[["cell_line_id", "perturbation_gene"]], config,
        )
        gene_bl, loco_train = compute_loco_gene_means(labels)
        cell_bl = shrink_cell_means(labels)
        g5 = build_collaborative_features(
            labels[["cell_line_id", "perturbation_gene"]], gene_bl, cell_bl,
        )
        for col in g5.columns:
            X[col] = g5[col].values if col not in X.columns else X[col]
        X["g5_gene_baseline"] = loco_train

        y = labels["label"].to_numpy(dtype=np.float64)
        cell_ids = labels["cell_line_id"].to_numpy()
        gene_ids = labels["perturbation_gene"].to_numpy()

        summary = validate_group_by_cell(X, y, cell_ids, gene_ids, n_folds=3, config=config)
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
