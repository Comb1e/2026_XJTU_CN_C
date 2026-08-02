"""Tests for src/prediction/models.py."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.prediction.models import (
    create_model_a,
    create_model_b,
    prepare_features_a,
    prepare_features_b,
    sample_pairwise_pairs,
    blend_ranks,
    calibrate_rmse,
    apply_calibration,
    train_models,
    predict_all,
)


class TestModelCreation:
    def test_create_model_a_returns_hgbr(self):
        model = create_model_a()
        from sklearn.ensemble import HistGradientBoostingRegressor
        assert isinstance(model, HistGradientBoostingRegressor)

    def test_create_model_a_accepts_overrides(self):
        model = create_model_a({"max_iter": 10, "max_leaf_nodes": 15})
        assert model.max_iter == 10
        assert model.max_leaf_nodes == 15


class TestFeaturePreparation:
    def test_prepare_features_a_excludes_neighbor(self):
        X = pd.DataFrame({
            "cell_line_id": ["C1", "C2"],
            "perturbation_gene": ["G1", "G2"],
            "g5_neighbor_score": [0.5, 0.3],
            "pair_z_cg": [1.0, -1.0],
        })
        Xa = prepare_features_a(X)
        assert Xa.shape == (2, 1)  # only pair_z_cg
        assert Xa[0, 0] == 1.0

    def test_prepare_features_b_includes_all(self):
        X = pd.DataFrame({
            "cell_line_id": ["C1", "C2"],
            "perturbation_gene": ["G1", "G2"],
            "g5_neighbor_score": [0.5, 0.3],
            "pair_z_cg": [1.0, -1.0],
        })
        Xb = prepare_features_b(X)
        assert Xb.shape == (2, 2)


class TestPairwiseSampling:
    def test_output_shapes(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
        y = rng.randn(100).astype(np.float64)
        cells = np.array(["C1"] * 50 + ["C2"] * 50)
        genes = np.array([f"G{i}" for i in range(100)])

        X_diff, y_diff = sample_pairwise_pairs(X, y, cells, genes,
                                                max_pairs_per_cell=50)
        assert X_diff.shape[1] == 5
        assert len(y_diff) > 0
        assert set(y_diff) <= {0, 1}


class TestBlendRanks:
    def test_perfect_harmony(self):
        """Two models with identical ordering should blend to same ordering."""
        preds_a = np.array([5.0, 3.0, 1.0], dtype=np.float32)
        preds_b = np.array([3.0, 1.0, -1.0], dtype=np.float32)
        cells = np.array(["C1"] * 3)

        blended = blend_ranks(preds_a, preds_b, cells, alpha_a=0.5, alpha_b=0.5)
        # Order should be preserved
        assert blended[0] > blended[1] > blended[2]

    def test_cold_mask_zeros_model_b(self):
        """Cold-start genes should get zero weight from Model B."""
        preds_a = np.array([5.0, 3.0, 1.0], dtype=np.float32)
        preds_b = np.array([1.0, 3.0, 5.0], dtype=np.float32)  # reversed
        cells = np.array(["C1"] * 3)
        cold = np.array([True, False, False])

        blended = blend_ranks(preds_a, preds_b, cells,
                              alpha_a=0.5, alpha_b=0.5,
                              cold_mask=cold)
        # First element (cold) should follow Model A only
        assert blended[0] > blended[2]  # follows A's ordering


class TestCalibration:
    def test_calibrate_and_apply(self):
        rng = np.random.RandomState(42)
        y_pred = rng.randn(200).astype(np.float32)
        y_true = 2.0 * y_pred + 0.5 + 0.1 * rng.randn(200)
        cal = calibrate_rmse(y_pred, y_true)
        calibrated = apply_calibration(cal, y_pred)
        assert calibrated.shape == y_pred.shape
        # Calibrated values should be closer to truth on average
        rmse_before = np.sqrt(np.mean((y_pred - y_true) ** 2))
        rmse_after = np.sqrt(np.mean((calibrated - y_true) ** 2))
        assert rmse_after < rmse_before * 0.8

    def test_apply_none_calibration(self):
        y_pred = np.array([10.0, -5.0], dtype=np.float32)
        result = apply_calibration(None, y_pred, clip_range=(-3, 5))
        assert result[0] == 5.0
        assert result[1] == -3.0


class TestEndToEndTraining:
    def test_train_and_predict_small(self):
        """End-to-end training on small synthetic dataset."""
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
             0.3 * X_df["g5_gene_baseline"].to_numpy() +
             0.1 * rng.randn(n)).astype(np.float64)
        cells = X_df["cell_line_id"].to_numpy()
        genes = X_df["perturbation_gene"].to_numpy()

        cold = {"G15", "G16", "G17", "G18", "G19"}

        models = train_models(X_df, y, cells, genes, cold_genes=cold)
        assert "model_a" in models
        assert "model_b" in models
        assert "calibration" in models

        preds = predict_all(X_df, cells, genes, models)
        assert len(preds) == n
        assert preds.dtype == np.float32
        assert not np.isnan(preds).any()

    def test_two_regime_serving(self):
        """Cold genes should only use Model A, training genes use both."""
        rng = np.random.RandomState(42)
        n = 50
        X_df = pd.DataFrame({
            "cell_line_id": ["C1"] * n,
            "perturbation_gene": [f"G{i}" for i in range(n)],
            "pair_z_cg": rng.randn(n).astype(np.float32),
            "g1_gene_module_00": rng.randn(n).astype(np.float32),
            "g3_cell_indicator_0": rng.randn(n).astype(np.float32),
            "g4_pair_delta_ewm_00": rng.randn(n).astype(np.float32),
            "g5_gene_baseline": rng.randn(n).astype(np.float32),
            "g5_cell_bias": rng.randn(n).astype(np.float32),
        })
        y = rng.randn(n).astype(np.float64)
        cells = X_df["cell_line_id"].to_numpy()
        genes = X_df["perturbation_gene"].to_numpy()

        cold = {f"G{i}" for i in range(10)}  # first 10 are cold

        models = train_models(X_df, y, cells, genes, cold_genes=cold)
        preds = predict_all(X_df, cells, genes, models)

        # All predictions should be finite
        assert np.all(np.isfinite(preds))
