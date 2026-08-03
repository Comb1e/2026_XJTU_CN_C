"""Tests for src/prediction/ interpretable white-box models.

Tests cover:
  - FactorAnalysisModel: fit, predict, cold gene imputation
  - SparseElasticNetModel: fit, predict, sparsity
  - PCARidgeModel: fit, predict, feature importance back-mapping
  - PLSModel: fit, predict
  - SplineGAMModel: fit, predict, partial dependence
  - RidgeBlend: optimal component weighting
  - QuantileCalibrator: monotone distribution matching
  - Pairwise ranking (LogisticRegression Bradley-Terry)
  - Calibration utilities (Ridge, quantile)
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.prediction.whitebox import (
    FactorAnalysisModel,
    SparseElasticNetModel,
    PCARidgeModel,
    PLSModel,
    SplineGAMModel,
    RidgeBlend,
    QuantileCalibrator,
)
from src.prediction.models import (
    sample_pairwise_pairs,
    train_pairwise_ranker,
    predict_pairwise_ranker,
    blend_ranks,
    blend_ranks_multi,
    calibrate_rmse,
    apply_calibration,
    calibrate_quantile,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Factor Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestFactorAnalysisModel:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n_cells, n_genes, K = 50, 30, 3
        # Generate structured data: cell_scores @ gene_loadings.T + noise
        cell_scores_true = rng.randn(n_cells, K).astype(np.float64)
        gene_loadings_true = rng.randn(n_genes, K).astype(np.float64)
        R = cell_scores_true @ gene_loadings_true.T + 0.1 * rng.randn(n_cells, n_genes)

        fa = FactorAnalysisModel(n_components=K, random_state=42)
        fa.fit(
            R,
            cell_index=pd.Index([f"C{i}" for i in range(n_cells)]),
            gene_index=pd.Index([f"G{i}" for i in range(n_genes)]),
        )

        # Predict
        cell_ids = [f"C{i}" for i in range(n_cells)]
        gene_ids = [f"G{i}" for i in range(n_genes)]
        preds = fa.predict(cell_ids * n_genes, [f"G{i % n_genes}" for i in range(n_cells * n_genes)])
        assert len(preds) == n_cells * n_genes
        assert preds.dtype == np.float32
        assert not np.isnan(preds).any()

    def test_top_genes_per_factor(self):
        rng = np.random.RandomState(42)
        n_cells, n_genes, K = 30, 20, 5
        R = rng.randn(n_cells, n_genes)
        fa = FactorAnalysisModel(n_components=K, random_state=42)
        fa.fit(
            R,
            cell_index=pd.Index([f"C{i}" for i in range(n_cells)]),
            gene_index=pd.Index([f"G{i}" for i in range(n_genes)]),
        )
        top = fa.get_top_genes_per_factor(0, top_n=5)
        assert len(top) == 5
        assert all(isinstance(g, str) for g, _ in top)
        assert all(isinstance(v, float) for _, v in top)

    def test_cold_gene_imputation(self):
        rng = np.random.RandomState(42)
        n_cells, n_genes = 40, 25
        R = rng.randn(n_cells, n_genes)

        # Gene features for imputation — use distinct column names to avoid overlap
        gene_static = pd.DataFrame(
            rng.randn(n_genes + 3, 10).astype(np.float32),
            index=[f"G{i}" for i in range(n_genes + 3)],
            columns=[f"gs_{i}" for i in range(10)],
        )
        gene_expr = pd.DataFrame(
            rng.randn(n_genes + 3, 5).astype(np.float32),
            index=[f"G{i}" for i in range(n_genes + 3)],
            columns=[f"ge_{i}" for i in range(5)],
        )
        cold_genes = {"G20", "G21", "G22"}

        fa = FactorAnalysisModel(n_components=3, random_state=42)
        fa.fit(
            R,
            cell_index=pd.Index([f"C{i}" for i in range(n_cells)]),
            gene_index=pd.Index([f"G{i}" for i in range(n_genes)]),
            cold_genes=cold_genes,
            gene_static_features=gene_static,
            gene_expr_profile_features=gene_expr,
        )

        # Cold genes should be in gene_index_ after imputation
        for g in cold_genes:
            assert g in fa.gene_index_


# ═══════════════════════════════════════════════════════════════════════════════
# Sparse ElasticNet
# ═══════════════════════════════════════════════════════════════════════════════

class TestSparseElasticNetModel:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n, d = 200, 30
        true_coef = np.zeros(d)
        true_coef[:5] = [1.0, -0.5, 0.3, -0.2, 0.1]
        X = rng.randn(n, d).astype(np.float64)
        y = X @ true_coef + 0.1 * rng.randn(n)

        model = SparseElasticNetModel(l1_ratio=0.5, n_alphas=20, random_state=42)
        model.fit(X, y)

        preds = model.predict(X)
        assert len(preds) == n
        assert model.n_nonzero_ > 0
        assert model.n_nonzero_ <= d

    def test_sparsity(self):
        """ElasticNet should produce sparse coefficients with L1 penalty."""
        rng = np.random.RandomState(42)
        n, d = 300, 50
        true_coef = np.zeros(d)
        true_coef[0] = 2.0  # only one real signal
        X = rng.randn(n, d).astype(np.float64)
        y = X @ true_coef + 0.5 * rng.randn(n)

        model = SparseElasticNetModel(l1_ratio=0.9, n_alphas=20, random_state=42)
        model.fit(X, y)
        # Should be sparse: many zeros
        assert model.n_nonzero_ < d // 2

    def test_top_features(self):
        rng = np.random.RandomState(42)
        n, d = 200, 20
        true_coef = np.zeros(d)
        true_coef[0] = 1.5
        true_coef[3] = -0.8
        X = rng.randn(n, d).astype(np.float64)
        y = X @ true_coef + 0.05 * rng.randn(n)
        names = [f"feat_{i}" for i in range(d)]

        model = SparseElasticNetModel(l1_ratio=0.5, n_alphas=20, random_state=42)
        model.fit(X, y, feature_names=names)

        top = model.get_top_features(top_n=5)
        assert len(top) == 5
        assert all(isinstance(n, str) for n, _ in top)


# ═══════════════════════════════════════════════════════════════════════════════
# PCA-Ridge
# ═══════════════════════════════════════════════════════════════════════════════

class TestPCARidgeModel:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n, d = 200, 40
        X = rng.randn(n, d).astype(np.float64)
        true_w = rng.randn(d)
        y = X @ true_w + 0.1 * rng.randn(n)

        model = PCARidgeModel(n_components=10, ridge_alpha=1.0, random_state=42)
        model.fit(X, y)

        preds = model.predict(X)
        assert len(preds) == n
        assert preds.dtype == np.float32
        assert not np.isnan(preds).any()

    def test_feature_importance_back_mapping(self):
        rng = np.random.RandomState(42)
        n, d = 200, 30
        X = rng.randn(n, d).astype(np.float64)
        true_w = rng.randn(d)
        y = X @ true_w + 0.1 * rng.randn(n)
        names = [f"feat_{i}" for i in range(d)]

        model = PCARidgeModel(n_components=10, ridge_alpha=1.0, random_state=42)
        model.fit(X, y, feature_names=names)

        assert model.feature_importance_ is not None
        assert len(model.feature_importance_) == d
        top = model.get_top_features(top_n=10)
        assert len(top) == 10

    def test_explained_variance(self):
        rng = np.random.RandomState(42)
        n, d = 200, 40
        X = rng.randn(n, d).astype(np.float64)
        y = rng.randn(n)

        model = PCARidgeModel(n_components=10, random_state=42)
        model.fit(X, y)
        evr = model.explained_variance_ratio_
        assert len(evr) == 10
        assert np.all(evr >= 0)
        assert np.all(evr <= 1)


# ═══════════════════════════════════════════════════════════════════════════════
# PLS
# ═══════════════════════════════════════════════════════════════════════════════

class TestPLSModel:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n, d = 200, 30
        X = rng.randn(n, d).astype(np.float64)
        # Make y a linear combination of first 5 features
        y = X[:, :5].sum(axis=1) + 0.1 * rng.randn(n)

        model = PLSModel(n_components=5, random_state=42)
        model.fit(X, y)

        preds = model.predict(X)
        assert len(preds) == n
        assert preds.dtype == np.float32
        # Should have reasonable correlation
        corr = np.corrcoef(preds, y)[0, 1]
        assert corr > 0.3

    def test_feature_importance(self):
        rng = np.random.RandomState(42)
        n, d = 200, 20
        X = rng.randn(n, d).astype(np.float64)
        y = X[:, 0] * 3.0 + X[:, 1] * 2.0 + 0.1 * rng.randn(n)

        model = PLSModel(n_components=5, random_state=42)
        model.fit(X, y)
        imp = model.get_feature_importance()
        assert len(imp) == d


# ═══════════════════════════════════════════════════════════════════════════════
# Spline-GAM
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplineGAMModel:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n, d = 300, 15
        X = rng.randn(n, d).astype(np.float64)
        # Nonlinear relationship in first feature
        y = np.sin(X[:, 0]) + 0.3 * X[:, 1] + 0.1 * rng.randn(n)

        model = SplineGAMModel(max_features=5, spline_smooth=0.3, random_state=42)
        model.fit(X, y)

        preds = model.predict(X)
        assert len(preds) == n
        assert preds.dtype == np.float32
        assert not np.isnan(preds).any()

    def test_partial_dependence(self):
        rng = np.random.RandomState(42)
        n, d = 200, 10
        X = rng.randn(n, d).astype(np.float64)
        y = np.sin(X[:, 0]) + 0.1 * rng.randn(n)

        model = SplineGAMModel(max_features=5, spline_smooth=0.3, random_state=42)
        model.fit(X, y)

        # First feature should be selected (highest correlation)
        assert 0 in model.selected_features_
        x_grid, y_vals = model.get_partial_dependence(0, n_points=50)
        assert len(x_grid) == 50
        assert len(y_vals) == 50


# ═══════════════════════════════════════════════════════════════════════════════
# RidgeBlend
# ═══════════════════════════════════════════════════════════════════════════════

class TestRidgeBlend:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n = 200
        # Two components: one good, one noisy
        comp1 = rng.randn(n).astype(np.float32)  # good signal
        comp2 = rng.randn(n).astype(np.float32)  # noise
        y = 2.0 * comp1 + 0.0 * comp2 + 0.1 * rng.randn(n)

        blend = RidgeBlend(alphas=[0.1, 1.0, 10.0])
        blend.fit([comp1, comp2], y, component_names=["good", "noisy"])

        preds = blend.predict([comp1, comp2])
        assert len(preds) == n

        # Good component should get larger weight
        weights = dict(blend.get_component_weights())
        assert abs(weights["good"]) > abs(weights["noisy"])

    def test_weights_summary(self):
        rng = np.random.RandomState(42)
        n = 100
        comps = [rng.randn(n).astype(np.float32) for _ in range(3)]
        y = comps[0] + 0.5 * comps[1] + 0.1 * rng.randn(n)

        blend = RidgeBlend()
        blend.fit(comps, y, component_names=["a", "b", "c"])
        weights = blend.get_component_weights()
        assert len(weights) == 3
        assert blend.alpha_ > 0


# ═══════════════════════════════════════════════════════════════════════════════
# QuantileCalibrator
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuantileCalibrator:
    def test_monotone_preservation(self):
        rng = np.random.RandomState(42)
        y_pred = rng.randn(500).astype(np.float32)
        y_true = 2.0 * y_pred + 0.5 + 0.1 * rng.randn(500)

        cal = QuantileCalibrator(n_quantiles=100)
        cal.fit(y_pred, y_true)
        calibrated = cal.transform(y_pred)

        # Monotonicity: if pred_a > pred_b, then cal_a >= cal_b
        for _ in range(100):
            i, j = rng.choice(500, 2, replace=False)
            if y_pred[i] > y_pred[j]:
                assert calibrated[i] >= calibrated[j] - 1e-10

    def test_distribution_matching(self):
        """Calibrated predictions should have similar distribution to labels."""
        rng = np.random.RandomState(42)
        y_pred = rng.rand(1000).astype(np.float32) * 2 - 1  # [-1, 1]
        y_true = rng.randn(1000) * 0.5 + 0.3  # N(0.3, 0.5)

        cal = QuantileCalibrator(n_quantiles=200)
        cal.fit(y_pred, y_true)
        calibrated = cal.transform(y_pred)

        # Mean should shift toward label mean
        assert abs(calibrated.mean() - y_true.mean()) < abs(y_pred.mean() - y_true.mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Pairwise ranking (LogisticRegression — interpretable)
# ═══════════════════════════════════════════════════════════════════════════════

class TestPairwiseRanking:
    def test_sample_pairwise_pairs(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
        y = rng.randn(100).astype(np.float64)
        cells = np.array(["C1"] * 50 + ["C2"] * 50)
        genes = np.array([f"G{i}" for i in range(100)])

        X_diff, y_diff = sample_pairwise_pairs(X, y, cells, genes,
                                                max_pairs_per_cell=50)
        assert X_diff.shape[1] == 5
        assert len(y_diff) > 0
        assert set(np.unique(y_diff)) <= {0, 1}

    def test_train_and_predict(self):
        rng = np.random.RandomState(42)
        n, d = 200, 10
        true_w = rng.randn(d)
        X = rng.randn(n, d).astype(np.float32)
        y = X @ true_w + 0.5 * rng.randn(n)
        cells = np.array(["C1"] * 50 + ["C2"] * 50 + ["C3"] * 50 + ["C4"] * 50)
        genes = np.array([f"G{i}" for i in range(n)])

        model = train_pairwise_ranker(X, y, cells, genes)
        assert model is not None
        preds = predict_pairwise_ranker(model, X)
        assert len(preds) == n
        assert preds.dtype == np.float32

    def test_returns_none_for_insufficient_data(self):
        rng = np.random.RandomState(42)
        # Only one example per cell — no pairs possible
        X = rng.randn(5, 3).astype(np.float32)
        y = np.zeros(5)
        cells = np.array(["C1", "C2", "C3", "C4", "C5"])
        genes = np.array(["G1", "G2", "G3", "G4", "G5"])
        model = train_pairwise_ranker(X, y, cells, genes)
        assert model is None


# ═══════════════════════════════════════════════════════════════════════════════
# Blend ranks (algebraic)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBlendRanks:
    def test_perfect_harmony(self):
        preds_a = np.array([5.0, 3.0, 1.0], dtype=np.float32)
        preds_b = np.array([3.0, 1.0, -1.0], dtype=np.float32)
        cells = np.array(["C1"] * 3)
        blended = blend_ranks(preds_a, preds_b, cells, alpha_a=0.5, alpha_b=0.5)
        assert blended[0] > blended[1] > blended[2]

    def test_cold_mask_zeros_model_b(self):
        preds_a = np.array([5.0, 3.0, 1.0], dtype=np.float32)
        preds_b = np.array([1.0, 3.0, 5.0], dtype=np.float32)  # reversed
        cells = np.array(["C1"] * 3)
        cold = np.array([True, False, False])
        blended = blend_ranks(preds_a, preds_b, cells,
                              alpha_a=0.5, alpha_b=0.5, cold_mask=cold)
        assert blended[0] > blended[2]  # follows A's ordering

    def test_blend_ranks_multi(self):
        preds_a = np.array([5.0, 3.0, 1.0], dtype=np.float32)
        preds_b = np.array([5.0, 3.0, 1.0], dtype=np.float32)
        cells = np.array(["C1"] * 3)
        result = blend_ranks_multi(
            cells,
            models=[(preds_a, 1.0, True), (preds_b, 1.0, True)],
        )
        assert len(result) == 3
        assert result[0] > result[2]


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration (Ridge + Quantile — interpretable)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCalibration:
    def test_calibrate_and_apply(self):
        rng = np.random.RandomState(42)
        y_pred = rng.randn(200).astype(np.float32)
        y_true = 2.0 * y_pred + 0.5 + 0.1 * rng.randn(200)
        cal = calibrate_rmse(y_pred, y_true)
        calibrated = apply_calibration(cal, y_pred)
        assert calibrated.shape == y_pred.shape
        rmse_before = np.sqrt(np.mean((y_pred - y_true) ** 2))
        rmse_after = np.sqrt(np.mean((calibrated - y_true) ** 2))
        assert rmse_after < rmse_before * 0.8

    def test_apply_none_calibration(self):
        y_pred = np.array([10.0, -5.0], dtype=np.float32)
        result = apply_calibration(None, y_pred, clip_range=(-3, 5))
        assert result[0] == 5.0
        assert result[1] == -3.0

    def test_quantile_calibrator_monotone(self):
        """Quantile calibration preserves ordering (Spearman correlation)."""
        from scipy.stats import spearmanr
        rng = np.random.RandomState(42)
        # calibrate_quantile expects predictions in [0, 1] range
        y_pred = rng.rand(500).astype(np.float32)  # uniform in [0, 1]
        y_true = 3.0 * y_pred + 1.0 + 0.2 * rng.randn(500)
        cal_fn = calibrate_quantile(y_pred, y_true)
        calibrated = cal_fn(y_pred)
        assert calibrated.shape == y_pred.shape
        # Quantile mapping is monotone → preserves rank ordering well
        spearman_before = spearmanr(y_pred, y_true)[0]
        spearman_after = spearmanr(calibrated, y_true)[0]
        # Spearman should be very close (monotone transform with fine grid)
        assert abs(spearman_after - spearman_before) < 0.01
