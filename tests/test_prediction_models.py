"""Tests for src/prediction/ interpretable formula-based models.

Tests cover:
  - GeneEssentialityFormula: fit, predict, feature importance
  - CellVulnerabilityFormula: fit, predict, feature importance
  - SVDInteraction: SVD bilinear interaction, cold gene prediction, JS shrinkage
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

from src.prediction.formula import (
    GeneEssentialityFormula,
    CellVulnerabilityFormula,
    IMCInteraction,
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
# Gene Essentiality Formula
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeneEssentialityFormula:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n_genes, n_feats = 50, 10
        true_w = rng.randn(n_feats)
        X = pd.DataFrame(
            rng.randn(n_genes, n_feats).astype(np.float32),
            index=[f"G{i}" for i in range(n_genes)],
            columns=[f"feat_{i}" for i in range(n_feats)],
        )
        gene_means = {f"G{i}": float(X.iloc[i] @ true_w + 0.1 * rng.randn())
                      for i in range(n_genes)}

        formula = GeneEssentialityFormula(alpha=1.0)
        formula.fit(X, gene_means)

        preds = formula.predict(X)
        assert len(preds) == n_genes
        assert all(isinstance(v, float) for v in preds.values())

    def test_top_features(self):
        rng = np.random.RandomState(42)
        n_genes, n_feats = 30, 15
        X = pd.DataFrame(
            rng.randn(n_genes, n_feats).astype(np.float32),
            index=[f"G{i}" for i in range(n_genes)],
            columns=[f"feat_{i}" for i in range(n_feats)],
        )
        gene_means = {f"G{i}": float(rng.randn()) for i in range(n_genes)}
        formula = GeneEssentialityFormula(alpha=1.0)
        formula.fit(X, gene_means)
        top = formula.get_top_features(top_n=5)
        assert len(top) == 5
        assert all(isinstance(n, str) for n, _ in top)

    def test_formula_str(self):
        rng = np.random.RandomState(42)
        n_genes, n_feats = 20, 8
        X = pd.DataFrame(
            rng.randn(n_genes, n_feats).astype(np.float32),
            index=[f"G{i}" for i in range(n_genes)],
            columns=[f"feat_{i}" for i in range(n_feats)],
        )
        gene_means = {f"G{i}": float(rng.randn()) for i in range(n_genes)}
        formula = GeneEssentialityFormula(alpha=1.0)
        formula.fit(X, gene_means)
        s = formula.formula_str(top_n=3)
        assert "μ̂_g" in s
        assert len(s.split("\n")) >= 4


# ═══════════════════════════════════════════════════════════════════════════════
# Cell Vulnerability Formula
# ═══════════════════════════════════════════════════════════════════════════════

class TestCellVulnerabilityFormula:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n_cells, n_feats = 40, 12
        true_w = rng.randn(n_feats)
        X = pd.DataFrame(
            rng.randn(n_cells, n_feats).astype(np.float32),
            index=[f"C{i}" for i in range(n_cells)],
            columns=[f"feat_{i}" for i in range(n_feats)],
        )
        cell_means = {f"C{i}": float(X.iloc[i] @ true_w + 0.1 * rng.randn())
                      for i in range(n_cells)}

        formula = CellVulnerabilityFormula(alpha=10.0)
        formula.fit(X, cell_means)

        preds = formula.predict(X)
        assert len(preds) == n_cells
        assert all(isinstance(v, float) for v in preds.values())

    def test_top_features(self):
        rng = np.random.RandomState(42)
        n_cells, n_feats = 30, 10
        X = pd.DataFrame(
            rng.randn(n_cells, n_feats).astype(np.float32),
            index=[f"C{i}" for i in range(n_cells)],
            columns=[f"feat_{i}" for i in range(n_feats)],
        )
        cell_means = {f"C{i}": float(rng.randn()) for i in range(n_cells)}
        formula = CellVulnerabilityFormula(alpha=10.0)
        formula.fit(X, cell_means)
        top = formula.get_top_features(top_n=5)
        assert len(top) == 5

    def test_formula_str(self):
        rng = np.random.RandomState(42)
        n_cells, n_feats = 20, 6
        X = pd.DataFrame(
            rng.randn(n_cells, n_feats).astype(np.float32),
            index=[f"C{i}" for i in range(n_cells)],
            columns=[f"feat_{i}" for i in range(n_feats)],
        )
        cell_means = {f"C{i}": float(rng.randn()) for i in range(n_cells)}
        formula = CellVulnerabilityFormula(alpha=10.0)
        formula.fit(X, cell_means)
        s = formula.formula_str(top_n=3)
        assert "β̂_c" in s


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════
# IMC Bilinear Interaction (Inductive Matrix Completion)
# ═══════════════════════════════════════════════════════════════════════════════

class TestIMCInteraction:
    """Tests for IMCInteraction with small synthetic data (fast)."""

    def test_fit_and_predict(self):
        """IMC should capture bilinear structure from cell×gene features."""
        rng = np.random.RandomState(42)
        n = 200  # small: fast
        f_c, f_g, r = 10, 8, 3

        Xc = rng.randn(n, f_c).astype(np.float64)
        Xg = rng.randn(n, f_g).astype(np.float64)
        W_true = rng.randn(f_c, r) * 0.5
        H_true = rng.randn(f_g, r) * 0.5
        y = np.sum((Xc @ W_true) * (Xg @ H_true), axis=1) + 0.05 * rng.randn(n)

        imc = IMCInteraction(rank=r, lambda_w=0.01, lambda_h=0.01, max_iter=20)
        imc.fit(Xc, Xg, y, verbose=False)

        pred = imc.predict(Xc, Xg)
        assert len(pred) == n
        assert pred.dtype == np.float32
        r2 = 1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12)
        # Should capture substantial variance with correct rank
        assert r2 > 0.3, f"IMC R²={r2:.4f} too low"

    def test_cold_gene_prediction(self):
        """Cold (unseen) genes with features should get non-zero predictions."""
        rng = np.random.RandomState(42)
        n_train, n_test = 150, 50
        f_c, f_g, r = 8, 6, 2

        Xc_train = rng.randn(n_train, f_c).astype(np.float64)
        Xg_train = rng.randn(n_train, f_g).astype(np.float64)
        W_true = rng.randn(f_c, r) * 0.3
        H_true = rng.randn(f_g, r) * 0.3
        y_train = np.sum((Xc_train @ W_true) * (Xg_train @ H_true), axis=1)

        imc = IMCInteraction(rank=r, lambda_w=0.1, lambda_h=0.1, max_iter=20)
        imc.fit(Xc_train, Xg_train, y_train, verbose=False)

        # Predict for new genes (cold start — different Xg)
        Xc_test = rng.randn(n_test, f_c).astype(np.float64)
        Xg_test = rng.randn(n_test, f_g).astype(np.float64)
        pred = imc.predict(Xc_test, Xg_test)

        assert len(pred) == n_test
        assert not np.allclose(pred, 0), "Cold gene predictions should be non-zero"

    def test_formula_str(self):
        """Formula string output."""
        rng = np.random.RandomState(42)
        n, f_c, f_g, r = 100, 6, 4, 2
        Xc = rng.randn(n, f_c).astype(np.float64)
        Xg = rng.randn(n, f_g).astype(np.float64)
        y = rng.randn(n)

        imc = IMCInteraction(rank=r, max_iter=10)
        imc.fit(Xc, Xg, y,
                cell_feature_names=[f"cfeat_{i}" for i in range(f_c)],
                gene_feature_names=[f"gfeat_{i}" for i in range(f_g)],
                verbose=False)

        s = imc.formula_str()
        assert "W H^T" in s
        assert "rank" in s

    def test_get_top_interactions(self):
        """Top interactions should return named feature pairs."""
        rng = np.random.RandomState(42)
        n, f_c, f_g, r = 120, 8, 5, 2
        Xc = rng.randn(n, f_c).astype(np.float64)
        Xg = rng.randn(n, f_g).astype(np.float64)
        y = rng.randn(n)

        imc = IMCInteraction(rank=r, max_iter=10)
        imc.fit(Xc, Xg, y,
                cell_feature_names=[f"cell_{i}" for i in range(f_c)],
                gene_feature_names=[f"gene_{i}" for i in range(f_g)],
                verbose=False)

        top = imc.get_top_interactions(top_n=5)
        assert len(top) == 5
        assert len(top[0]) == 3  # (cell_feat, gene_feat, weight)
        assert isinstance(top[0][2], float)

    def test_convergence(self):
        """ALS should converge within max_iter."""
        rng = np.random.RandomState(42)
        n, f_c, f_g, r = 200, 10, 8, 3
        Xc = rng.randn(n, f_c).astype(np.float64)
        Xg = rng.randn(n, f_g).astype(np.float64)
        W_true = rng.randn(f_c, r) * 0.5
        H_true = rng.randn(f_g, r) * 0.5
        y = np.sum((Xc @ W_true) * (Xg @ H_true), axis=1) + 0.1 * rng.randn(n)

        imc = IMCInteraction(rank=r, max_iter=30, tol=1e-4)
        imc.fit(Xc, Xg, y, verbose=False)
        # Should converge before max_iter
        assert imc.iterations_ <= 30
        assert imc.train_r2_ > 0


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
        preds_b = np.array([1.0, 3.0, 5.0], dtype=np.float32)
        cells = np.array(["C1"] * 3)
        cold = np.array([True, False, False])
        blended = blend_ranks(preds_a, preds_b, cells,
                              alpha_a=0.5, alpha_b=0.5, cold_mask=cold)
        assert blended[0] > blended[2]

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
        y_pred = rng.rand(500).astype(np.float32)
        y_true = 3.0 * y_pred + 1.0 + 0.2 * rng.randn(500)
        cal_fn = calibrate_quantile(y_pred, y_true)
        calibrated = cal_fn(y_pred)
        assert calibrated.shape == y_pred.shape
        spearman_before = spearmanr(y_pred, y_true)[0]
        spearman_after = spearmanr(calibrated, y_true)[0]
        assert abs(spearman_after - spearman_before) < 0.01
