"""Tests for src/prediction/ interpretable formula-based models.

Tests cover:
  - GeneEssentialityFormula: fit, predict, feature importance
  - CellVulnerabilityFormula: fit, predict, feature importance
  - ModuleInteractionFormula: fit, predict, named coefficients
  - ExpressionEffectFormula: fit, predict, asymmetric effect
  - ModuleMatchFormula: fit, predict
  - EvidenceWeightedFormula: fit, predict, single coefficient
  - InteractionBlend: optimal component weighting
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
    ModuleInteractionFormula,
    ExpressionEffectFormula,
    ModuleMatchFormula,
    EvidenceWeightedFormula,
    InteractionBlend,
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
# Module×Indicator Interaction Formula
# ═══════════════════════════════════════════════════════════════════════════════

class TestModuleInteractionFormula:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n = 200
        n_modules = 14
        module_membership = np.zeros((n, n_modules), dtype=np.float32)
        for i in range(n):
            module_membership[i, i % n_modules] = 1.0
        cell_indicators = rng.randn(n, n_modules).astype(np.float32)
        # Generate labels with known module interaction
        true_coef = np.array([0.5 if m == 0 else -0.2 if m == 3 else 0.1 if m == 6 else 0.0
                             for m in range(n_modules)])
        residuals = (module_membership * cell_indicators) @ true_coef + 0.1 * rng.randn(n)

        formula = ModuleInteractionFormula(alpha=1.0, n_modules=n_modules)
        formula.fit(
            np.array([f"C{i}" for i in range(n)]),
            np.array([f"G{i}" for i in range(n)]),
            residuals,
            module_membership,
            cell_indicators,
        )

        preds = formula.predict(module_membership, cell_indicators)
        assert len(preds) == n
        assert preds.dtype == np.float32

    def test_named_coefficients(self):
        rng = np.random.RandomState(42)
        n = 100
        n_modules = 14
        module_membership = np.zeros((n, n_modules), dtype=np.float32)
        for i in range(n):
            module_membership[i, i % n_modules] = 1.0
        cell_indicators = rng.randn(n, n_modules).astype(np.float32)
        residuals = rng.randn(n)

        formula = ModuleInteractionFormula(alpha=1.0, n_modules=n_modules)
        formula.fit(
            np.array([f"C{i}" for i in range(n)]),
            np.array([f"G{i}" for i in range(n)]),
            residuals,
            module_membership,
            cell_indicators,
        )

        coefs = formula.get_coefficients()
        assert len(coefs) == n_modules
        assert all(isinstance(name, str) for name, _ in coefs)
        # Check that all 14 module names are present
        for name in ModuleInteractionFormula.MODULE_NAMES:
            assert any(c[0] == name for c in coefs)

    def test_formula_str(self):
        rng = np.random.RandomState(42)
        n = 50
        n_modules = 14
        module_membership = np.zeros((n, n_modules), dtype=np.float32)
        for i in range(n):
            module_membership[i, i % n_modules] = 1.0
        cell_indicators = rng.randn(n, n_modules).astype(np.float32)
        residuals = rng.randn(n)

        formula = ModuleInteractionFormula(alpha=1.0, n_modules=n_modules)
        formula.fit(
            np.array([f"C{i}" for i in range(n)]),
            np.array([f"G{i}" for i in range(n)]),
            residuals, module_membership, cell_indicators,
        )
        s = formula.formula_str()
        assert "I_mod" in s


# ═══════════════════════════════════════════════════════════════════════════════
# Expression Effect Formula
# ═══════════════════════════════════════════════════════════════════════════════

class TestExpressionEffectFormula:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n = 300
        z_cg = rng.randn(n).astype(np.float32)
        # Nonlinear asymmetric effect
        residuals = -0.5 * z_cg + 0.3 * np.abs(z_cg) - 0.2 * np.maximum(0, z_cg) + 0.1 * rng.randn(n)

        formula = ExpressionEffectFormula(alpha=1.0)
        formula.fit(z_cg, residuals)

        preds = formula.predict(z_cg)
        assert len(preds) == n
        assert preds.dtype == np.float32
        assert not np.isnan(preds).any()

    def test_coefficients_shape(self):
        rng = np.random.RandomState(42)
        n = 200
        z_cg = rng.randn(n).astype(np.float32)
        residuals = rng.randn(n)

        formula = ExpressionEffectFormula(alpha=1.0)
        formula.fit(z_cg, residuals)
        assert formula.coefficients_ is not None
        assert len(formula.coefficients_) == 4  # θ₁, θ₂, θ₃, θ₄

    def test_formula_str(self):
        rng = np.random.RandomState(42)
        n = 100
        z_cg = rng.randn(n).astype(np.float32)
        residuals = rng.randn(n)

        formula = ExpressionEffectFormula(alpha=1.0)
        formula.fit(z_cg, residuals)
        s = formula.formula_str()
        assert "I_expr" in s
        assert "|z|" in s


# ═══════════════════════════════════════════════════════════════════════════════
# Module Match Formula
# ═══════════════════════════════════════════════════════════════════════════════

class TestModuleMatchFormula:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n = 200
        n_modules = 14
        module_membership = np.zeros((n, n_modules), dtype=np.float32)
        for i in range(n):
            module_membership[i, i % n_modules] = 1.0
        expr_percentile = rng.rand(n).astype(np.float32)
        residuals = rng.randn(n)

        formula = ModuleMatchFormula(alpha=1.0, n_modules=n_modules)
        formula.fit(module_membership, expr_percentile, residuals)

        preds = formula.predict(module_membership, expr_percentile)
        assert len(preds) == n
        assert preds.dtype == np.float32

    def test_formula_str(self):
        rng = np.random.RandomState(42)
        n = 50
        n_modules = 14
        module_membership = np.zeros((n, n_modules), dtype=np.float32)
        for i in range(n):
            module_membership[i, i % n_modules] = 1.0
        expr_percentile = rng.rand(n).astype(np.float32)
        residuals = rng.randn(n)

        formula = ModuleMatchFormula(alpha=1.0, n_modules=n_modules)
        formula.fit(module_membership, expr_percentile, residuals)
        s = formula.formula_str()
        assert "I_match" in s


# ═══════════════════════════════════════════════════════════════════════════════
# Evidence-Weighted Expression Formula
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvidenceWeightedFormula:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n = 300
        evidence = rng.rand(n).astype(np.float32) * 0.5 + 0.5
        z_cg = rng.randn(n).astype(np.float32)
        true_omega = 0.7
        residuals = true_omega * evidence * z_cg + 0.1 * rng.randn(n)

        formula = EvidenceWeightedFormula()
        formula.fit(evidence, z_cg, residuals)

        preds = formula.predict(evidence, z_cg)
        assert len(preds) == n
        assert preds.dtype == np.float32
        # Should recover approximate omega
        assert abs(formula.omega_ - true_omega) < 0.5

    def test_single_coefficient(self):
        rng = np.random.RandomState(42)
        n = 200
        evidence = rng.rand(n).astype(np.float32)
        z_cg = rng.randn(n).astype(np.float32)
        residuals = rng.randn(n)

        formula = EvidenceWeightedFormula()
        formula.fit(evidence, z_cg, residuals)
        assert isinstance(formula.omega_, float)

    def test_formula_str(self):
        rng = np.random.RandomState(42)
        n = 100
        evidence = rng.rand(n).astype(np.float32)
        z_cg = rng.randn(n).astype(np.float32)
        residuals = rng.randn(n)

        formula = EvidenceWeightedFormula()
        formula.fit(evidence, z_cg, residuals)
        s = formula.formula_str()
        assert "I_ew" in s
        assert "EvidenceWeight" in s


# ═══════════════════════════════════════════════════════════════════════════════
# Interaction Blend
# ═══════════════════════════════════════════════════════════════════════════════

class TestInteractionBlend:
    def test_fit_and_predict(self):
        rng = np.random.RandomState(42)
        n = 200
        comp1 = rng.randn(n).astype(np.float32)
        comp2 = rng.randn(n).astype(np.float32)
        y = 2.0 * comp1 + 0.0 * comp2 + 0.1 * rng.randn(n)

        blend = InteractionBlend(alphas=[0.1, 1.0, 10.0])
        blend.fit([comp1, comp2], y, component_names=["good", "noisy"])

        preds = blend.predict([comp1, comp2])
        assert len(preds) == n

        weights = dict(blend.get_weights())
        assert abs(weights["good"]) > abs(weights["noisy"])

    def test_weights_summary(self):
        rng = np.random.RandomState(42)
        n = 100
        comps = [rng.randn(n).astype(np.float32) for _ in range(3)]
        y = comps[0] + 0.5 * comps[1] + 0.1 * rng.randn(n)

        blend = InteractionBlend()
        blend.fit(comps, y, component_names=["a", "b", "c"])
        weights = blend.get_weights()
        assert len(weights) == 3


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
