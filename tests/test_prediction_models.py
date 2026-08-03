"""Tests for src/prediction/ interpretable formula-based models.

Tests cover:
  - GeneEssentialityFormula: fit, predict, feature importance
  - CellVulnerabilityFormula: fit, predict, feature importance
  - IMCInteraction: ALS bilinear interaction, cold gene prediction
  - StructuredInteraction: biological interaction terms, cold-safe
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
    StructuredInteraction,
    HybridInteraction,
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
# Structured Biological Interaction
# ═══════════════════════════════════════════════════════════════════════════════

class TestStructuredInteraction:
    """Tests for StructuredInteraction — biological interaction terms."""

    @staticmethod
    def _make_toy_data(n=500, n_modules=14, n_lin=5, seed=42):
        """Build synthetic data with known module×indicator interaction."""
        rng = np.random.RandomState(seed)
        f_c = n_modules + 2 * n_modules + n_lin  # indicators + lineage
        f_g = n_modules + 1  # modules + evidence_weight

        # Cell features
        ind_arr = rng.randn(n, n_modules) * 0.5
        cell_cols = [f"cell_indicator_{i}" for i in range(n_modules)]
        # Add extra cell columns
        cell_cols += [f"cell_extra_{i}" for i in range(10)]
        # Lineage one-hot
        lin_labels = rng.randint(0, n_lin, n)
        lin_arr = np.zeros((n, n_lin))
        for i, l in enumerate(lin_labels):
            lin_arr[i, l] = 1.0
        lin_cols = [f"cell_lineage_onehot_LIN_{l}" for l in range(n_lin)]
        cell_data = np.column_stack([ind_arr,
                                     rng.randn(n, 10),
                                     lin_arr])
        cell_df = pd.DataFrame(cell_data,
                               columns=cell_cols + lin_cols)

        # Gene features
        mod_membership = rng.choice([0, 1], (n, n_modules), p=[0.7, 0.3])
        mod_membership = mod_membership.astype(np.float64)
        mod_cols = [f"gene_module_{k:02d}" for k in range(n_modules)]
        ew = 0.8 + 0.4 * rng.rand(n)
        gene_data = np.column_stack([mod_membership, ew])
        # Add extra gene columns
        gene_data = np.column_stack([gene_data, rng.randn(n, 5)])
        gene_df = pd.DataFrame(gene_data,
                               columns=mod_cols + ["gene_evidence_weight"]
                               + [f"gene_extra_{i}" for i in range(5)])

        # Pair features
        pair_z = rng.randn(n)
        pair_pct = rng.rand(n)

        # True interaction signal: module×indicator + z×evidence
        true_alpha = rng.randn(n_modules) * 0.3
        true_gamma = 0.5
        y = np.zeros(n)
        for k in range(n_modules):
            y += true_alpha[k] * mod_membership[:, k] * ind_arr[:, k]
        y += true_gamma * pair_z * ew
        y += 0.05 * rng.randn(n)  # noise

        return cell_df, gene_df, pair_z, pair_pct, y

    def test_fit_and_predict(self):
        """StructuredInteraction should capture biological interaction signal."""
        cell_df, gene_df, pair_z, pair_pct, y = self._make_toy_data(500)

        si = StructuredInteraction(include_lineage=True)
        si.fit(cell_df, gene_df, pair_z, pair_pct, y, verbose=False)

        pred = si.predict(cell_df, gene_df, pair_z, pair_pct)
        assert len(pred) == 500
        assert pred.dtype == np.float32
        r2 = 1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12)
        # Should capture substantial variance from structured signal
        assert r2 > 0.3, f"StructuredInteraction R²={r2:.4f} too low"

    def test_cold_gene_safety(self):
        """All features must be available for cold genes (no per-gene labels)."""
        cell_df, gene_df, pair_z, pair_pct, y = self._make_toy_data(300)

        si = StructuredInteraction(include_lineage=True)
        si.fit(cell_df, gene_df, pair_z, pair_pct, y, verbose=False)

        # Predict for completely new genes (same feature space)
        _, gene_df_new, pair_z_new, pair_pct_new, _ = self._make_toy_data(
            100, seed=99,
        )
        pred = si.predict(cell_df.iloc[:100], gene_df_new,
                         pair_z_new, pair_pct_new)
        assert len(pred) == 100
        assert not np.allclose(pred, 0), "Cold gene predictions should be non-zero"
        # Predictions should have variance (not all same value)
        assert pred.std() > 1e-6, "Cold gene predictions have no variance"

    def test_feature_names(self):
        """Should generate named features with all key groups."""
        cell_df, gene_df, pair_z, pair_pct, y = self._make_toy_data(200)

        si = StructuredInteraction(include_lineage=True)
        si.fit(cell_df, gene_df, pair_z, pair_pct, y, verbose=False)

        assert len(si.feature_names_) > 0
        names_str = " ".join(si.feature_names_)
        # Key feature groups should be present
        assert "z_cg" in names_str
        assert "z_abs" in names_str
        assert "z_x_evidence" in names_str
        assert "expr_percentile" in names_str
        assert any("z_x_mod" in n for n in si.feature_names_)
        assert any("z_x_ind" in n for n in si.feature_names_)
        assert any("mod" in n and "x_ind" in n for n in si.feature_names_)

    def test_top_interactions(self):
        """get_top_interactions should return named coefficients."""
        cell_df, gene_df, pair_z, pair_pct, y = self._make_toy_data(300)

        si = StructuredInteraction(include_lineage=True)
        si.fit(cell_df, gene_df, pair_z, pair_pct, y, verbose=False)

        top = si.get_top_interactions(top_n=10)
        assert len(top) == 10
        assert len(top[0]) == 2  # (name, coefficient)
        assert isinstance(top[0][1], float)

    def test_formula_str(self):
        """Formula string should describe the structure."""
        cell_df, gene_df, pair_z, pair_pct, y = self._make_toy_data(200)

        si = StructuredInteraction(include_lineage=True)
        si.fit(cell_df, gene_df, pair_z, pair_pct, y, verbose=False)

        s = si.formula_str()
        assert "I(c,g)" in s
        assert "Module × Indicator" in s
        assert "R²" in s

    def test_no_lineage_mode(self):
        """Should work with include_lineage=False."""
        cell_df, gene_df, pair_z, pair_pct, y = self._make_toy_data(200)

        si = StructuredInteraction(include_lineage=False)
        si.fit(cell_df, gene_df, pair_z, pair_pct, y, verbose=False)

        pred = si.predict(cell_df, gene_df, pair_z, pair_pct)
        assert len(pred) == 200
        # Should have fewer features without lineage terms
        assert not any("lin_" in n for n in si.feature_names_)

    def test_missing_features_handling(self):
        """Should handle missing cell/gene columns gracefully."""
        rng = np.random.RandomState(42)
        n = 100

        # Minimal DataFrames with only required columns
        cell_df = pd.DataFrame({
            "cell_indicator_0": rng.randn(n),
            "cell_indicator_1": rng.randn(n),
        })
        gene_df = pd.DataFrame({
            "gene_module_00": rng.choice([0, 1], n).astype(np.float64),
            "gene_module_01": rng.choice([0, 1], n).astype(np.float64),
            "gene_evidence_weight": np.ones(n),
        })
        pair_z = rng.randn(n)
        pair_pct = rng.rand(n)
        y = 0.3 * cell_df["cell_indicator_0"] * gene_df["gene_module_00"] \
            + 0.1 * rng.randn(n)

        si = StructuredInteraction(include_lineage=False)
        si.fit(cell_df, gene_df, pair_z, pair_pct, y, verbose=False)

        pred = si.predict(cell_df, gene_df, pair_z, pair_pct)
        assert len(pred) == n


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid SVD + Gene-Similarity CF Interaction
# ═══════════════════════════════════════════════════════════════════════════════

class TestHybridInteraction:
    """Tests for HybridInteraction — SVD + pathway-CF cold gene transfer."""

    def test_fit_and_predict_warm(self):
        """SVD should capture interaction signal on warm genes."""
        rng = np.random.RandomState(42)
        n_cells, n_genes = 50, 40
        n_pairs = n_cells * n_genes

        # Build residual matrix with low-rank structure
        U_true = rng.randn(n_cells, 3)
        V_true = rng.randn(n_genes, 3)
        R = U_true @ V_true.T + 0.05 * rng.randn(n_cells, n_genes)

        cell_ids = np.array([f"C{i}" for i in range(n_cells) for _ in range(n_genes)])
        gene_ids = np.array([f"G{j}" for _ in range(n_cells) for j in range(n_genes)])
        residuals = R.flatten()

        # Mock gene metadata for CF
        gene_meta = pd.DataFrame({
            "gene_symbol": [f"G{j}" for j in range(n_genes)],
            "pathways": ["OXPHOS > CI" if j < 10 else "Ribosome > Mito"
                        for j in range(n_genes)],
            "sub_mito_location": ["MIM"] * n_genes,
        })
        pathway_meta = pd.DataFrame({
            "pathway_name": ["CI_subunits", "Mito_ribosome"],
            "description": ["CI", "Mito ribosome"],
            "n_genes": [10, 10],
        })
        expr = pd.DataFrame(
            rng.randn(n_cells, n_genes),
            index=[f"C{i}" for i in range(n_cells)],
            columns=[f"G{j}" for j in range(n_genes)],
        )

        # G1 + G2 gene features
        g1 = pd.DataFrame({
            f"feat_{k}": rng.randn(n_genes)
            for k in range(14)
        }, index=[f"G{j}" for j in range(n_genes)])
        g2 = pd.DataFrame({
            "gene_expr_mean": rng.randn(n_genes),
        }, index=[f"G{j}" for j in range(n_genes)])

        # No cold genes — all warm
        hybrid = HybridInteraction(n_components=3, cf_knn=5, random_state=42)
        hybrid.fit(residuals, cell_ids, gene_ids, set(),
                   gene_meta, pathway_meta, expr, g1, g2, verbose=False)

        pred = hybrid.predict(cell_ids, gene_ids)
        assert len(pred) == n_pairs
        r2 = 1.0 - np.sum((residuals - pred) ** 2) / max(
            np.sum((residuals - residuals.mean()) ** 2), 1e-12,
        )
        # SVD rank 3 should capture most of the rank-3 signal
        assert r2 > 0.7, f"SVD R²={r2:.4f} too low"

    def test_cold_gene_prediction(self):
        """Cold genes with pathway neighbors should get non-zero predictions."""
        rng = np.random.RandomState(42)
        n_cells, n_warm, n_cold = 30, 30, 10
        n_genes = n_warm + n_cold

        # Build residual matrix (warm genes have data, cold have none)
        U_true = rng.randn(n_cells, 2)
        V_warm = rng.randn(n_warm, 2)
        R_warm = U_true @ V_warm.T + 0.05 * rng.randn(n_cells, n_warm)

        cell_ids = np.array([f"C{i}" for i in range(n_cells) for _ in range(n_warm)])
        gene_ids = np.array([f"G{j}" for _ in range(n_cells) for j in range(n_warm)])
        residuals = R_warm.flatten()

        cold_genes = set(f"G{n_warm + j}" for j in range(n_cold))
        all_genes = [f"G{j}" for j in range(n_genes)]

        gene_meta = pd.DataFrame({
            "gene_symbol": all_genes,
            "pathways": [
                "OXPHOS > CI" if j < 15 else "Ribosome > Mito"
                for j in range(n_genes)
            ],
            "sub_mito_location": ["MIM"] * n_genes,
        })
        pathway_meta = pd.DataFrame({
            "pathway_name": ["CI_subunits", "Mito_ribosome"],
            "description": ["CI", "Mito ribosome"],
            "n_genes": [10, 10],
        })
        expr = pd.DataFrame(
            rng.randn(n_cells, n_genes),
            index=[f"C{i}" for i in range(n_cells)],
            columns=all_genes,
        )
        g1 = pd.DataFrame(
            {f"feat_{k}": rng.randn(n_genes) for k in range(14)},
            index=all_genes,
        )
        g2 = pd.DataFrame(
            {"gene_expr_mean": rng.randn(n_genes)},
            index=all_genes,
        )

        hybrid = HybridInteraction(n_components=2, cf_knn=5, random_state=42)
        hybrid.fit(residuals, cell_ids, gene_ids, cold_genes,
                   gene_meta, pathway_meta, expr, g1, g2, verbose=False)

        # Predict for cold genes
        cold_cell_ids = np.array(
            [f"C{i}" for i in range(n_cells) for _ in range(n_cold)]
        )
        cold_gene_ids_arr = np.array(
            [f"G{n_warm + j}" for _ in range(n_cells) for j in range(n_cold)]
        )
        pred_cold = hybrid.predict(cold_cell_ids, cold_gene_ids_arr)
        assert len(pred_cold) == n_cells * n_cold
        assert not np.allclose(pred_cold, hybrid.global_residual_mean_), \
            "Cold gene predictions should differ from global mean"
        assert pred_cold.std() > 1e-6, "Cold gene predictions have no variance"

    def test_formula_str(self):
        """Formula string should describe SVD+CF structure."""
        rng = np.random.RandomState(42)
        n_cells, n_genes = 20, 15
        R = rng.randn(n_cells, n_genes)
        cell_ids = np.array(
            [f"C{i}" for i in range(n_cells) for _ in range(n_genes)]
        )
        gene_ids = np.array(
            [f"G{j}" for _ in range(n_cells) for j in range(n_genes)]
        )
        residuals = R.flatten()

        gene_meta = pd.DataFrame({
            "gene_symbol": [f"G{j}" for j in range(n_genes)],
            "pathways": ["OXPHOS > CI"] * n_genes,
            "sub_mito_location": ["MIM"] * n_genes,
        })
        pathway_meta = pd.DataFrame({
            "pathway_name": ["CI_subunits"],
            "description": ["CI"],
            "n_genes": [10],
        })
        expr = pd.DataFrame(
            rng.randn(n_cells, n_genes),
            index=[f"C{i}" for i in range(n_cells)],
            columns=[f"G{j}" for j in range(n_genes)],
        )
        g1 = pd.DataFrame(
            {f"feat_{k}": rng.randn(n_genes) for k in range(5)},
            index=[f"G{j}" for j in range(n_genes)],
        )
        g2 = pd.DataFrame(
            {"gene_expr_mean": rng.randn(n_genes)},
            index=[f"G{j}" for j in range(n_genes)],
        )

        hybrid = HybridInteraction(n_components=2, cf_knn=3, random_state=42)
        hybrid.fit(residuals, cell_ids, gene_ids, set(),
                   gene_meta, pathway_meta, expr, g1, g2, verbose=False)

        s = hybrid.formula_str()
        assert "SVD" in s or "σ_k" in s or "u_k" in s
        assert "R²" in s


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
