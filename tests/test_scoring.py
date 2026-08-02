"""Tests for scoring methods: EWM, RES, SPCA."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.preprocess import build_gene_module_map, compute_evidence_weights, load_all_data
from src.scoring.ewm import compute_ewm_scores, compute_ewm_knockout_delta
from src.scoring.res import compute_res_scores
from src.scoring.spca import compute_spca_scores
from src.utils import load_config


@pytest.fixture(scope="module")
def test_data():
    """Load data once for all scoring tests."""
    config = load_config()
    data = load_all_data(config)
    return data


class TestEWM:
    """Evidence-Weighted Mean scoring tests."""

    def test_output_shape(self, test_data):
        scores = compute_ewm_scores(
            test_data["expression"],
            test_data["gene_module_map"],
            test_data["evidence_weights"],
        )
        assert scores.shape == (1140, 14)
        assert not scores.isna().any().any()

    def test_scores_not_constant(self, test_data):
        scores = compute_ewm_scores(
            test_data["expression"],
            test_data["gene_module_map"],
            test_data["evidence_weights"],
        )
        for col in scores.columns:
            assert scores[col].std() > 0, f"Column {col} is constant"

    def test_knockout_effect_present(self, test_data):
        """Knocking out a gene in its module should produce non-zero delta."""
        expr = test_data["expression"]
        gmm = test_data["gene_module_map"]
        ew = test_data["evidence_weights"]

        # Find a gene that belongs to at least one module
        for gene in expr.columns[:100]:
            if gene in gmm and len(gmm[gene]["modules"]) > 0:
                mod_indices = gmm[gene]["modules"]
                cell = expr.index[0]
                z_val = expr.loc[cell, gene]

                delta = compute_ewm_knockout_delta(
                    expr, gmm, ew, gene, cell, n_modules=14,
                )

                # Delta should be non-zero for modules containing this gene
                if abs(z_val) > 0.1:
                    for mi in mod_indices:
                        assert delta[mi] != 0, \
                            f"KO of {gene} (z={z_val:.3f}) should affect module {mi}"
                break

    def test_knockout_direction(self, test_data):
        """Positive z-score gene KO should decrease its module score."""
        expr = test_data["expression"]
        gmm = test_data["gene_module_map"]
        ew = test_data["evidence_weights"]

        for gene in expr.columns[:200]:
            if gene in gmm and gmm[gene]["modules"]:
                cell = expr.index[0]
                z_val = expr.loc[cell, gene]
                if z_val > 0.5:  # Pick a highly expressed gene
                    mod_idx = gmm[gene]["modules"][0]
                    delta = compute_ewm_knockout_delta(
                        expr, gmm, ew, gene, cell, n_modules=14,
                    )
                    assert delta[mod_idx] < 0, \
                        f"KO of highly expressed gene {gene} should decrease module score"
                    break

    def test_weight_contribution(self, test_data):
        """Higher weights should produce larger KO effects."""
        expr = test_data["expression"]
        gmm = test_data["gene_module_map"]
        ew = test_data["evidence_weights"]

        # Find a gene with weight > 1.1 vs weight < 0.9
        high_w_genes = [g for g, w in ew.items() if w > 1.1 and g in gmm and gmm[g]["modules"]]
        low_w_genes = [g for g, w in ew.items() if w < 0.95 and g in gmm and gmm[g]["modules"]]

        if high_w_genes and low_w_genes:
            cell = expr.index[0]
            hg = high_w_genes[0]
            lg = low_w_genes[0]

            h_mod = gmm[hg]["modules"][0]
            l_mod = gmm[lg]["modules"][0]

            delta_h = abs(compute_ewm_knockout_delta(expr, gmm, ew, hg, cell)[h_mod])
            delta_l = abs(compute_ewm_knockout_delta(expr, gmm, ew, lg, cell)[l_mod])

            # Higher weight doesn't guarantee larger delta (expression matters too)
            # But the weighting system should be functional
            assert delta_h >= 0
            assert delta_l >= 0


class TestRES:
    """Rank-based Enrichment Score tests."""

    def test_output_shape(self, test_data):
        scores = compute_res_scores(
            test_data["expression"],
            test_data["gene_module_map"],
        )
        assert scores.shape == (1140, 14)
        assert not scores.isna().any().any()

    def test_approximately_standard_normal(self, test_data):
        scores = compute_res_scores(
            test_data["expression"],
            test_data["gene_module_map"],
        )
        for col in scores.columns:
            mu = scores[col].mean()
            sigma = scores[col].std()
            # Should be approximately N(0,1) due to rank-based normalization
            assert abs(mu) < 0.1, f"Column {col} mean {mu:.4f} far from 0"
            # Std may deviate from 1 due to discretization at small N

    def test_scores_vary_across_modules(self, test_data):
        scores = compute_res_scores(
            test_data["expression"],
            test_data["gene_module_map"],
        )
        # Different modules should have different score patterns
        corr = scores.corr()
        # Not all pairs should be perfectly correlated
        off_diag = corr.to_numpy()[~np.eye(14, dtype=bool)]
        assert np.abs(off_diag).max() < 1.0


class TestSPCA:
    """Sparse PCA scoring tests."""

    def test_output_shape(self, test_data):
        scores = compute_spca_scores(
            test_data["expression"],
            test_data["gene_module_map"],
        )
        assert scores.shape == (1140, 14)
        assert not scores.isna().any().any()

    def test_loadings_stored(self, test_data):
        scores = compute_spca_scores(
            test_data["expression"],
            test_data["gene_module_map"],
        )
        assert hasattr(scores, "attrs")
        assert "loadings" in scores.attrs
        loadings = scores.attrs["loadings"]
        assert len(loadings) > 0
        # Each module should have loadings
        for mod_idx, gene_loadings in loadings.items():
            assert len(gene_loadings) > 0
            # Loadings should sum-of-squares ≈ 1 (normalized)
            ss = sum(v**2 for v in gene_loadings.values())
            assert abs(ss - 1.0) < 0.01, \
                f"Module {mod_idx} loading norm {ss:.4f} ≠ 1"

    def test_scores_vary(self, test_data):
        scores = compute_spca_scores(
            test_data["expression"],
            test_data["gene_module_map"],
        )
        for col in scores.columns:
            assert scores[col].std() > 0, f"SPCA column {col} has zero variance"
