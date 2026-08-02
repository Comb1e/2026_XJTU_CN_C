"""Tests for src/prediction/features.py."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.prediction.features import (
    build_gene_static_features,
    build_gene_expression_profile_features,
    build_cell_features,
    build_lineage_onehot,
    build_pair_features,
    assemble_feature_table,
)
from src.preprocess import build_gene_module_map, compute_evidence_weights, load_all_data
from src.scoring.ewm import compute_ewm_knockout_delta
from src.utils import load_config


@pytest.fixture(scope="module")
def test_data():
    config = load_config()
    return load_all_data(config)


@pytest.fixture(scope="module")
def config():
    return load_config()


class TestGeneStaticFeatures:
    def test_output_shape(self, test_data):
        g1 = build_gene_static_features(
            test_data["gene_meta"],
            test_data["gene_module_map"],
            test_data["evidence_weights"],
        )
        assert len(g1) == 1123
        assert g1.index.name == "gene_symbol"

    def test_module_membership_columns(self, test_data):
        g1 = build_gene_static_features(
            test_data["gene_meta"],
            test_data["gene_module_map"],
            test_data["evidence_weights"],
        )
        for k in range(14):
            assert f"gene_module_{k:02d}" in g1.columns

    def test_no_nan(self, test_data):
        g1 = build_gene_static_features(
            test_data["gene_meta"],
            test_data["gene_module_map"],
            test_data["evidence_weights"],
        )
        assert not g1.isna().any().any()

    def test_evidence_weight_in_data(self, test_data):
        g1 = build_gene_static_features(
            test_data["gene_meta"],
            test_data["gene_module_map"],
            test_data["evidence_weights"],
        )
        assert "gene_evidence_weight" in g1.columns
        assert g1["gene_evidence_weight"].min() >= 0.5


class TestGeneExpressionProfileFeatures:
    def test_output_shape(self, test_data):
        expression = test_data["expression"]
        g2 = build_gene_expression_profile_features(expression)
        assert len(g2) == 1123
        # Should have at least mean, std, min, max, quartiles
        assert "gene_expr_mean" in g2.columns
        assert "gene_expr_std" in g2.columns

    def test_no_nan(self, test_data):
        g2 = build_gene_expression_profile_features(test_data["expression"])
        assert not g2.isna().any().any()


class TestCellFeatures:
    def test_output_shape(self, config):
        outputs_dir = Path(config["paths"]["output_dir"])
        cells = pd.read_csv(
            Path(config["paths"]["features_dir"]) / "cell_expression_zscore.csv",
            index_col=0,
        ).index[:50].tolist()
        g3 = build_cell_features(outputs_dir, cells)
        assert len(g3) == 50

    def test_no_nan(self, config):
        outputs_dir = Path(config["paths"]["output_dir"])
        cells = pd.read_csv(
            Path(config["paths"]["features_dir"]) / "cell_expression_zscore.csv",
            index_col=0,
        ).index[:50].tolist()
        g3 = build_cell_features(outputs_dir, cells)
        assert not g3.isna().any().any()


class TestLineageOnehot:
    def test_output_shape(self, test_data):
        cell_meta = test_data["cell_meta"]
        cells = cell_meta["cell_line_id"].head(20).tolist()
        onehot = build_lineage_onehot(cell_meta, cells)
        assert len(onehot) == 20

    def test_each_row_sums_to_one(self, test_data):
        cell_meta = test_data["cell_meta"]
        cells = cell_meta["cell_line_id"].head(20).tolist()
        onehot = build_lineage_onehot(cell_meta, cells)
        row_sums = onehot.sum(axis=1)
        assert (row_sums == 1.0).all()


class TestPairFeatures:
    @pytest.fixture(scope="module")
    def pair_data(self, test_data, config):
        expression = test_data["expression"]
        gmm = test_data["gene_module_map"]
        ew = test_data["evidence_weights"]

        outputs_dir = Path(config["paths"]["output_dir"])
        ewm_scores = pd.read_csv(outputs_dir / "ewm_scores.csv", index_col=0)

        pairs = pd.DataFrame({
            "cell_line_id": [expression.index[0], expression.index[1], expression.index[2]],
            "perturbation_gene": ["CYC1", "SDHA", "COX5A"],
        })
        return {
            "pairs": pairs,
            "expression": expression,
            "gmm": gmm,
            "ew": ew,
            "ewm_scores": ewm_scores,
        }

    def test_output_shape(self, pair_data):
        g4 = build_pair_features(
            pair_data["pairs"],
            pair_data["expression"],
            pair_data["gmm"],
            pair_data["ew"],
            pair_data["ewm_scores"],
        )
        assert len(g4) == 3

    def test_contains_z_cg(self, pair_data):
        g4 = build_pair_features(
            pair_data["pairs"],
            pair_data["expression"],
            pair_data["gmm"],
            pair_data["ew"],
            pair_data["ewm_scores"],
        )
        assert "pair_z_cg" in g4.columns

    def test_no_nan(self, pair_data):
        g4 = build_pair_features(
            pair_data["pairs"],
            pair_data["expression"],
            pair_data["gmm"],
            pair_data["ew"],
            pair_data["ewm_scores"],
        )
        assert not g4.isna().any().any()

    def test_delta_ewm_parity_with_original(self, pair_data):
        """Vectorized Δ_EWM must match original per-pair compute_ewm_knockout_delta."""
        g4 = build_pair_features(
            pair_data["pairs"],
            pair_data["expression"],
            pair_data["gmm"],
            pair_data["ew"],
            pair_data["ewm_scores"],
            n_modules=14,
        )
        for i, (_, row) in enumerate(pair_data["pairs"].iterrows()):
            cell = row["cell_line_id"]
            gene = row["perturbation_gene"]
            delta_orig = compute_ewm_knockout_delta(
                pair_data["expression"], pair_data["gmm"], pair_data["ew"],
                gene, cell, n_modules=14,
            )
            for k in range(14):
                col = f"pair_delta_ewm_{k:02d}"
                assert abs(g4.loc[i, col] - delta_orig[k]) < 1e-8, \
                    f"Mismatch for ({cell}, {gene}) module {k}: {g4.loc[i, col]} vs {delta_orig[k]}"

    def test_gene_not_in_module_delta_is_zero(self, pair_data):
        """Δ should be zero for modules the gene doesn't belong to."""
        g4 = build_pair_features(
            pair_data["pairs"],
            pair_data["expression"],
            pair_data["gmm"],
            pair_data["ew"],
            pair_data["ewm_scores"],
            n_modules=14,
        )
        # CYC1 is in OXPHOS_CII_CIII (module 1). It should NOT be in OXPHOS_CI (module 0)
        cyc1_mods = pair_data["gmm"]["CYC1"]["modules"]
        for k in range(14):
            if k not in cyc1_mods:
                assert abs(g4.loc[0, f"pair_delta_ewm_{k:02d}"]) < 1e-10

    def test_ko_direction(self, pair_data):
        """Knocking out a highly expressed gene should decrease module scores."""
        g4 = build_pair_features(
            pair_data["pairs"],
            pair_data["expression"],
            pair_data["gmm"],
            pair_data["ew"],
            pair_data["ewm_scores"],
            n_modules=14,
        )
        # CYC1: if z_cg > 0, delta should be negative for its modules
        z_cyc1 = pair_data["expression"].loc[pair_data["expression"].index[0], "CYC1"]
        if z_cyc1 > 0.5:
            for k in pair_data["gmm"]["CYC1"]["modules"]:
                assert g4.loc[0, f"pair_delta_ewm_{k:02d}"] < 0


class TestAssembleFeatureTable:
    def test_no_nan_after_assembly(self, test_data, config):
        """Assembly must produce NaN-free output."""
        expression = test_data["expression"]

        pairs = pd.DataFrame({
            "cell_line_id": [expression.index[0], expression.index[1]],
            "perturbation_gene": ["CYC1", "SDHA"],
        })

        g1 = build_gene_static_features(
            test_data["gene_meta"],
            test_data["gene_module_map"],
            test_data["evidence_weights"],
        )
        g2 = build_gene_expression_profile_features(expression)

        outputs_dir = Path(config["paths"]["output_dir"])
        cells = sorted(pairs["cell_line_id"].unique())
        g3 = build_cell_features(outputs_dir, cells)
        g3_onehot = build_lineage_onehot(test_data["cell_meta"], cells)

        ewm_scores = pd.read_csv(outputs_dir / "ewm_scores.csv", index_col=0)
        g4 = build_pair_features(
            pairs, expression, test_data["gene_module_map"],
            test_data["evidence_weights"], ewm_scores,
        )

        X = assemble_feature_table(pairs, g1, g2, g3, g3_onehot, g4)
        assert not X.isna().any().any()
