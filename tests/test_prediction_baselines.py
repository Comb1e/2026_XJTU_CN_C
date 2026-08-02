"""Tests for src/prediction/baselines.py."""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import pytest

from src.prediction.baselines import (
    shrink_gene_means,
    shrink_cell_means,
    train_gene_baseline_teacher,
    predict_gene_baselines,
    train_cell_bias_imputer,
    impute_cell_biases,
    compute_label_svd,
    build_module_priors,
    build_collaborative_features,
)
from src.prediction.features import (
    build_gene_static_features,
    build_gene_expression_profile_features,
    build_cell_features,
)
from src.preprocess import build_gene_module_map, compute_evidence_weights, load_all_data
from src.utils import load_config


@pytest.fixture(scope="module")
def test_data():
    config = load_config()
    return load_all_data(config)


@pytest.fixture(scope="module")
def config():
    return load_config()


class TestShrinkage:
    def test_shrink_gene_means_values(self):
        labels = pd.DataFrame({
            "perturbation_gene": ["G1", "G1", "G2", "G3"],
            "cell_line_id": ["C1", "C2", "C1", "C1"],
            "label": [1.0, 3.0, 0.0, 0.5],
        })
        shrunk = shrink_gene_means(labels, global_mean=1.0, prior_weight=10.0)
        # G1: n=2, mean=2.0 → (2*2 + 10*1)/(2+10) = 14/12 = 1.167
        assert abs(shrunk["G1"] - 14.0 / 12.0) < 1e-8
        # G2: n=1, mean=0 → (1*0 + 10*1)/(1+10) = 10/11 ≈ 0.909
        assert abs(shrunk["G2"] - 10.0 / 11.0) < 1e-8
        # G3: n=1, mean=0.5 → (1*0.5 + 10*1)/(1+10) = 10.5/11 ≈ 0.955
        assert abs(shrunk["G3"] - 10.5 / 11.0) < 1e-8

    def test_shrink_cell_means_values(self):
        labels = pd.DataFrame({
            "cell_line_id": ["C1", "C1", "C2"],
            "perturbation_gene": ["G1", "G2", "G1"],
            "label": [0.0, 2.0, 0.0],
        })
        shrunk = shrink_cell_means(labels, global_mean=0.5, prior_weight=5.0)
        # C1: n=2, mean=1.0 → (2*1 + 5*0.5)/(2+5) = 4.5/7 ≈ 0.643
        assert abs(shrunk["C1"] - 4.5 / 7.0) < 1e-8


class TestGeneBaselineTeacher:
    def test_train_and_predict(self, test_data):
        labels = pd.read_csv(
            Path(project_root, "数据文件/labels/gene_dependency.csv"),
            nrows=10000,
        )
        g1 = build_gene_static_features(
            test_data["gene_meta"], test_data["gene_module_map"],
            test_data["evidence_weights"],
        )
        g2 = build_gene_expression_profile_features(test_data["expression"])

        teacher, oof_preds, gene_feats = train_gene_baseline_teacher(
            g1, g2, labels, n_folds=3,
        )

        # Teacher should predict finite values
        assert len(oof_preds) > 0
        for g, v in oof_preds.items():
            assert np.isfinite(v)

        # Predict cold genes
        train_genes = set(labels["perturbation_gene"].unique())
        all_genes = set(gene_feats.index)
        cold = sorted(all_genes - train_genes)[:10]
        if cold:
            cold_baselines = predict_gene_baselines(teacher, gene_feats, cold)
            assert len(cold_baselines) == len(cold)
            for g, v in cold_baselines.items():
                assert np.isfinite(v)


class TestCellBiasImputer:
    def test_train_and_impute(self, test_data, config):
        labels = pd.read_csv(
            Path(project_root, "数据文件/labels/gene_dependency.csv"),
            nrows=10000,
        )
        outputs_dir = Path(config["paths"]["output_dir"])
        train_cells = sorted(labels["cell_line_id"].unique())
        g3 = build_cell_features(outputs_dir, train_cells)

        imputer, cell_shrunk = train_cell_bias_imputer(g3, labels)
        assert len(cell_shrunk) > 0

        # Impute for a new cell
        all_cells = test_data["expression"].index.tolist()
        new_cells = sorted(set(all_cells) - set(train_cells))[:5]
        if new_cells:
            g3_new = build_cell_features(outputs_dir, new_cells)
            imputed = impute_cell_biases(imputer, g3_new, new_cells)
            assert len(imputed) == len(new_cells)
            for v in imputed.values():
                assert np.isfinite(v)


class TestLabelSVD:
    def test_svd_output_shape(self, test_data):
        labels = pd.read_csv(
            Path(project_root, "数据文件/labels/gene_dependency.csv"),
            nrows=20000,
        )
        U, V, cell_idx, gene_idx, gm = compute_label_svd(labels, k=5)
        assert U.shape == (len(cell_idx), 5)
        assert V.shape == (len(gene_idx), 5)
        assert gm == float(labels["label"].mean())


class TestModulePriors:
    def test_all_modules_covered(self):
        priors = build_module_priors()
        for k in range(14):
            assert k in priors

    def test_ribosome_essential_higher_than_signaling(self):
        priors = build_module_priors()
        assert priors[6] > priors[13]  # MITO_RIBOSOME > SIGNALING


class TestCollaborativeFeatures:
    def test_output_shape(self):
        pairs = pd.DataFrame({
            "cell_line_id": ["C1", "C2"],
            "perturbation_gene": ["G1", "G2"],
        })
        gene_bl = {"G1": 0.5, "G2": -0.1}
        cell_bl = {"C1": 0.2, "C2": 0.3}
        g5 = build_collaborative_features(pairs, gene_bl, cell_bl)
        assert len(g5) == 2
        assert "g5_gene_baseline" in g5.columns
        assert "g5_cell_bias" in g5.columns

    def test_no_nan(self):
        pairs = pd.DataFrame({
            "cell_line_id": ["C1", "C2"],
            "perturbation_gene": ["G1", "G2"],
        })
        gene_bl = {"G1": 0.5, "G2": -0.1}
        cell_bl = {"C1": 0.2, "C2": 0.3}
        g5 = build_collaborative_features(pairs, gene_bl, cell_bl)
        assert not g5.isna().any().any()
