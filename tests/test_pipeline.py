"""Integration tests for the full pipeline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.indicators import run_pipeline, export_outputs
from src.preprocess import load_all_data
from src.ensemble import fuse_ensemble
from src.orthogonalize import orthogonalize, lowdin_orthogonalize, partial_orthogonalize
from src.knockout import compute_knockout_response, compute_knockout_summary
from src.utils import load_config


@pytest.fixture(scope="module")
def pipeline_results():
    """Run pipeline once for all integration tests."""
    config = load_config()
    return run_pipeline(config)


class TestEnsemble:
    """Ensemble fusion tests."""

    def test_weights_sum_to_one(self, pipeline_results):
        ewm = pipeline_results["ewm_scores"]
        res = pipeline_results["res_scores"]
        spca = pipeline_results["spca_scores"]
        # Should not raise
        result = fuse_ensemble(ewm, res, spca, 0.5, 0.25, 0.25)
        assert result.shape == (1140, 14)

    def test_shape_mismatch_raises(self):
        a = pd.DataFrame(np.random.randn(10, 3))
        b = pd.DataFrame(np.random.randn(10, 4))
        c = pd.DataFrame(np.random.randn(10, 3))
        with pytest.raises(AssertionError):
            fuse_ensemble(a, b, c)


class TestOrthogonalization:
    """Orthogonalization tests."""

    def test_lowdin_cov_identity(self, pipeline_results):
        indicators = pipeline_results["indicators_raw"]
        orth, S = lowdin_orthogonalize(indicators)
        cov = orth.cov().to_numpy()
        # Off-diagonals should be near zero
        off_diag = cov[~np.eye(cov.shape[0], dtype=bool)]
        assert np.abs(off_diag).max() < 1e-10, \
            f"Max off-diagonal after Löwdin: {np.abs(off_diag).max():.2e}"

    def test_lowdin_diagonal_dominance(self, pipeline_results):
        indicators = pipeline_results["indicators_raw"]
        orth, S = lowdin_orthogonalize(indicators)
        diag = np.diag(S)
        assert (diag > 0).all(), "Transform matrix diagonal should be positive"
        assert diag.min() > 0.5, f"Min diagonal {diag.min():.3f} — poor meaning preservation"

    def test_none_method_returns_input(self, pipeline_results):
        indicators = pipeline_results["indicators_raw"]
        result, S = orthogonalize(indicators, method="none")
        pd.testing.assert_frame_equal(result, indicators)
        assert S is None

    def test_partial_reduces_correlation(self):
        # Create correlated data
        np.random.seed(42)
        x = np.random.randn(100)
        y = 0.9 * x + 0.1 * np.random.randn(100)
        z = np.random.randn(100)
        df = pd.DataFrame({"a": x, "b": y, "c": z})
        corr_before = df.corr().loc["a", "b"]
        assert abs(corr_before) > 0.7

        result = partial_orthogonalize(df, corr_threshold=0.7)
        corr_after = result.corr().loc["a", "b"]
        assert abs(corr_after) < abs(corr_before)


class TestKnockout:
    """Gene knockout response tests."""

    def test_knockout_response_shape(self, pipeline_results):
        expr = pipeline_results["ewm_scores"]  # Just need expression-like data
        # Use actual expression data
        config = load_config()
        data = load_all_data(config)
        expr = data["expression"]
        gmm = data["gene_module_map"]
        ew = data["evidence_weights"]

        delta = compute_knockout_response(
            expr, gmm, ew, "CYC1", expr.index[0],
            spca_loadings=pipeline_results.get("spca_loadings"),
        )
        assert delta.shape == (14,)
        assert not np.isnan(delta).any()

    def test_knockout_summary_has_expected_columns(self, pipeline_results):
        config = load_config()
        data = load_all_data(config)
        summary = compute_knockout_summary(
            data["expression"],
            data["gene_module_map"],
            data["evidence_weights"],
            sample_genes=["SDHA", "CYC1", "COX5A"],
            sample_cells=[data["expression"].index[0]],
            spca_loadings=pipeline_results.get("spca_loadings"),
        )
        assert "cell_line_id" in summary.columns
        assert "perturbation_gene" in summary.columns
        for k in range(14):
            assert f"delta_{k:02d}" in summary.columns

    def test_nonexistent_gene_raises(self, pipeline_results):
        config = load_config()
        data = load_all_data(config)
        with pytest.raises(ValueError):
            compute_knockout_response(
                data["expression"],
                data["gene_module_map"],
                data["evidence_weights"],
                "NONEXISTENT_GENE",
                data["expression"].index[0],
            )


class TestPipeline:
    """Full pipeline integration tests."""

    def test_indicators_shape(self, pipeline_results):
        indicators = pipeline_results["indicators"]
        assert indicators.shape == (1140, 14)

    def test_no_nan_inf(self, pipeline_results):
        for key in ["indicators", "indicators_raw", "ewm_scores", "res_scores", "spca_scores"]:
            df = pipeline_results[key]
            assert not df.isna().any().any(), f"{key} contains NaN"
            assert not np.isinf(df.to_numpy()).any(), f"{key} contains Inf"

    def test_indicators_have_variance(self, pipeline_results):
        indicators = pipeline_results["indicators"]
        for col in indicators.columns:
            assert indicators[col].std() > 0.01, \
                f"Indicator {col} has near-zero variance: {indicators[col].std():.6f}"

    def test_cell_line_count_matches_expression(self, pipeline_results):
        config = load_config()
        data = load_all_data(config)
        assert len(pipeline_results["indicators"]) == len(data["expression"])

    def test_deterministic(self, pipeline_results):
        """Pipeline should produce identical results on same input."""
        config = load_config()
        results2 = run_pipeline(config)
        pd.testing.assert_frame_equal(
            pipeline_results["indicators"],
            results2["indicators"],
        )

    def test_pathway_scores_output(self, pipeline_results):
        pw = pipeline_results.get("pathway_scores")
        if pw is not None:
            assert pw.shape == (1140, 140)
            assert not pw.isna().any().any()

    def test_gene_module_map_coverage(self, pipeline_results):
        gmm = pipeline_results["gene_module_map"]
        assert len(gmm) == 1123
        unmapped = sum(1 for info in gmm.values() if not info["modules"])
        assert unmapped == 0, f"{unmapped} genes have no module assignment"

    def test_transform_matrix_structure(self, pipeline_results):
        S = pipeline_results["transform_matrix"]
        if S is not None:
            assert S.shape == (14, 14)
            assert not np.isnan(S).any()
            # Symmetric
            assert np.allclose(S, S.T)

    def test_indicator_biological_signal(self, pipeline_results):
        """Indicators should vary by lineage (biological signal)."""
        config = load_config()
        data = load_all_data(config)
        indicators = pipeline_results["indicators"].copy()
        indicators["lineage"] = data["cell_meta"].set_index("cell_line_id").loc[
            indicators.index, "OncotreeLineage"
        ].values

        # At least some indicators should show lineage differences
        from scipy.stats import f_oneway
        lineages = indicators["lineage"].unique()
        if len(lineages) > 5:
            significant = 0
            for col in [c for c in indicators.columns if c != "lineage"]:
                groups = [indicators[indicators["lineage"] == l][col].dropna().values
                          for l in lineages[:10]]
                groups = [g for g in groups if len(g) > 5]
                if len(groups) >= 3:
                    try:
                        f_stat, p_val = f_oneway(*groups)
                        if p_val < 0.05:
                            significant += 1
                    except Exception:
                        pass
            assert significant >= 3, \
                f"Only {significant}/14 indicators show lineage differences (p<0.05)"
