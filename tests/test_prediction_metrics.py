"""Parity tests for src/prediction/metrics.py against official calculate_metric.py."""

import sys
from pathlib import Path

# Add project root and 数据文件 to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "数据文件"))

import numpy as np
import pandas as pd
import pytest

from src.prediction.metrics import (
    compute_all_metrics,
    metric_by_cell,
    pearson_or_zero,
)


class TestPearsonOrZero:
    def test_perfect_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        assert abs(pearson_or_zero(x, y) - 1.0) < 1e-10

    def test_zero_variance_returns_zero(self):
        x = np.array([3.0, 3.0, 3.0])
        y = np.array([1.0, 2.0, 3.0])
        assert pearson_or_zero(x, y) == 0.0

    def test_negative_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert abs(pearson_or_zero(x, y) - (-1.0)) < 1e-10


class TestMetricByCell:
    def test_single_cell_perfect_ranking(self):
        """Perfect prediction => perfect scores."""
        df = pd.DataFrame({
            "cell_line_id": ["C1"] * 5,
            "perturbation_gene": [f"G{i}" for i in range(5)],
            "prediction": [5.0, 4.0, 3.0, 2.0, 1.0],
            "truth": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        metrics = metric_by_cell(df)
        assert abs(metrics["spearman_score"] - 1.0) < 1e-10
        assert abs(metrics["precision_at_5"] - 1.0) < 1e-10
        assert abs(metrics["ndcg_at_5"] - 1.0) < 1e-10

    def test_single_cell_reversed_ranking(self):
        """Reversed prediction => Spearman = 0 (normalized)."""
        df = pd.DataFrame({
            "cell_line_id": ["C1"] * 5,
            "perturbation_gene": [f"G{i}" for i in range(5)],
            "prediction": [1.0, 2.0, 3.0, 4.0, 5.0],
            "truth": [5.0, 4.0, 3.0, 2.0, 1.0],
        })
        metrics = metric_by_cell(df)
        assert abs(metrics["spearman_score"]) < 0.01

    def test_multiple_cells_macro_average(self):
        """Two cells, both perfect => macro average = 1."""
        df = pd.DataFrame({
            "cell_line_id": ["C1"] * 3 + ["C2"] * 3,
            "perturbation_gene": ["G1", "G2", "G3", "G4", "G5", "G6"],
            "prediction": [3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
            "truth": [3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
        })
        metrics = metric_by_cell(df)
        assert abs(metrics["spearman_score"] - 1.0) < 1e-10

    def test_ties_handled_with_average_rank(self):
        """Tied predictions get average rank."""
        df = pd.DataFrame({
            "cell_line_id": ["C1"] * 3,
            "perturbation_gene": ["G1", "G2", "G3"],
            "prediction": [5.0, 5.0, 1.0],  # G1 and G2 tied
            "truth": [5.0, 3.0, 1.0],
        })
        metrics = metric_by_cell(df)
        # G3 is clearly last; G1 and G2 tied at top
        # With only 3 genes and ties, Spearman should be computable
        assert "spearman_score" in metrics
        assert metrics["spearman_score"] >= 0.0

    def test_precision_correct(self):
        """Precision@K checks overlap of top-K sets."""
        df = pd.DataFrame({
            "cell_line_id": ["C1"] * 5,
            "perturbation_gene": ["A", "B", "C", "D", "E"],
            "prediction": [5.0, 4.0, 3.0, 2.0, 1.0],  # top-3 = A,B,C
            "truth": [5.0, 4.0, 1.0, 3.0, 2.0],       # top-3 = A,B,D
        })
        metrics = metric_by_cell(df)
        # pred top-5: A,B,C,D,E; true top-5: A,B,D,C,E; overlap: all 5 → 1.0
        assert abs(metrics["precision_at_5"] - 1.0) < 1e-10


class TestComputeAllMetrics:
    def test_rmse_zero_gives_perfect_score(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        cells = np.array(["C1"] * 5)
        genes = np.array([f"G{i}" for i in range(5)])
        metrics = compute_all_metrics(y, y, cells, genes)
        assert abs(metrics["rmse"] - 0.0) < 1e-10
        assert abs(metrics["rmse_score"] - 1.0) < 1e-10

    def test_final_score_range(self):
        np.random.seed(42)
        n = 200
        y_true = np.random.randn(n).astype(np.float64)
        y_pred = y_true + 0.3 * np.random.randn(n).astype(np.float64)
        cells = np.array([f"C{i % 10}" for i in range(n)])
        genes = np.array([f"G{i}" for i in range(n)])
        metrics = compute_all_metrics(y_true, y_pred, cells, genes)
        assert 0.0 <= metrics["final_score"] <= 100.0
        assert 0.0 <= metrics["spearman_score"] <= 1.0
        assert 0.0 <= metrics["ndcg_score"] <= 1.0
        assert 0.0 <= metrics["precision_score"] <= 1.0
        assert 0.0 <= metrics["rmse_score"] <= 1.0


class TestParityWithOfficial:
    """Verify parity with the official scoring script on a small fixture."""

    @pytest.fixture
    def official_script(self):
        """Import the official calculate_metric module."""
        import calculate_metric as official
        return official

    def test_synthetic_parity(self, official_script, tmp_path):
        """Our metrics must match the official script exactly on synthetic data."""
        np.random.seed(123)
        n_cells = 10
        n_genes = 50
        cell_pool = [f"ACH-{i:06d}" for i in range(n_cells)]
        gene_pool = [f"GENE_{i}" for i in range(n_genes)]

        # Generate unique (cell, gene) pairs
        pairs = set()
        while len(pairs) < 500:
            c = cell_pool[np.random.randint(n_cells)]
            g = gene_pool[np.random.randint(n_genes)]
            pairs.add((c, g))
        pairs = list(pairs)
        cell_ids = np.array([p[0] for p in pairs])
        genes = np.array([p[1] for p in pairs])
        truth = np.random.randn(len(pairs)).astype(np.float64) * 0.5 + 0.2
        pred = truth + np.random.randn(len(pairs)).astype(np.float64) * 0.2

        # Build submission & answer CSVs
        sub = pd.DataFrame({
            "cell_line_id": cell_ids,
            "perturbation_gene": genes,
            "label": pred,
        })
        ans = pd.DataFrame({
            "cell_line_id": cell_ids,
            "perturbation_gene": genes,
            "label": truth,
        })
        sub_path = tmp_path / "sub.csv"
        ans_path = tmp_path / "ans.csv"
        sub.to_csv(sub_path, index=False)
        ans.to_csv(ans_path, index=False)

        # Official computation
        official_sub = official_script.read_submission(sub_path)
        official_ans = official_script.read_answer(ans_path)
        official_df = official_ans.merge(
            official_sub, on=["cell_line_id", "perturbation_gene"],
            how="left", validate="one_to_one",
        )
        off_metrics = official_script.metric_by_cell(official_df)
        off_rmse = float(np.sqrt(np.mean(
            np.square(official_df["prediction"] - official_df["truth"])
        )))
        off_sigma = float(official_df["truth"].std(ddof=0))
        off_nrmse = off_rmse / (off_sigma + 1e-12)
        off_rmse_score = 1.0 / (1.0 + off_nrmse)

        # Our computation
        df = pd.DataFrame({
            "cell_line_id": cell_ids,
            "perturbation_gene": genes,
            "prediction": pred,
            "truth": truth,
        })
        from src.prediction.metrics import compute_metrics_df
        our_metrics = compute_metrics_df(df)

        # Compare metric_by_cell outputs
        assert abs(our_metrics["spearman_score"] - off_metrics["spearman_score"]) < 1e-9
        assert abs(our_metrics["rho_macro"] - off_metrics["rho_macro"]) < 1e-9
        assert abs(our_metrics["ndcg_at_5"] - off_metrics["ndcg_at_5"]) < 1e-9
        assert abs(our_metrics["ndcg_at_10"] - off_metrics["ndcg_at_10"]) < 1e-9
        assert abs(our_metrics["ndcg_at_15"] - off_metrics["ndcg_at_15"]) < 1e-9
        assert abs(our_metrics["ndcg_score"] - off_metrics["ndcg_score"]) < 1e-9
        assert abs(our_metrics["precision_at_5"] - off_metrics["precision_at_5"]) < 1e-9
        assert abs(our_metrics["precision_at_10"] - off_metrics["precision_at_10"]) < 1e-9
        assert abs(our_metrics["precision_at_15"] - off_metrics["precision_at_15"]) < 1e-9
        assert abs(our_metrics["precision_score"] - off_metrics["precision_score"]) < 1e-9

        # Compare RMSE
        assert abs(our_metrics["rmse"] - off_rmse) < 1e-9
        assert abs(our_metrics["nrmse"] - off_nrmse) < 1e-9
        assert abs(our_metrics["rmse_score"] - off_rmse_score) < 1e-9

    def test_real_data_subset_parity(self, official_script, tmp_path):
        """Parity on a small subset of real training labels."""
        config_path = project_root / "config.yaml"
        from src.utils import load_config
        config = load_config(str(config_path))
        labels_dir = project_root / config["paths"]["data_dir"] / "labels"
        labels = pd.read_csv(labels_dir / "gene_dependency.csv", nrows=2000)

        pred = labels["label"] + np.random.randn(len(labels)) * 0.1
        sub = labels[["cell_line_id", "perturbation_gene"]].copy()
        sub["label"] = pred
        ans = labels[["cell_line_id", "perturbation_gene"]].copy()
        ans["label"] = labels["label"]

        sub_path = tmp_path / "sub_real.csv"
        ans_path = tmp_path / "ans_real.csv"
        sub.to_csv(sub_path, index=False)
        ans.to_csv(ans_path, index=False)

        official_sub = official_script.read_submission(sub_path)
        official_ans = official_script.read_answer(ans_path)
        official_df = official_ans.merge(
            official_sub, on=["cell_line_id", "perturbation_gene"],
            how="left", validate="one_to_one",
        )
        off_metrics = official_script.metric_by_cell(official_df)

        df = pd.DataFrame({
            "cell_line_id": labels["cell_line_id"].values,
            "perturbation_gene": labels["perturbation_gene"].values,
            "prediction": pred.values,
            "truth": labels["label"].values,
        })
        from src.prediction.metrics import compute_metrics_df
        our_metrics = compute_metrics_df(df)

        for key in ["spearman_score", "ndcg_score", "precision_score",
                     "ndcg_at_5", "ndcg_at_10", "ndcg_at_15",
                     "precision_at_5", "precision_at_10", "precision_at_15"]:
            assert abs(our_metrics[key] - off_metrics[key]) < 1e-9, \
                f"Mismatch on {key}: ours={our_metrics[key]:.10f}, official={off_metrics[key]:.10f}"
