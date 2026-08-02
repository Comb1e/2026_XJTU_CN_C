"""Vectorized reimplementation of the official gene-dependency scoring metric.

Exact parity with 数据文件/calculate_metric.py. All metrics are computed
per cell line then macro-averaged.

Reference: calculate_metric.py in the competition data package.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

KS = (5, 10, 15)
EPS = 1e-12


def pearson_or_zero(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation, returning 0 when variance is negligible."""
    x = x - x.mean()
    y = y - y.mean()
    denom = math.sqrt(float(np.dot(x, x) * np.dot(y, y)))
    if denom <= EPS:
        return 0.0
    return float(np.dot(x, y) / denom)


def metric_by_cell(df: pd.DataFrame) -> dict[str, float]:
    """Compute per-cell-line metrics from a merged DataFrame.

    Required columns: cell_line_id, prediction, truth, _order.

    Returns dict with rho_macro, spearman_score, precision_at_5/10/15,
    precision_score, ndcg_at_5/10/15, ndcg_score.
    """
    if "_order" not in df.columns:
        df = df.copy()
        df["_order"] = np.arange(len(df))

    spearman_values = []
    precision_values = {k: [] for k in KS}
    ndcg_values = {k: [] for k in KS}

    for _, group in df.groupby("cell_line_id", sort=False):
        group = group.copy()
        group["pred_rank"] = group["prediction"].rank(ascending=False, method="average")
        group["true_rank"] = group["truth"].rank(ascending=False, method="average")
        spearman_values.append(
            pearson_or_zero(
                group["pred_rank"].to_numpy(dtype=np.float64),
                group["true_rank"].to_numpy(dtype=np.float64),
            )
        )

        pred_sorted = group.sort_values(
            ["prediction", "_order"], ascending=[False, True]
        )
        true_sorted = group.sort_values(
            ["truth", "_order"], ascending=[False, True]
        )

        for k in KS:
            pred_top = pred_sorted.head(k)
            true_top_genes = set(true_sorted.head(k)["perturbation_gene"])
            precision_values[k].append(
                len(set(pred_top["perturbation_gene"]) & true_top_genes) / k
            )

            # Graded relevance: rel = max(k - true_rank + 1, 0), zero if true_rank > k
            rel = np.maximum(
                k - pred_top["true_rank"].to_numpy(dtype=np.float64) + 1.0, 0.0
            )
            rel[pred_top["true_rank"].to_numpy(dtype=np.float64) > k] = 0.0
            discounts = np.log2(np.arange(2, len(rel) + 2, dtype=np.float64))
            dcg = float(np.sum((np.power(2.0, rel) - 1.0) / discounts))
            ideal_rel = np.arange(k, 0, -1, dtype=np.float64)
            ideal_discounts = np.log2(np.arange(2, k + 2, dtype=np.float64))
            idcg = float(np.sum((np.power(2.0, ideal_rel) - 1.0) / ideal_discounts))
            ndcg_values[k].append(dcg / idcg if idcg > 0 else 0.0)

    rho_macro = float(np.mean(spearman_values))
    precision_macro = {k: float(np.mean(precision_values[k])) for k in KS}
    ndcg_macro = {k: float(np.mean(ndcg_values[k])) for k in KS}

    return {
        "rho_macro": rho_macro,
        "spearman_score": (rho_macro + 1.0) / 2.0,
        "precision_at_5": precision_macro[5],
        "precision_at_10": precision_macro[10],
        "precision_at_15": precision_macro[15],
        "precision_score": float(np.mean(list(precision_macro.values()))),
        "ndcg_at_5": ndcg_macro[5],
        "ndcg_at_10": ndcg_macro[10],
        "ndcg_at_15": ndcg_macro[15],
        "ndcg_score": float(np.mean(list(ndcg_macro.values()))),
    }


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
) -> dict[str, float]:
    """Compute the full competition metric from raw arrays.

    Args:
        y_true: true labels, shape (N,).
        y_pred: predicted labels, shape (N,).
        cell_ids: cell_line_id for each pair, shape (N,).
        gene_ids: perturbation_gene for each pair, shape (N,).

    Returns:
        dict with keys: final_score, spearman_score, ndcg_score,
        precision_score, rmse_score, rmse, nrmse, plus per-K breakdowns.
    """
    df = pd.DataFrame({
        "cell_line_id": cell_ids,
        "perturbation_gene": gene_ids,
        "prediction": np.asarray(y_pred, dtype=np.float64),
        "truth": np.asarray(y_true, dtype=np.float64),
    })
    return compute_metrics_df(df)


def compute_metrics_df(df: pd.DataFrame) -> dict[str, float]:
    """Compute full metric from a DataFrame with columns:
    cell_line_id, perturbation_gene, prediction, truth.
    """
    if "_order" not in df.columns:
        df = df.copy()
        df["_order"] = np.arange(len(df))

    metrics = metric_by_cell(df)

    # Global RMSE
    rmse = float(np.sqrt(np.mean(np.square(
        df["prediction"].to_numpy(dtype=np.float64)
        - df["truth"].to_numpy(dtype=np.float64)
    ))))
    sigma_y = float(df["truth"].std(ddof=0))
    nrmse = rmse / (sigma_y + EPS)
    rmse_score = 1.0 / (1.0 + nrmse)

    final_score = 100.0 * (
        0.30 * metrics["spearman_score"]
        + 0.30 * metrics["ndcg_score"]
        + 0.25 * metrics["precision_score"]
        + 0.15 * rmse_score
    )

    return {
        "final_score": final_score,
        "spearman_score": metrics["spearman_score"],
        "rho_macro": metrics["rho_macro"],
        "ndcg_score": metrics["ndcg_score"],
        "ndcg_at_5": metrics["ndcg_at_5"],
        "ndcg_at_10": metrics["ndcg_at_10"],
        "ndcg_at_15": metrics["ndcg_at_15"],
        "precision_score": metrics["precision_score"],
        "precision_at_5": metrics["precision_at_5"],
        "precision_at_10": metrics["precision_at_10"],
        "precision_at_15": metrics["precision_at_15"],
        "rmse_score": rmse_score,
        "rmse": rmse,
        "nrmse": nrmse,
        "n_rows": len(df),
        "n_cells": int(df["cell_line_id"].nunique()),
    }


def format_metric_report(metrics: dict[str, float]) -> str:
    """Format metrics as a human-readable report string."""
    width = 18
    lines = []
    lines.append("Gene Dependency Score")
    lines.append("=" * (width + 14))
    lines.append(f"{'Final score':<{width}} : {metrics['final_score']:.6f} / 100")
    lines.append(
        f"{'SpearmanScore':<{width}} : {metrics['spearman_score']:.6f}"
        f"  (rho_macro={metrics['rho_macro']:.6f})"
    )
    lines.append(f"{'NDCGScore':<{width}} : {metrics['ndcg_score']:.6f}")
    lines.append(f"{'  NDCG@5':<{width}} : {metrics['ndcg_at_5']:.6f}")
    lines.append(f"{'  NDCG@10':<{width}} : {metrics['ndcg_at_10']:.6f}")
    lines.append(f"{'  NDCG@15':<{width}} : {metrics['ndcg_at_15']:.6f}")
    lines.append(f"{'PrecisionScore':<{width}} : {metrics['precision_score']:.6f}")
    lines.append(f"{'  Precision@5':<{width}} : {metrics['precision_at_5']:.6f}")
    lines.append(f"{'  Precision@10':<{width}} : {metrics['precision_at_10']:.6f}")
    lines.append(f"{'  Precision@15':<{width}} : {metrics['precision_at_15']:.6f}")
    lines.append(f"{'RMSEScore':<{width}} : {metrics['rmse_score']:.6f}")
    lines.append(f"{'  RMSE':<{width}} : {metrics['rmse']:.6f}")
    lines.append(f"{'  NRMSE':<{width}} : {metrics['nrmse']:.6f}")
    lines.append(f"{'Rows':<{width}} : {metrics['n_rows']:,}")
    lines.append(f"{'Cell lines':<{width}} : {metrics['n_cells']:,}")
    lines.append("=" * (width + 14))
    return "\n".join(lines)
