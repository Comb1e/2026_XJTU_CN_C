"""Interpretable model building blocks for gene dependency prediction.

All models in this module are fully interpretable:
  - LogisticRegression for pairwise ranking (Bradley-Terry)
  - Ridge regression for calibration
  - Quantile mapping for monotone distribution matching
  - Rank-space blending (algebraic, no learned parameters)

NO tree ensembles, gradient boosting, or neural networks.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV


# ── Pairwise ranking (Bradley-Terry) ────────────────────────────────────────


def sample_pairwise_pairs(
    X: np.ndarray,
    y: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    label_diff_threshold: float = 0.05,
    max_pairs_per_cell: int = 2000,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample pairwise comparisons within each cell line.

    For each cell, sample pairs (g1, g2) where |y1 - y2| > threshold.
    Returns feature differences Δx = x1 - x2 and labels (1 if y1 > y2 else 0).

    This is a data utility — no model training happens here.
    """
    rng = np.random.RandomState(random_state)
    diff_list = []
    label_list = []

    for cell in np.unique(cell_ids):
        mask = cell_ids == cell
        if mask.sum() < 2:
            continue
        X_cell = X[mask]
        y_cell = y[mask]
        n = len(y_cell)

        # Find all pairs with sufficient label difference
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if abs(y_cell[i] - y_cell[j]) > label_diff_threshold:
                    pairs.append((i, j))

        if len(pairs) > max_pairs_per_cell:
            idx = rng.choice(len(pairs), max_pairs_per_cell, replace=False)
            pairs = [pairs[i] for i in idx]

        for i, j in pairs:
            diff_list.append(X_cell[i] - X_cell[j])
            label_list.append(1 if y_cell[i] > y_cell[j] else 0)

    if not diff_list:
        return np.array([]).reshape(0, X.shape[1]), np.array([], dtype=int)

    return np.array(diff_list, dtype=np.float32), np.array(label_list, dtype=np.int32)


def train_pairwise_ranker(
    X: np.ndarray,
    y: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression | None:
    """Train a Bradley-Terry pairwise ranker (LogisticRegression).

    The learned weight vector w is directly interpretable:
      score(g) = w · x(g)
    Each coefficient w_j shows how feature j affects the gene's relative
    dependency rank within a cell line.
    """
    X_diff, y_diff = sample_pairwise_pairs(X, y, cell_ids, gene_ids)
    if len(X_diff) == 0:
        return None

    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_diff, y_diff)
    return model


def predict_pairwise_ranker(
    model: LogisticRegression | None,
    X: np.ndarray,
) -> np.ndarray:
    """Predict scores from a pairwise ranker.

    Returns w·x(g) for each gene — a linear scoring function where
    each feature's contribution is w_j × x_j(g).
    """
    if model is None:
        return np.zeros(len(X), dtype=np.float32)
    return (X.astype(np.float64) @ model.coef_[0]).astype(np.float32)


# ── Rank-space blending ──────────────────────────────────────────────────────


def _to_ranks(values: np.ndarray) -> np.ndarray:
    """Convert values to ranks (1 = highest, ties get average)."""
    from scipy.stats import rankdata
    return rankdata(-values, method="average")


def blend_ranks(
    preds_a: np.ndarray,
    preds_b: np.ndarray | None,
    cell_ids: np.ndarray,
    alpha_a: float = 0.5,
    alpha_b: float = 0.5,
    alpha_c: float = 0.0,
    preds_c: np.ndarray | None = None,
    cold_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Blend model predictions in rank space (algebraic, no learned parameters).

    Within each cell: rank_final = α_A·rank_A + α_B·rank_B + α_C·rank_C.
    For cold-start genes, α_B = 0.
    Per-row normalization prevents cold-gene dynamic range compression.

    Returns blended raw scores (not ranks).
    """
    result = np.zeros(len(preds_a), dtype=np.float32)

    for cell in np.unique(cell_ids):
        mask = cell_ids == cell
        n = mask.sum()
        if n == 0:
            continue

        rank_a = _to_ranks(preds_a[mask])
        rank_sum = alpha_a * rank_a
        per_row_weight = np.full(n, alpha_a, dtype=np.float64)

        if preds_b is not None:
            if cold_mask is not None:
                b_weight = np.where(cold_mask[mask], 0.0, alpha_b)
            else:
                b_weight = np.full(n, alpha_b, dtype=np.float64)
            rank_b = _to_ranks(preds_b[mask])
            rank_sum += b_weight * rank_b
            per_row_weight += b_weight

        if preds_c is not None and alpha_c > 0:
            rank_c = _to_ranks(preds_c[mask])
            rank_sum += alpha_c * rank_c
            per_row_weight += alpha_c

        # Per-row normalization
        valid = per_row_weight > 0
        rank_sum[valid] /= per_row_weight[valid]

        # Invert: smaller rank = higher score
        result[mask] = (n - rank_sum) / n

    return result


def blend_ranks_multi(
    cell_ids: np.ndarray,
    cold_mask: np.ndarray | None = None,
    models: list[tuple | None] | None = None,
) -> np.ndarray:
    """Blend N model predictions in rank space with per-row normalization.

    Args:
        cell_ids: cell identifier for each row.
        cold_mask: boolean mask for cold-start genes (True = cold).
        models: list of (predictions, alpha_weight, is_cold_safe) tuples.
                None entries are skipped.

    Returns blended raw scores (not ranks) in [0, 1].

    This is purely algebraic — no parameters are learned.
    """
    if models is None:
        models = []
    active = [(p, a, cs) for item in models if item is not None
              for p, a, cs in [item] if p is not None and a > 0]
    if not active:
        return np.zeros(len(cell_ids), dtype=np.float32)

    result = np.zeros(len(cell_ids), dtype=np.float32)

    for cell in np.unique(cell_ids):
        mask = cell_ids == cell
        n = mask.sum()
        if n == 0:
            continue

        rank_sum = np.zeros(n, dtype=np.float64)
        per_row_weight = np.zeros(n, dtype=np.float64)

        for preds, alpha, is_cold_safe in active:
            if not is_cold_safe and cold_mask is not None:
                weight = np.where(cold_mask[mask], 0.0, alpha)
            else:
                weight = np.full(n, alpha, dtype=np.float64)
            rank_sum += weight * _to_ranks(preds[mask])
            per_row_weight += weight

        valid = per_row_weight > 0
        rank_sum[valid] /= per_row_weight[valid]
        result[mask] = (n - rank_sum) / n

    return result


# ── Calibration (interpretable linear + quantile methods) ────────────────────


def calibrate_quantile(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    n_quantiles: int = 1000,
) -> callable:
    """Build a monotone quantile calibrator.

    Maps predictions through the empirical CDF⁻¹ of training labels.
    This is monotone → zero risk to Spearman/NDCG/Precision, and forces the
    prediction marginal onto the label marginal, fixing scale mismatch.

    Fully interpretable: the mapping is a lookup table y_q = F⁻¹(q) where
    q = percentile of prediction and F⁻¹ is the inverse CDF of labels.
    """
    quantiles = np.linspace(0.0, 1.0, n_quantiles)
    label_quantiles = np.quantile(y_true, quantiles)
    # Ensure monotonicity
    label_quantiles = np.maximum.accumulate(label_quantiles)

    def _apply(y: np.ndarray) -> np.ndarray:
        return np.interp(
            np.clip(y, 0.0, 1.0), quantiles, label_quantiles,
        ).astype(np.float32)

    _apply.label_quantiles = label_quantiles
    _apply.quantiles = quantiles
    return _apply


def calibrate_rmse(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    gene_baselines: np.ndarray | None = None,
    cell_biases: np.ndarray | None = None,
    module_match_sum: np.ndarray | None = None,
    alphas: list[float] | None = None,
) -> Ridge:
    """Fit a Ridge regression to calibrate predictions to absolute scale.

    Features: [1, ŷ, μ̂_g, β̂_c, ŷ², module_match]

    Interpretable: Ridge coefficients show how each calibration feature
    (prediction, baseline, bias, squared prediction) contributes to the
    final calibrated value.
    """
    if alphas is None:
        alphas = [0.1, 1.0, 10.0, 100.0, 1000.0]

    feats = [np.ones(len(y_pred), dtype=np.float32), y_pred.astype(np.float32)]
    if gene_baselines is not None:
        feats.append(gene_baselines.astype(np.float32))
    if cell_biases is not None:
        feats.append(cell_biases.astype(np.float32))
    feats.append((y_pred ** 2).astype(np.float32))
    if module_match_sum is not None:
        feats.append(module_match_sum.astype(np.float32))

    X_cal = np.column_stack(feats)
    cal = RidgeCV(alphas=alphas, store_cv_results=False)
    cal.fit(X_cal, y_true)
    return cal


def apply_calibration(
    cal: Ridge | None,
    y_pred: np.ndarray,
    gene_baselines: np.ndarray | None = None,
    cell_biases: np.ndarray | None = None,
    module_match_sum: np.ndarray | None = None,
    clip_range: tuple[float, float] = (-3.0, 5.0),
) -> np.ndarray:
    """Apply a trained Ridge calibration to predictions."""
    if cal is None:
        return np.clip(y_pred, clip_range[0], clip_range[1])

    feats = [np.ones(len(y_pred), dtype=np.float32), y_pred.astype(np.float32)]
    if gene_baselines is not None:
        feats.append(gene_baselines.astype(np.float32))
    if cell_biases is not None:
        feats.append(cell_biases.astype(np.float32))
    feats.append((y_pred ** 2).astype(np.float32))
    if module_match_sum is not None:
        feats.append(module_match_sum.astype(np.float32))

    X_cal = np.column_stack(feats)
    calibrated = cal.predict(X_cal)
    return np.clip(calibrated, clip_range[0], clip_range[1])
