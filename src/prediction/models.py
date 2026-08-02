"""Model training, prediction, blending, and calibration.

Model A  — HistGradientBoostingRegressor on G1–G5 (cold-start safe).
Model B  — HGBR with extra collaborative features (labeled genes only).
Model C  — Pairwise ranking (nonlinear tree-based, optional).
Model D  — XGBoost ensemble (cold-start safe).
Model E  — LightGBM LambdaMART ranker (directly optimizes NDCG).
Blending — Rank-space weighted average with per-regime weights.
Calibration — Ridge regression on global scale (RMSE optimization).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.model_selection import GroupKFold, cross_val_predict

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# ── Model A: Cold-start-safe primary regression ─────────────────────────────


DEFAULT_HGBR_PARAMS = {
    "max_iter": 400,
    "learning_rate": 0.05,
    "max_leaf_nodes": 63,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "max_features": 0.8,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": 42,
    "verbose": 0,
}


def create_model_a(params: dict[str, Any] | None = None) -> HistGradientBoostingRegressor:
    """Create Model A: HGBR on G1-G5 (no neighbor essentiality)."""
    p = {**DEFAULT_HGBR_PARAMS, **(params or {})}
    return HistGradientBoostingRegressor(**p)


def create_model_b(params: dict[str, Any] | None = None) -> HistGradientBoostingRegressor:
    """Create Model B: HGBR with extra collaborative features."""
    p = {**DEFAULT_HGBR_PARAMS, **(params or {})}
    return HistGradientBoostingRegressor(**p)


def prepare_features_a(X: pd.DataFrame) -> np.ndarray:
    """Extract Model A feature matrix (exclude neighbor_score if present)."""
    feature_cols = [c for c in X.columns
                    if c not in ("cell_line_id", "perturbation_gene")
                    and "neighbor_score" not in c]
    return X[feature_cols].to_numpy(dtype=np.float32)


def prepare_features_b(X: pd.DataFrame) -> np.ndarray:
    """Extract Model B feature matrix (all features)."""
    feature_cols = [c for c in X.columns
                    if c not in ("cell_line_id", "perturbation_gene")]
    return X[feature_cols].to_numpy(dtype=np.float32)


# ── Model C: Pairwise ranking (Bradley-Terry) ────────────────────────────────


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


def train_model_c(
    X: np.ndarray,
    y: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> LogisticRegression:
    """Train Model C (pairwise LogisticRegression)."""
    X_diff, y_diff = sample_pairwise_pairs(X, y, cell_ids, gene_ids)
    if len(X_diff) == 0:
        return None

    model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
    model.fit(X_diff, y_diff)
    return model


def predict_model_c(model: LogisticRegression | None, X: np.ndarray) -> np.ndarray:
    """Predict scores from Model C (linear scoring function)."""
    if model is None:
        return np.zeros(len(X), dtype=np.float32)
    # For Bradley-Terry, score(g) = w · x(g)
    return (X.astype(np.float64) @ model.coef_[0]).astype(np.float32)


# ── Rank-space blending ──────────────────────────────────────────────────────


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
    """Blend model predictions in rank space.

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

        # Get per-model ranks (1 = highest score)
        rank_a = _to_ranks(preds_a[mask])
        rank_sum = alpha_a * rank_a
        per_row_weight = np.full(n, alpha_a, dtype=np.float64)

        if preds_b is not None:
            # Check cold mask
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

        # Per-row normalization: avoids compressing cold-gene outputs
        valid = per_row_weight > 0
        rank_sum[valid] /= per_row_weight[valid]

        # Convert blended ranks back to scores (smaller rank = higher score)
        # Invert rank so that highest blended rank → highest score
        result[mask] = (n - rank_sum) / n

    return result


def _to_ranks(values: np.ndarray) -> np.ndarray:
    """Convert values to ranks (1 = highest, ties get average)."""
    from scipy.stats import rankdata
    return rankdata(-values, method="average")


# ── RMSE level calibration ───────────────────────────────────────────────────


def calibrate_rmse(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    gene_baselines: np.ndarray | None = None,
    cell_biases: np.ndarray | None = None,
    module_match_sum: np.ndarray | None = None,
    alphas: list[float] | None = None,
) -> Ridge:
    """Fit a Ridge regression to calibrate raw predictions to absolute scale.

    Features: [1, ŷ, μ̂_g, β̂_c, ŷ², module_match]
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
    """Apply calibration to predictions."""
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


# ── Full training pipeline ───────────────────────────────────────────────────


def train_models(
    X: pd.DataFrame,
    y: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    cold_genes: set[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train all models for gene dependency prediction.

    Args:
        X: full feature DataFrame.
        y: label array.
        cell_ids, gene_ids: identifiers for each row.
        cold_genes: set of cold-start gene symbols.
        config: prediction configuration.

    Returns:
        dict with trained models, blending weights, and calibration.
    """
    if config is None:
        config = {}
    pred_cfg = config.get("prediction", {})
    model_cfg = pred_cfg.get("models", {})
    blend_cfg = pred_cfg.get("blending", {})

    if cold_genes is None:
        cold_genes = set()

    cold_mask = np.array([g in cold_genes for g in gene_ids])

    # Model A (cold-start safe)
    print("  Training Model A (cold-start safe)...")
    Xa = prepare_features_a(X)
    model_a = create_model_a(model_cfg.get("model_a", {}))
    model_a.fit(Xa, y)
    preds_a = model_a.predict(Xa).astype(np.float32)

    # Model B (with collaborative features, only for training genes)
    print("  Training Model B (collaborative)...")
    Xb = prepare_features_b(X)
    train_mask = ~cold_mask if cold_genes else np.ones(len(y), dtype=bool)
    model_b = create_model_b(model_cfg.get("model_b", {}))
    model_b.fit(Xb[train_mask], y[train_mask])
    preds_b = model_b.predict(Xb).astype(np.float32)

    # Model C (optional pairwise ranking)
    model_c = None
    if blend_cfg.get("alpha_c", 0.0) > 0:
        print("  Training Model C (pairwise ranking)...")
        model_c = train_model_c(Xb, y, cell_ids, gene_ids)

    # Blend
    alpha_a = blend_cfg.get("alpha_a", 0.5)
    alpha_b = blend_cfg.get("alpha_b", 0.5)
    alpha_c = blend_cfg.get("alpha_c", 0.0)

    preds_c_arr = predict_model_c(model_c, Xb) if model_c else None

    print(f"  Blending ranks: α_A={alpha_a}, α_B={alpha_b}, α_C={alpha_c}")
    blended = blend_ranks(
        preds_a, preds_b, cell_ids,
        alpha_a, alpha_b, alpha_c, preds_c_arr,
        cold_mask=cold_mask,
    )

    # Calibration
    print("  Fitting RMSE calibration...")
    gene_bl = X.get("g5_gene_baseline", pd.Series(np.zeros(len(X))))
    cell_bl = X.get("g5_cell_bias", pd.Series(np.zeros(len(X))))
    cal = calibrate_rmse(
        blended, y,
        gene_baselines=gene_bl.to_numpy(dtype=np.float32) if isinstance(gene_bl, pd.Series) else gene_bl,
        cell_biases=cell_bl.to_numpy(dtype=np.float32) if isinstance(cell_bl, pd.Series) else cell_bl,
    )

    return {
        "model_a": model_a,
        "model_b": model_b,
        "model_c": model_c,
        "blend_alpha_a": alpha_a,
        "blend_alpha_b": alpha_b,
        "blend_alpha_c": alpha_c,
        "calibration": cal,
        "cold_genes": cold_genes,
    }


def predict_all(
    X: pd.DataFrame,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    models: dict[str, Any],
    add_jitter: bool = True,
) -> np.ndarray:
    """Generate final predictions using trained models.

    Args:
        X: full feature DataFrame.
        cell_ids, gene_ids: identifiers for each row.
        models: dict from train_models().
        add_jitter: add tiny deterministic jitter to avoid ties.

    Returns:
        Array of predicted labels.
    """
    cold_mask = np.array([g in models["cold_genes"] for g in gene_ids])

    Xa = prepare_features_a(X)
    preds_a = models["model_a"].predict(Xa).astype(np.float32)

    Xb = prepare_features_b(X)
    preds_b = models["model_b"].predict(Xb).astype(np.float32)

    preds_c = None
    if models.get("model_c") is not None:
        preds_c = predict_model_c(models["model_c"], Xb)

    blended = blend_ranks(
        preds_a, preds_b, cell_ids,
        models["blend_alpha_a"], models["blend_alpha_b"],
        models["blend_alpha_c"], preds_c,
        cold_mask=cold_mask,
    )

    # Calibration
    gene_bl = X.get("g5_gene_baseline", None)
    cell_bl = X.get("g5_cell_bias", None)
    if gene_bl is not None:
        gene_bl = gene_bl.to_numpy(dtype=np.float32) if isinstance(gene_bl, pd.Series) else gene_bl
    if cell_bl is not None:
        cell_bl = cell_bl.to_numpy(dtype=np.float32) if isinstance(cell_bl, pd.Series) else cell_bl

    final = apply_calib