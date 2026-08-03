"""Per-cell calibration and quantile alignment for gene dependency predictions.

Key insight (CAIRO 2026, Menon et al. ICML 2012):
  Ranking-first training + isotonic calibration recovers the true regression
  function. Since 85% of the scoring metric depends on per-cell ranking,
  improving calibration within each cell directly boosts the score.

Two complementary approaches:
  1. PerCellIsotonicCalibrator — isotonic regression per cell (uses truth)
  2. PerCellQuantileAligner — quantile mapping cold→warm distribution (no truth needed)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


class PerCellIsotonicCalibrator:
    """Per-cell isotonic regression calibration.

    For each cell c, learns a monotonic mapping f_c: pred → truth
    using isotonic regression (PAV algorithm) on warm genes. Then applies
    f_c to ALL genes (warm + cold) in that cell.

    Properties:
      - Monotonic: preserves gene ordering within each cell
      - Auto-calibrated: f_c(pred) ≈ E[truth | pred] within each level set
      - Cold-safe: calibration curve learned from warm genes, applied to cold

    Reference:
      - CAIRO (2026): "Calibrate After Initial Rank Ordering"
      - Menon et al. (ICML 2012): "Rank + Isotonic Regression"
    """

    def __init__(
        self,
        y_min: float | None = None,
        y_max: float | None = None,
        min_samples: int = 10,
    ):
        """
        Args:
            y_min, y_max: clip calibrated values to this range.
            min_samples: minimum warm genes per cell to fit calibration.
                         Cells with fewer samples use global calibration.
        """
        self.y_min = y_min
        self.y_max = y_max
        self.min_samples = min_samples

        # Per-cell calibrators: cell_id → IsotonicRegression
        self._cell_calibrators: dict[str, IsotonicRegression] = {}
        # Global fallback calibrator
        self._global_calibrator: IsotonicRegression | None = None
        # Stats
        self._n_cells_calibrated: int = 0
        self._n_cells_global: int = 0

    def fit(
        self,
        preds: np.ndarray,
        truths: np.ndarray,
        cell_ids: np.ndarray,
        gene_ids: np.ndarray,
        cold_genes: set[str] | None = None,
        verbose: bool = True,
    ) -> "PerCellIsotonicCalibrator":
        """Fit per-cell isotonic calibration curves.

        Only warm genes are used for fitting. Cold genes are ignored
        during fit but will be transformed later.

        Args:
            preds: (N,) raw predictions.
            truths: (N,) ground truth labels.
            cell_ids: (N,) cell identifiers.
            gene_ids: (N,) gene identifiers.
            cold_genes: set of cold gene symbols (excluded from fit).
        """
        if cold_genes is None:
            cold_genes = set()

        # ── Global calibration (fallback) ──
        warm_mask = np.array([g not in cold_genes for g in gene_ids])
        if warm_mask.sum() >= self.min_samples:
            self._global_calibrator = IsotonicRegression(
                y_min=self.y_min, y_max=self.y_max,
                out_of_bounds="clip", increasing=True,
            )
            self._global_calibrator.fit(
                preds[warm_mask].astype(np.float64),
                truths[warm_mask].astype(np.float64),
            )

        # ── Per-cell calibration ──
        unique_cells = sorted(set(cell_ids))
        for cell in unique_cells:
            cell_mask = (cell_ids == cell) & warm_mask
            if cell_mask.sum() >= self.min_samples:
                cal = IsotonicRegression(
                    y_min=self.y_min, y_max=self.y_max,
                    out_of_bounds="clip", increasing=True,
                )
                cal.fit(
                    preds[cell_mask].astype(np.float64),
                    truths[cell_mask].astype(np.float64),
                )
                self._cell_calibrators[cell] = cal
                self._n_cells_calibrated += 1
            else:
                self._n_cells_global += 1

        if verbose:
            print(f"  [Calibrate] {self._n_cells_calibrated} cells with "
                  f"per-cell calibration, {self._n_cells_global} using global")

        return self

    def transform(
        self,
        preds: np.ndarray,
        cell_ids: np.ndarray,
    ) -> np.ndarray:
        """Apply per-cell isotonic calibration.

        For cells with a fitted calibrator, applies the cell-specific curve.
        Otherwise falls back to global calibration. For cells with neither
        (extremely rare), returns raw predictions unchanged.
        """
        calibrated = preds.copy().astype(np.float64)
        unique_cells = set(cell_ids)

        for cell in unique_cells:
            cell_mask = cell_ids == cell
            if cell in self._cell_calibrators:
                calibrated[cell_mask] = self._cell_calibrators[cell].transform(
                    preds[cell_mask].astype(np.float64),
                )
            elif self._global_calibrator is not None:
                calibrated[cell_mask] = self._global_calibrator.transform(
                    preds[cell_mask].astype(np.float64),
                )

        return calibrated

    def fit_transform(
        self,
        preds: np.ndarray,
        truths: np.ndarray,
        cell_ids: np.ndarray,
        gene_ids: np.ndarray,
        cold_genes: set[str] | None = None,
    ) -> np.ndarray:
        """Fit and transform in one call."""
        self.fit(preds, truths, cell_ids, gene_ids, cold_genes)
        return self.transform(preds, cell_ids)


class PerCellQuantileAligner:
    """Align cold-gene prediction quantiles to match warm-gene quantiles per cell.

    Problem: cold gene predictions often cluster at a narrow range (dominated by
    cell-invariant Φ(g)), causing them to all rank similarly within each cell.
    This kills per-cell ranking metrics (Spearman, nDCG, Precision@K).

    Solution: within each cell, map cold gene predictions so their empirical CDF
    matches the warm gene prediction CDF. This is a monotonic (rank-preserving)
    transform for cold genes, but spreads them across the full warm-gene
    prediction range.

    Specifically:
        ŷ'_cold = F^{-1}_warm(F_cold(ŷ_cold))
    where F_warm, F_cold are empirical CDFs of warm/cold predictions in cell c.

    This ensures cold genes are properly interleaved with warm genes in the
    per-cell ranking, rather than all clustered at one position.
    """

    def __init__(self, min_warm: int = 10, min_cold: int = 3):
        """
        Args:
            min_warm: minimum warm genes per cell for alignment.
            min_cold: minimum cold genes per cell for alignment.
        """
        self.min_warm = min_warm
        self.min_cold = min_cold

    def align(
        self,
        preds: np.ndarray,
        cell_ids: np.ndarray,
        gene_ids: np.ndarray,
        cold_genes: set[str],
        verbose: bool = True,
    ) -> np.ndarray:
        """Align cold gene quantiles to match warm gene distribution per cell.

        Args:
            preds: (N,) raw predictions.
            cell_ids: (N,) cell identifiers.
            gene_ids: (N,) gene identifiers.
            cold_genes: set of cold gene symbols.

        Returns:
            aligned predictions (same shape as preds).
        """
        aligned = preds.copy().astype(np.float64)
        n_cells_aligned = 0
        n_cold_aligned = 0

        unique_cells = set(cell_ids)
        for cell in unique_cells:
            cell_mask = cell_ids == cell
            cold_mask = cell_mask & np.array([g in cold_genes for g in gene_ids])
            warm_mask = cell_mask & ~cold_mask

            n_warm = warm_mask.sum()
            n_cold = cold_mask.sum()

            if n_warm < self.min_warm or n_cold < self.min_cold:
                continue

            warm_preds = preds[warm_mask]
            cold_preds = preds[cold_mask]

            # Compute empirical CDF of warm predictions
            warm_sorted = np.sort(warm_preds)
            # Compute empirical CDF of cold predictions
            cold_sorted = np.sort(cold_preds)

            # Map each cold prediction to warm quantile
            # F_cold(p) = rank of p in cold_preds / n_cold
            # ŷ' = warm value at quantile F_cold(p)
            cold_ranks = np.searchsorted(cold_sorted, cold_preds, side="right")
            cold_quantiles = cold_ranks / n_cold  # in [0, 1]

            # Map quantile to warm value
            warm_indices = np.clip(
                (cold_quantiles * (n_warm - 1)).astype(int),
                0, n_warm - 1,
            )
            aligned[cold_mask] = warm_sorted[warm_indices]

            n_cells_aligned += 1
            n_cold_aligned += n_cold

        if verbose and n_cells_aligned > 0:
            print(f"  [QuantileAlign] {n_cells_aligned} cells, "
                  f"{n_cold_aligned:,} cold gene predictions aligned")

        return aligned


def per_cell_variance_match(
    preds: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    cold_genes: set[str],
    min_warm: int = 10,
    min_cold: int = 3,
    verbose: bool = True,
) -> np.ndarray:
    """Match cold gene prediction variance to warm gene variance per cell.

    Problem: Cold gene predictions cluster near the mean (low within-cell variance),
    so they never appear at the top or bottom of per-cell rankings. This kills
    NDCG and Precision scores.

    Solution: Within each cell, apply a linear transform to cold gene predictions:
        p'_cold = (p_cold - μ_cold) * (σ_warm / σ_cold) + μ_warm

    This preserves the RELATIVE ORDER of cold genes (linear = monotonic) but
    spreads them to match the warm gene prediction spread. Cold genes can then
    reach top/bottom positions in the per-cell ranking.

    Args:
        preds: (N,) raw predictions.
        cell_ids: (N,) cell identifiers.
        gene_ids: (N,) gene identifiers.
        cold_genes: set of cold gene symbols.
        min_warm, min_cold: minimum genes per cell to apply.

    Returns:
        aligned predictions (same shape as preds).
    """
    matched = preds.copy().astype(np.float64)
    n_cells_done = 0

    unique_cells = set(cell_ids)
    for cell in unique_cells:
        cell_mask = cell_ids == cell
        cold_mask = cell_mask & np.array([g in cold_genes for g in gene_ids])
        warm_mask = cell_mask & ~cold_mask

        n_warm = warm_mask.sum()
        n_cold = cold_mask.sum()

        if n_warm < min_warm or n_cold < min_cold:
            continue

        warm_preds = preds[warm_mask]
        cold_preds = preds[cold_mask]

        mu_w = np.mean(warm_preds)
        std_w = np.std(warm_preds)
        mu_c = np.mean(cold_preds)
        std_c = np.std(cold_preds)

        if std_c < 1e-8 or std_w < 1e-8:
            continue

        # Linear transform: match mean and variance
        matched[cold_mask] = (cold_preds - mu_c) * (std_w / std_c) + mu_w
        n_cells_done += 1

    if verbose and n_cells_done > 0:
        print(f"  [VarMatch] {n_cells_done} cells: cold gene variance "
              f"matched to warm gene variance")

    return matched


def per_cell_standardize(
    preds: np.ndarray,
    cell_ids: np.ndarray,
    method: str = "zscore",
) -> np.ndarray:
    """Per-cell standardization/ranking of predictions.

    Simple baseline: within each cell, transform predictions to have
    consistent scale. This removes cell-specific bias that might cause
    one cell's predictions to dominate.

    Args:
        preds: (N,) predictions.
        cell_ids: (N,) cell identifiers.
        method: "zscore" | "rank" | "quantile".

    Returns:
        transformed predictions.
    """
    transformed = preds.copy().astype(np.float64)
    unique_cells = set(cell_ids)

    for cell in unique_cells:
        mask = cell_ids == cell
        cell_preds = preds[mask]

        if method == "zscore":
            std = np.std(cell_preds)
            if std > 1e-8:
                transformed[mask] = (cell_preds - np.mean(cell_preds)) / std
            else:
                transformed[mask] = 0.0
        elif method == "rank":
            from scipy.stats import rankdata
            n = len(cell_preds)
            ranks = rankdata(cell_preds, method="average")
            # Map to [-1, 1] via inverse normal CDF
            quantiles = (ranks - 0.5) / n
            transformed[mask] = quantiles
        elif method == "quantile":
            n = len(cell_preds)
            ranks = np.argsort(np.argsort(cell_preds))
            transformed[mask] = ranks.astype(np.float64) / max(n - 1, 1)

    return transformed
