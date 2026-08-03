"""Formula-based interpretable gene dependency prediction.

Architecture (Empirical Bayes + Inductive Matrix Completion):

    ŷ(c,g) = μ̂_g + β̂_c + x_c^T W H^T y_g

where:
  - μ̂_g = w_g·x̄_g + (1-w_g)·Φ(g)  ← empirical Bayes shrinkage
    w_g = n_g/(n_g+λ),  Φ(g) = Ridge(G1+G2 → gene mean)
  - β̂_c = v_c·r̄_c + (1-v_c)·Ψ(c)  ← same shrinkage for cells
    v_c = m_c/(m_c+λ_cell), Ψ(c) = Ridge(G3 → cell mean)
  - x_c^T W H^T y_g  ← IMC bilinear interaction
    x_c: cell features (G3), y_g: gene features (G1+G2)
    W ∈ R^{f_c×r}, H ∈ R^{f_g×r}, rank r ≪ min(f_c,f_g)
    Solved via Alternating Least Squares (ALS)

Key innovation: NO hard warm/cold distinction. The IMC interaction term
naturally handles cold genes via their feature vectors y_g — no per-gene
latent factors needed. All parameters are in W and H (shared across all
genes and cells).

References:
  - Chronos: r_cg = R*_cg/R_c − 1 (Dempster et al., Genome Biology 2021)
  - CERES: hierarchical prior + partial pooling (Meyers et al., Nat Genet 2017)
  - IMC: Provable Inductive Matrix Completion (Jain & Dhillon, 2013)
  - IMC for gene-disease: Natarajan & Dhillon, Bioinformatics 2014
  - ashr: adaptive shrinkage (Stephens, Biostatistics 2017)
  - MitoCarta3.0: 149 MitoPathways, 14 modules (Rath et al., 2021)
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════════
# Empirical Bayes Shrinkage: smooth warm→cold transition
# ═══════════════════════════════════════════════════════════════════════════════

class ShrinkageGeneFormula:
    """Empirical Bayes shrinkage for gene essentiality — NO hard warm/cold split.

    μ̂_g = w_g · x̄_g + (1 − w_g) · Φ(g)

    where:
      - x̄_g = observed mean dependency for gene g (from training labels)
      - Φ(g) = GeneEssentialityFormula prediction (Ridge on G1+G2 gene features)
      - w_g = n_g / (n_g + λ)  — evidence weight
      - n_g = number of training cell lines for gene g
      - λ = prior strength (learned via cross-validation)

    This is the same principle as:
      - Chronos: hierarchical Gaussian-kernel prior on gene effects
      - CERES: hierarchical prior + partial pooling
      - Stein/James-Stein estimator (adaptive shrinkage toward pooled mean)
      - ashr: adaptive shrinkage with data-driven prior

    Smooth transition:
      n_g ≈ 1140 → w_g ≈ 0.99 → data dominates
      n_g ≈ 10   → w_g ≈ 0.50 → balanced
      n_g = 0     → w_g = 0    → pure prior (cold gene)
    """

    def __init__(
        self,
        lambda_grid: list[float] | None = None,
        random_state: int = 42,
    ):
        if lambda_grid is None:
            lambda_grid = [1, 3, 10, 30, 100, 300, 1000, 3000]
        self.lambda_grid = lambda_grid
        self.random_state = random_state
        self.lambda_: float | None = None
        self.gene_formula_: GeneEssentialityFormula | None = None
        self.x_bar_g_: dict[str, float] = {}   # observed per-gene mean
        self.n_g_: dict[str, int] = {}          # cell count per gene
        self.phi_g_: dict[str, float] = {}      # formula prior prediction
        self.mu_g_: dict[str, float] = {}       # final shrunk estimate

    def fit(
        self,
        gene_features: pd.DataFrame,         # index=gene, columns=G1+G2 features
        labels: pd.DataFrame,                 # [cell_line_id, perturbation_gene, label]
        n_folds: int = 5,
    ) -> "ShrinkageGeneFormula":
        """Fit empirical Bayes shrinkage model.

        1. Compute per-gene stats (x̄_g, n_g)
        2. Fit GeneEssentialityFormula → Φ(g)
        3. Learn λ via group-by-gene CV
        4. Compute shrunk μ̂_g for ALL genes
        """
        from sklearn.model_selection import GroupKFold

        gene_groups = labels.groupby("perturbation_gene")["label"]
        self.x_bar_g_ = gene_groups.mean().to_dict()
        self.n_g_ = gene_groups.count().to_dict()
        all_genes = sorted(set(gene_features.index) | set(self.x_bar_g_.keys()))

        # Step 1: Fit gene formula prior Φ(g)
        print("  [ShrinkGene] Fitting gene formula prior Φ(g)...")
        self.gene_formula_ = GeneEssentialityFormula(alpha=1.0)
        self.gene_formula_.fit(gene_features, self.x_bar_g_)
        self.phi_g_ = self.gene_formula_.predict(gene_features)

        # Fill missing
        for g in all_genes:
            if g not in self.phi_g_:
                self.phi_g_[g] = 0.0
            if g not in self.x_bar_g_:
                self.x_bar_g_[g] = self.phi_g_[g]
            if g not in self.n_g_:
                self.n_g_[g] = 0

        # Step 2: Learn λ via group-by-gene cross-validation
        best_lambda = self._learn_lambda(gene_features, labels, n_folds)
        self.lambda_ = best_lambda
        print(f"  [ShrinkGene] Best λ = {self.lambda_:.1f}  "
              f"(grid: {self.lambda_grid})")

        # Step 3: Compute final shrunk μ̂_g for all genes
        for g in all_genes:
            n = self.n_g_.get(g, 0)
            w = n / (n + self.lambda_) if (n + self.lambda_) > 0 else 0.0
            self.mu_g_[g] = w * self.x_bar_g_.get(g, 0.0) + (1.0 - w) * self.phi_g_.get(g, 0.0)

        # Report
        warm_genes = [g for g, n in self.n_g_.items() if n > 0]
        cold_genes = [g for g, n in self.n_g_.items() if n == 0]
        warm_weights = [self.n_g_[g] / (self.n_g_[g] + self.lambda_) for g in warm_genes]
        print(f"  [ShrinkGene] {len(warm_genes)} warm genes: "
              f"w_g ∈ [{min(warm_weights):.3f}, {max(warm_weights):.3f}]")
        print(f"  [ShrinkGene] {len(cold_genes)} cold genes: w_g = 0 (pure prior)")
        print(f"  [ShrinkGene] σ²(x̄_g)={np.var(list(self.x_bar_g_.values())):.6f}, "
              f"σ²(μ̂_g)={np.var(list(self.mu_g_.values())):.6f}")

        return self

    def _learn_lambda(
        self,
        gene_features: pd.DataFrame,
        labels: pd.DataFrame,
        n_folds: int,
    ) -> float:
        """Learn optimal λ via group-by-gene cross-validation.

        For each fold, hold out 20% of genes, compute shrunk estimates with
        candidate λ, and evaluate RMSE on held-out genes.
        """
        from sklearn.model_selection import GroupKFold

        unique_genes = labels["perturbation_gene"].unique()
        if len(unique_genes) < 5:
            return 100.0  # default

        kf = GroupKFold(n_splits=min(n_folds, len(unique_genes)))
        gene_arr = labels["perturbation_gene"].to_numpy()
        y_arr = labels["label"].to_numpy(dtype=np.float64)

        best_lambda = 100.0
        best_score = float("inf")

        for lam in self.lambda_grid:
            scores = []
            for train_idx, val_idx in kf.split(np.arange(len(labels)), groups=gene_arr):
                val_genes = set(gene_arr[val_idx])
                train_labels = labels.iloc[train_idx]

                # Compute per-gene stats from training fold
                tg = train_labels.groupby("perturbation_gene")["label"]
                train_means = tg.mean().to_dict()
                train_counts = tg.count().to_dict()

                # Fit gene formula on training fold means
                gf = GeneEssentialityFormula(alpha=1.0)
                gf.fit(gene_features, train_means, verbose=False)
                phi = gf.predict(gene_features)

                # Predict for validation genes using shrinkage
                val_preds = []
                val_truths = []
                for _, row in labels.iloc[val_idx].iterrows():
                    g = row["perturbation_gene"]
                    n = train_counts.get(g, 0)
                    w = n / (n + lam) if (n + lam) > 0 else 0.0
                    x_bar = train_means.get(g, phi.get(g, 0.0))
                    mu = w * x_bar + (1.0 - w) * phi.get(g, 0.0)
                    val_preds.append(mu)
                    val_truths.append(row["label"])

                if len(val_preds) > 0:
                    rmse = np.sqrt(np.mean((np.array(val_preds) - np.array(val_truths)) ** 2))
                    scores.append(rmse)

            if scores:
                avg_score = np.mean(scores)
                if avg_score < best_score:
                    best_score = avg_score
                    best_lambda = lam

        return best_lambda

    def predict(self, gene: str) -> float:
        """Return shrunk gene essentiality estimate. Works for ANY gene."""
        return self.mu_g_.get(gene, 0.0)

    def predict_batch(self, gene_ids: list[str]) -> np.ndarray:
        """Return shrunk estimates for a batch of genes."""
        return np.array([self.mu_g_.get(g, 0.0) for g in gene_ids], dtype=np.float32)

    @property
    def gene_formula(self) -> "GeneEssentialityFormula | None":
        return self.gene_formula_

    @property
    def weights(self) -> dict[str, float]:
        """Return evidence weights w_g for each gene."""
        return {g: self.n_g_.get(g, 0) / (self.n_g_.get(g, 0) + self.lambda_)
                if (self.n_g_.get(g, 0) + self.lambda_) > 0 else 0.0
                for g in self.mu_g_}


class ShrinkageCellFormula:
    """Empirical Bayes shrinkage for cell vulnerability.

    β̂_c = v_c · r̄_c + (1 − v_c) · Ψ(c)

    Same principle as ShrinkageGeneFormula, applied at cell level.
    """

    def __init__(self, lambda_grid: list[float] | None = None):
        if lambda_grid is None:
            lambda_grid = [1, 3, 10, 30, 100, 300, 1000]
        self.lambda_grid = lambda_grid
        self.lambda_: float | None = None
        self.cell_formula_: CellVulnerabilityFormula | None = None
        self.r_bar_c_: dict[str, float] = {}
        self.m_c_: dict[str, int] = {}
        self.psi_c_: dict[str, float] = {}
        self.beta_c_: dict[str, float] = {}

    def fit(
        self,
        cell_features: pd.DataFrame,
        residuals: np.ndarray,               # y - μ̂_g
        cell_ids: np.ndarray,
        gene_ids: np.ndarray,
    ) -> "ShrinkageCellFormula":
        """Fit empirical Bayes shrinkage for cell vulnerability.

        residuals[c] = y_{c,g} - μ̂_g
        r̄_c = mean(residuals for cell c)
        """
        # Per-cell mean residual
        unique_cells = sorted(set(cell_ids))
        for c in unique_cells:
            mask = cell_ids == c
            if mask.sum() > 0:
                self.r_bar_c_[c] = float(residuals[mask].mean())
                self.m_c_[c] = int(mask.sum())

        # Fit cell formula prior Ψ(c)
        print("  [ShrinkCell] Fitting cell formula prior Ψ(c)...")
        self.cell_formula_ = CellVulnerabilityFormula(alpha=10.0)
        self.cell_formula_.fit(cell_features, self.r_bar_c_)
        self.psi_c_ = self.cell_formula_.predict(cell_features)

        # Fill missing cells
        for c in unique_cells:
            if c not in self.psi_c_:
                self.psi_c_[c] = 0.0
            if c not in self.r_bar_c_:
                self.r_bar_c_[c] = 0.0
                self.m_c_[c] = 0

        # Learn λ
        best_lambda = self._learn_lambda(residuals, cell_ids)
        self.lambda_ = best_lambda
        print(f"  [ShrinkCell] Best λ_cell = {self.lambda_:.1f}")

        # Compute shrunk β̂_c
        for c in unique_cells:
            m = self.m_c_.get(c, 0)
            v = m / (m + self.lambda_) if (m + self.lambda_) > 0 else 0.0
            self.beta_c_[c] = v * self.r_bar_c_.get(c, 0.0) + (1.0 - v) * self.psi_c_.get(c, 0.0)

        return self

    def _learn_lambda(
        self, residuals: np.ndarray, cell_ids: np.ndarray,
    ) -> float:
        """Simple heuristic: use median cell count as prior strength."""
        from statistics import median
        counts = [int(np.sum(cell_ids == c)) for c in set(cell_ids)]
        if counts:
            return float(median(counts))
        return 100.0

    def predict(self, cell: str) -> float:
        return self.beta_c_.get(cell, 0.0)

    def predict_batch(self, cell_ids: list[str]) -> np.ndarray:
        return np.array([self.beta_c_.get(c, 0.0) for c in cell_ids], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Formula 1: Gene Essentiality  μ̂_g = w₀ + Σ w_i · gene_feature_i(g)
# ═══════════════════════════════════════════════════════════════════════════════

class GeneEssentialityFormula:
    """Explicit linear formula for per-gene mean dependency.

    Learns: μ̂_g = w₀ + w₁·is_ribosomal(g) + w₂·is_OXPHOS(g) + ...
    where each coefficient w_i has a named biological interpretation.

    Features are automatically selected from G1 (gene static) and G2 (gene
    expression profile) — both available for ALL genes including cold-start.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model_: Ridge | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.coefficients_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(
        self,
        gene_features: pd.DataFrame,       # index=gene_symbol, columns=features
        gene_means: dict[str, float],       # gene → mean label
        feature_names: list[str] | None = None,
        verbose: bool = True,
    ) -> "GeneEssentialityFormula":
        """Fit Ridge regression to predict per-gene mean dependency."""
        common = sorted(set(gene_features.index) & set(gene_means.keys()))
        if len(common) < 10:
            warnings.warn(f"Only {len(common)} genes for gene formula fitting")
            return self

        X = gene_features.reindex(common).to_numpy(dtype=np.float64)
        y = np.array([gene_means[g] for g in common], dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)

        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = list(gene_features.columns)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Use RidgeCV for automatic alpha selection
        ridge_cv = RidgeCV(alphas=np.logspace(-2, 3, 20), store_cv_results=False)
        ridge_cv.fit(X_scaled, y)
        best_alpha = ridge_cv.alpha_

        self.model_ = Ridge(alpha=best_alpha)
        self.model_.fit(X_scaled, y)
        self.coefficients_ = self.model_.coef_
        self.intercept_ = float(self.model_.intercept_)

        if verbose:
            from sklearn.metrics import r2_score
            y_pred = self.model_.predict(X_scaled)
            r2 = r2_score(y, y_pred)
            n_nonzero = int(np.sum(np.abs(self.coefficients_) > 1e-6))
            print(f"  Gene formula: R²={r2:.4f}, {n_nonzero}/{len(self.feature_names_)} nonzero, "
                  f"α={best_alpha:.2f}")
        return self

    def predict(self, gene_features: pd.DataFrame) -> dict[str, float]:
        """Predict per-gene essentiality. Returns gene → μ̂_g dict."""
        if self.model_ is None:
            return {g: 0.0 for g in gene_features.index}

        genes = list(gene_features.index)
        X = gene_features.reindex(genes).to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler_.transform(X)
        preds = self.model_.predict(X_scaled).astype(np.float64)
        return dict(zip(genes, preds))

    def get_top_features(self, top_n: int = 20) -> list[tuple[str, float]]:
        """Return features with largest absolute coefficients (most influential)."""
        if self.coefficients_ is None:
            return []
        idx = np.argsort(-np.abs(self.coefficients_))[:top_n]
        return [(self.feature_names_[i], float(self.coefficients_[i])) for i in idx]

    def formula_str(self, top_n: int = 8) -> str:
        """Return human-readable formula string."""
        top = self.get_top_features(top_n)
        parts = [f"μ̂_g = {self.intercept_:.4f}"]
        for name, coef in top:
            sign = "+" if coef >= 0 else "-"
            parts.append(f"  {sign} {abs(coef):.4f} · {name}")
        if len(self.get_top_features(100)) > top_n:
            parts.append(f"  ... ({len(self.get_top_features(100))} total features)")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Formula 2: Cell Mitochondrial Vulnerability  β̂_c = v₀ + Σ v_i · cell_feature_i(c)
# ═══════════════════════════════════════════════════════════════════════════════

class CellVulnerabilityFormula:
    """Explicit linear formula for per-cell mitochondrial dependency.

    Learns: β̂_c = v₀ + v₁·OXPHOS_CI(c) + v₂·CELL_DEATH(c) + ...
    where each coefficient v_i measures how much that cell feature
    contributes to baseline mitochondrial vulnerability.
    """

    def __init__(self, alpha: float = 10.0):
        self.alpha = alpha
        self.model_: Ridge | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.coefficients_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(
        self,
        cell_features: pd.DataFrame,        # index=cell_line_id, columns=features
        cell_means: dict[str, float],        # cell → mean residual
        feature_names: list[str] | None = None,
    ) -> "CellVulnerabilityFormula":
        """Fit Ridge regression to predict per-cell mean residual."""
        common = sorted(set(cell_features.index) & set(cell_means.keys()))
        if len(common) < 10:
            warnings.warn(f"Only {len(common)} cells for cell formula fitting")
            return self

        X = cell_features.reindex(common).to_numpy(dtype=np.float64)
        y = np.array([cell_means[g] for g in common], dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)

        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = list(cell_features.columns)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        ridge_cv = RidgeCV(alphas=np.logspace(-2, 3, 20), store_cv_results=False)
        ridge_cv.fit(X_scaled, y)
        best_alpha = ridge_cv.alpha_

        self.model_ = Ridge(alpha=best_alpha)
        self.model_.fit(X_scaled, y)
        self.coefficients_ = self.model_.coef_
        self.intercept_ = float(self.model_.intercept_)

        from sklearn.metrics import r2_score
        y_pred = self.model_.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        n_nonzero = int(np.sum(np.abs(self.coefficients_) > 1e-6))
        print(f"  Cell formula: R²={r2:.4f}, {n_nonzero}/{len(self.feature_names_)} nonzero, "
              f"α={best_alpha:.2f}")
        return self

    def predict(self, cell_features: pd.DataFrame) -> dict[str, float]:
        """Predict per-cell vulnerability. Returns cell → β̂_c dict."""
        if self.model_ is None:
            return {c: 0.0 for c in cell_features.index}

        cells = list(cell_features.index)
        X = cell_features.reindex(cells).to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler_.transform(X)
        preds = self.model_.predict(X_scaled).astype(np.float64)
        return dict(zip(cells, preds))

    def get_top_features(self, top_n: int = 20) -> list[tuple[str, float]]:
        """Return features with largest absolute coefficients."""
        if self.coefficients_ is None:
            return []
        idx = np.argsort(-np.abs(self.coefficients_))[:top_n]
        return [(self.feature_names_[i], float(self.coefficients_[i])) for i in idx]

    def formula_str(self, top_n: int = 8) -> str:
        """Return human-readable formula string."""
        top = self.get_top_features(top_n)
        parts = [f"β̂_c = {self.intercept_:.4f}"]
        for name, coef in top:
            sign = "+" if coef >= 0 else "-"
            parts.append(f"  {sign} {abs(coef):.4f} · {name}")
        if len(self.get_top_features(100)) > top_n:
            parts.append(f"  ... ({len(self.get_top_features(100))} total features)")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Inductive Matrix Completion (IMC) Bilinear Interaction
#   r̂(c,g) = x_c^T W H^T y_g
#
# Models the double-residual r = y − μ̂_g − β̂_c as a bilinear form of
# cell features x_c (G3: ~115 dims) and gene features y_g (G1+G2: ~50 dims).
#
# Key properties:
#   - W ∈ R^{f_c × r}, H ∈ R^{f_g × r}, rank r ≪ min(f_c, f_g)
#   - Solved via alternating least squares (ALS): fix H → Ridge(W), fix W → Ridge(H)
#   - NATURALLY handles cold genes: prediction uses gene features y_g
#     which exist for ALL genes — no per-gene label-derived parameters
#   - Fully formula-interpretable: Z = WH^T gives named feature×feature weights
#
# References:
#   - Jain & Dhillon (2013): "Provable Inductive Matrix Completion"
#   - Natarajan & Dhillon (2014): IMC for gene–disease association prediction
#   - MM-LDA (2023): IMC + GAT for lncRNA–disease cold-start prediction
# ═══════════════════════════════════════════════════════════════════════════════


class IMCInteraction:
    """Inductive Matrix Completion bilinear interaction model.

    Models the double-residual r = y − μ̂_g − β̂_c as:

        r̂(c,g) = x_c^T W H^T y_g

    where:
      - x_c ∈ R^{f_c}: cell feature vector (G3 indicators, lineage, pathway PCA, etc.)
      - y_g ∈ R^{f_g}: gene feature vector (G1 module membership + G2 expression profile)
      - W ∈ R^{f_c × r}, H ∈ R^{f_g × r}: low-rank bilinear kernel factors
      - r: latent dimension (configurable, default 10)

    Solved via Alternating Least Squares (ALS):
      1. Fix H, solve Ridge for vec(W) — design matrix N × (f_c·r)
      2. Fix W, solve Ridge for vec(H) — design matrix N × (f_g·r)
      3. Repeat until convergence or max_iter.

    Cold-start: prediction uses gene features y_g directly — no per-gene
    latent factors needed. All parameters are in W and H (shared across
    all genes and cells).
    """

    def __init__(
        self,
        rank: int = 10,
        lambda_w: float = 1.0,
        lambda_h: float = 1.0,
        max_iter: int = 30,
        tol: float = 1e-4,
        random_state: int = 42,
    ):
        self.rank = rank
        self.lambda_w = lambda_w
        self.lambda_h = lambda_h
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

        # Learned parameters
        self.W_: np.ndarray | None = None       # (f_c, r)
        self.H_: np.ndarray | None = None       # (f_g, r)
        self.cell_feature_names_: list[str] = []
        self.gene_feature_names_: list[str] = []
        self.Z_: np.ndarray | None = None       # (f_c, f_g) = W H^T
        self.train_r2_: float = 0.0
        self.iterations_: int = 0

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(
        self,
        X_cell: np.ndarray,                # (N, f_c) cell feature matrix
        X_gene: np.ndarray,                # (N, f_g) gene feature matrix
        residuals: np.ndarray,             # (N,) double residual
        cell_feature_names: list[str] | None = None,
        gene_feature_names: list[str] | None = None,
        verbose: bool = True,
    ) -> "IMCInteraction":
        """Fit IMC bilinear model via Alternating Least Squares.

        Args:
            X_cell: (N, f_c) cell feature matrix (float32/float64).
            X_gene: (N, f_g) gene feature matrix.
            residuals: (N,) double residual y − μ̂_g − β̂_c.
            cell_feature_names: optional names for cell feature columns.
            gene_feature_names: optional names for gene feature columns.
        """
        from sklearn.linear_model import Ridge

        N, f_c = X_cell.shape
        _, f_g = X_gene.shape
        r = min(self.rank, min(f_c, f_g))

        if cell_feature_names is not None:
            self.cell_feature_names_ = list(cell_feature_names)
        if gene_feature_names is not None:
            self.gene_feature_names_ = list(gene_feature_names)

        y = residuals.astype(np.float64)
        Xc = X_cell.astype(np.float64)
        Xg = X_gene.astype(np.float64)

        # Initialize W and H randomly
        rng = np.random.RandomState(self.random_state)
        self.W_ = rng.randn(f_c, r).astype(np.float64) * 0.01
        self.H_ = rng.randn(f_g, r).astype(np.float64) * 0.01

        prev_loss = float("inf")

        for iteration in range(self.max_iter):
            # ── Step 1: Fix H, solve for W ──
            # Prediction: ŷ_i = x_{c_i}^T W (H^T y_{g_i})
            # = trace(W^T x_{c_i} (H^T y_{g_i})^T)
            # = vec(W)^T (x_{c_i} ⊗ (H^T y_{g_i}))
            # Design matrix X_w: (N, f_c*r)
            z_i = Xg @ self.H_  # (N, r)
            X_w = np.zeros((N, f_c * r), dtype=np.float64)
            for a in range(f_c):
                col_start = a * r
                X_w[:, col_start:col_start + r] = Xc[:, a:a+1] * z_i

            ridge_w = Ridge(alpha=self.lambda_w, fit_intercept=False, solver="lsqr")
            ridge_w.fit(X_w, y)
            self.W_ = ridge_w.coef_.reshape(f_c, r)

            # ── Step 2: Fix W, solve for H ──
            # Prediction: ŷ_i = y_{g_i}^T H (W^T x_{c_i})
            # = vec(H)^T (y_{g_i} ⊗ (W^T x_{c_i}))
            u_i = Xc @ self.W_  # (N, r)
            X_h = np.zeros((N, f_g * r), dtype=np.float64)
            for a in range(f_g):
                col_start = a * r
                X_h[:, col_start:col_start + r] = Xg[:, a:a+1] * u_i

            ridge_h = Ridge(alpha=self.lambda_h, fit_intercept=False, solver="lsqr")
            ridge_h.fit(X_h, y)
            self.H_ = ridge_h.coef_.reshape(f_g, r)

            # ── Convergence check ──
            pred = self._predict_from_features(Xc, Xg)
            loss = np.mean((y - pred) ** 2)
            r2 = 1.0 - np.sum((y - pred) ** 2) / max(np.sum((y - np.mean(y)) ** 2), 1e-12)

            if abs(prev_loss - loss) < self.tol * max(loss, 1e-12):
                if verbose:
                    print(f"  IMC converged at iter {iteration+1}: loss={loss:.6f}, R²={r2:.4f}")
                break

            prev_loss = loss

            if verbose and (iteration == 0 or (iteration + 1) % 5 == 0):
                print(f"  IMC iter {iteration+1}: loss={loss:.6f}, R²={r2:.4f}")

        self.iterations_ = iteration + 1
        self.train_r2_ = r2
        self.Z_ = self.W_ @ self.H_.T  # (f_c, f_g) full interaction kernel

        return self

    # ── Predict ─────────────────────────────────────────────────────────────

    def predict(
        self,
        X_cell: np.ndarray,                # (N, f_c)
        X_gene: np.ndarray,                # (N, f_g)
    ) -> np.ndarray:
        """Predict interaction values: r̂ = diag(X_cell W H^T X_gene^T).

        Equivalent to: ŷ_i = x_{c_i}^T W H^T y_{g_i}
        Efficiently computed as: sum((X_cell @ W) * (X_gene @ H), axis=1)
        """
        return self._predict_from_features(
            X_cell.astype(np.float64),
            X_gene.astype(np.float64),
        ).astype(np.float32)

    def _predict_from_features(
        self,
        Xc: np.ndarray,   # (N, f_c) float64
        Xg: np.ndarray,   # (N, f_g) float64
    ) -> np.ndarray:
        """Core prediction: ŷ_i = x_i^T W H^T y_i = sum((x_i^T W) * (y_i^T H))."""
        if self.W_ is None or self.H_ is None:
            return np.zeros(len(Xc), dtype=np.float64)
        left = Xc @ self.W_    # (N, r)
        right = Xg @ self.H_   # (N, r)
        return np.sum(left * right, axis=1)

    # ── Formula & Interpretation ───────────────────────────────────────────

    def formula_str(self, top_n: int = 10) -> str:
        """Return human-readable formula string."""
        if self.Z_ is None:
            return "r̂(c,g) = 0"

        lines = [
            f"r̂(c,g) = x_c^T W H^T y_g",
            f"  rank = {self.rank}, f_c = {self.W_.shape[0]}, f_g = {self.H_.shape[0]}",
            f"  Training R² = {self.train_r2_:.4f} (on double residual)",
            f"  ALS iterations: {self.iterations_}",
            "",
            "Top feature×feature interactions (|Z_ij|):",
        ]

        # Find top |Z_ij| entries
        Z_abs = np.abs(self.Z_)
        flat_indices = np.argsort(-Z_abs.ravel())
        cell_names = (self.cell_feature_names_
                      if self.cell_feature_names_
                      else [f"cell_feat_{i}" for i in range(self.Z_.shape[0])])
        gene_names = (self.gene_feature_names_
                      if self.gene_feature_names_
                      else [f"gene_feat_{j}" for j in range(self.Z_.shape[1])])

        for idx in flat_indices[:top_n]:
            i = idx // self.Z_.shape[1]
            j = idx % self.Z_.shape[1]
            z_val = self.Z_[i, j]
            cn = cell_names[i] if i < len(cell_names) else f"c{i}"
            gn = gene_names[j] if j < len(gene_names) else f"g{j}"
            lines.append(f"  Z[{cn}, {gn}] = {z_val:+.4f}")

        return "\n".join(lines)

    def get_top_interactions(self, top_n: int = 20) -> list[tuple[str, str, float]]:
        """Return top |Z_ij| entries as (cell_feature, gene_feature, weight)."""
        if self.Z_ is None:
            return []

        Z_abs = np.abs(self.Z_)
        flat_indices = np.argsort(-Z_abs.ravel())
        cell_names = (self.cell_feature_names_
                      if self.cell_feature_names_
                      else [f"cell_feat_{i}" for i in range(self.Z_.shape[0])])
        gene_names = (self.gene_feature_names_
                      if self.gene_feature_names_
                      else [f"gene_feat_{j}" for j in range(self.Z_.shape[1])])

        results = []
        for idx in flat_indices[:top_n]:
            i = idx // self.Z_.shape[1]
            j = idx % self.Z_.shape[1]
            results.append((cell_names[i], gene_names[j], float(self.Z_[i, j])))
        return results

# ═══════════════════════════════════════════════════════════════════════════════
# Structured Biological Interaction Model
#   I(c,g) = Σ_k α_k·I_k(c)·M_k(g)                          [Module×Indicator]
#          + Σ_k β_k·I_k(c)·M_k(g)·z_{c,g}                   [Expr-Modulated]
#          + γ₁·z_{c,g}·w(g)                                 [Evidence-Weighted]
#          + γ₂·p_{c,g}·w(g)                                 [Percentile-Weighted]
#          + Σ_l Σ_k δ_{l,k}·L_l(c)·M_k(g)                   [Lineage×Module]
#
# All features are available for cold genes (from MitoCarta annotations,
# Problem 1 indicators, expression data, and cell metadata).
# ═══════════════════════════════════════════════════════════════════════════════


class StructuredInteraction:
    """Biologically-structured gene×cell interaction model.

    Models the double-residual r = y − μ̂_g − β̂_c with explicit
    biological interaction terms derived from mitochondrial pathway
    structure and expression data.

    All features are cold-gene safe — gene module membership comes from
    MitoCarta3.0 annotations (available for all genes), cell indicators
    from Problem 1 (available for all cells), and expression z-scores
    from the provided expression matrix.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        include_lineage: bool = True,
        random_state: int = 42,
    ):
        self.alpha = alpha
        self.include_lineage = include_lineage
        self.random_state = random_state

        # Learned
        self.model_: Ridge | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.coefficients_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.train_r2_: float = 0.0

    # ── Feature Construction ──────────────────────────────────────────────

    def _build_features(
        self,
        cell_df: pd.DataFrame,
        gene_df: pd.DataFrame,
        pair_z: np.ndarray,
        pair_pct: np.ndarray,
    ) -> tuple[np.ndarray, list[str]]:
        """Build structured interaction feature matrix.

        Feature groups:
          A. Expression main effects (4): z, |z|, max(0,z), min(0,z)
          B. Expression × Module (14): z_cg · M_k(g)
          C. Expression × Indicator (14): z_cg · I_k(c)
          D. Module × Indicator (14): M_k(g) · I_k(c)
          E. Evidence-weighted (2): z·w(g), |z|·w(g)
          F. Percentile (2): pct, pct·w(g)
          G. Lineage × Module (optional, n_lin × 14)

        Total: ~50 (without lineage) or ~450 (with lineage).
        All features are cold-gene safe.
        """
        N = len(cell_df)
        n_modules = 14
        features: dict[str, np.ndarray] = {}

        # ── Extract key columns ──
        ind_cols = [c for c in cell_df.columns if c.startswith("cell_indicator_")]
        if not ind_cols:
            ind_arr = np.zeros((N, 0), dtype=np.float64)
        else:
            ind_arr = cell_df[ind_cols].to_numpy(dtype=np.float64)
        n_ind = ind_arr.shape[1]
        n_interact = min(n_modules, n_ind)

        mod_cols = [f"gene_module_{k:02d}" for k in range(n_modules)
                   if f"gene_module_{k:02d}" in gene_df.columns]
        n_modules_found = len(mod_cols)
        mod_arr = gene_df[mod_cols].to_numpy(dtype=np.float64) if mod_cols else \
            np.zeros((N, 0), dtype=np.float64)

        ew_col = "gene_evidence_weight"
        if ew_col in gene_df.columns:
            ew = gene_df[ew_col].to_numpy(dtype=np.float64)
        else:
            ew = np.ones(N, dtype=np.float64)

        z = pair_z.astype(np.float64)
        pct = pair_pct.astype(np.float64)
        z_abs = np.abs(z)
        z_pos = np.maximum(z, 0)
        z_neg = np.minimum(z, 0)

        # ── A. Expression main effects (4) ──
        features["z_cg"] = z
        features["z_abs"] = z_abs
        features["z_pos"] = z_pos
        features["z_neg"] = z_neg

        # ── B. Expression × Module (z_cg · M_k(g)) ──
        for k in range(n_modules_found):
            features[f"z_x_mod{k:02d}"] = z * mod_arr[:, k]

        # ── C. Expression × Indicator (z_cg · I_k(c)) ──
        for k in range(n_interact):
            features[f"z_x_ind{k:02d}"] = z * ind_arr[:, k]

        # ── D. Module × Indicator (M_k(g) · I_k(c)) ──
        for k in range(n_interact):
            features[f"mod{k:02d}_x_ind"] = mod_arr[:, k] * ind_arr[:, k]

        # ── E. Evidence-weighted expression (2) ──
        features["z_x_evidence"] = z * ew
        features["z_abs_x_evidence"] = z_abs * ew

        # ── F. Expression percentile (2) ──
        features["expr_percentile"] = pct
        features["pct_x_evidence"] = pct * ew

        # ── G. Lineage × Module (optional) ──
        if self.include_lineage:
            lin_cols = [c for c in cell_df.columns
                       if c.startswith("cell_lineage_onehot_")]
            if lin_cols:
                lin_arr = cell_df[lin_cols].to_numpy(dtype=np.float64)
                lin_names = [c.replace("cell_lineage_onehot_", "")
                           for c in lin_cols]
                n_lin = lin_arr.shape[1]
                for l in range(n_lin):
                    for k in range(n_modules_found):
                        features[f"lin_{lin_names[l]}_x_mod{k:02d}"] = (
                            lin_arr[:, l] * mod_arr[:, k]
                        )

        # Build matrix
        feature_names = list(features.keys())
        X = np.column_stack([features[name] for name in feature_names])
        return X, feature_names

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(
        self,
        cell_df: pd.DataFrame,
        gene_df: pd.DataFrame,
        pair_z: np.ndarray,
        pair_pct: np.ndarray,
        residuals: np.ndarray,
        verbose: bool = True,
    ) -> "StructuredInteraction":
        """Fit structured interaction model on double residuals.

        Uses RidgeCV for automatic alpha selection. All features are
        standardized before fitting.
        """
        X, self.feature_names_ = self._build_features(
            cell_df, gene_df, pair_z, pair_pct,
        )
        y = residuals.astype(np.float64)

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        ridge_cv = RidgeCV(
            alphas=np.logspace(-2, 3, 20), store_cv_results=False,
        )
        ridge_cv.fit(X_scaled, y)

        self.model_ = Ridge(alpha=ridge_cv.alpha_)
        self.model_.fit(X_scaled, y)
        self.coefficients_ = self.model_.coef_
        self.intercept_ = float(self.model_.intercept_)

        from sklearn.metrics import r2_score
        y_pred = self.model_.predict(X_scaled)
        self.train_r2_ = r2_score(y, y_pred)

        if verbose:
            n_nonzero = int(np.sum(np.abs(self.coefficients_) > 1e-6))
            print(f"  StructuredInteraction: R²={self.train_r2_:.4f}, "
                  f"{n_nonzero}/{len(self.feature_names_)} nonzero coeffs, "
                  f"α={ridge_cv.alpha_:.2f}")

        return self

    # ── Predict ────────────────────────────────────────────────────────────

    def predict(
        self,
        cell_df: pd.DataFrame,
        gene_df: pd.DataFrame,
        pair_z: np.ndarray,
        pair_pct: np.ndarray,
    ) -> np.ndarray:
        """Predict interaction values for (cell, gene) pairs."""
        if self.model_ is None:
            return np.zeros(len(cell_df), dtype=np.float32)

        X, _ = self._build_features(cell_df, gene_df, pair_z, pair_pct)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled).astype(np.float32)

    # ── Interpretation ─────────────────────────────────────────────────────

    def get_top_interactions(self, top_n: int = 20) -> list[tuple[str, float]]:
        """Return interaction terms with largest |coefficient|."""
        if self.coefficients_ is None:
            return []
        idx = np.argsort(-np.abs(self.coefficients_))[:top_n]
        return [(self.feature_names_[i], float(self.coefficients_[i]))
                for i in idx]

    def formula_str(self, top_n: int = 15) -> str:
        """Return human-readable formula string."""
        top = self.get_top_interactions(top_n)
        lines = [
            "I(c,g) = Σ_k α_k·z_cg·M_k(g)  [Expr × Module]",
            "       + Σ_k β_k·z_cg·I_k(c)  [Expr × Indicator]",
            "       + Σ_k γ_k·M_k(g)·I_k(c)  [Module × Indicator]",
            "       + δ₁·z_cg·w(g) + δ₂·|z_cg|·w(g)  [Evidence-Weighted]",
            "       + η₁·z_cg + η₂·|z_cg| + η₃·z⁺ + η₄·z⁻  [Expr Main]",
            "       + θ₁·p_cg + θ₂·p_cg·w(g)  [Percentile]",
        ]
        if self.include_lineage:
            lines.append("       + Σ_l Σ_k ζ_{l,k}·L_l(c)·M_k(g)  [Lineage × Module]")
        lines.extend([
            f"  n_features = {len(self.feature_names_)}, "
            f"Training R² = {self.train_r2_:.4f} (on double residual)",
            "",
            "Top interaction terms:",
        ])
        for name, coef in top:
            sign = "+" if coef >= 0 else "-"
            lines.append(f"  {sign} {abs(coef):.4f} · {name}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Hybrid SVD + Gene-Similarity CF Interaction
#
#   Warm genes: I(c,g) = Σ_k σ_k · u_k(c) · v_k(g)   [SVD, R²≈0.42]
#   Cold genes: I(c,g) = Σ_w sim(g,w) · I_SVD(c,w)    [CF transfer]
#
# SVD captures rich low-rank structure for genes with training labels.
# For cold genes, pathway-similarity-weighted transfer from warm neighbors
# provides cell-specific predictions instead of cell-invariant Φ(g).
# ═══════════════════════════════════════════════════════════════════════════════


class HybridInteraction:
    """SVD bilinear interaction with cold-gene pathway-CF transfer.

    Warm genes: I(c,g) = Σ_k σ_k · u_k(c) · v_k(g)
      — SVD on the double-residual matrix, capturing low-rank gene×cell
        interaction structure with high fidelity (R²≈0.42).

    Cold genes: I(c,g) = Σ_{w} sim(g,w) · I_SVD(c,w) / Σ sim(g,w)
      — Pathway-similarity-weighted average of warm neighbors' SVD
        interactions, providing cell-specific cold gene predictions.

    The gene similarity uses MitoCarta3.0's 149-pathway annotations (PW140)
    combined with co-expression correlation, both available for all genes.
    """

    def __init__(
        self,
        n_components: int = 30,
        cf_knn: int = 30,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.cf_knn = cf_knn
        self.random_state = random_state

        # SVD results
        self.U_: np.ndarray | None = None       # (n_cells, K) cell factors
        self.V_: np.ndarray | None = None       # (n_warm_genes, K) gene factors
        self.singular_values_: np.ndarray | None = None
        self.cell_index_: pd.Index | None = None
        self.gene_index_: pd.Index | None = None
        self.global_residual_mean_: float = 0.0

        # Cold gene transfer
        self.cold_gene_factors_: dict[str, np.ndarray] = {}  # gene → (K,)
        self.cold_to_warm_: dict[str, list[tuple[str, float]]] = {}

        # Metadata
        self.train_r2_: float = 0.0
        self.n_components_used_: int = 0

    # ── Fit ────────────────────────────────────────────────────────────────

    def fit(
        self,
        residuals: np.ndarray,
        cell_ids: np.ndarray,
        gene_ids: np.ndarray,
        cold_genes: set[str],
        gene_meta: pd.DataFrame,
        pathway_meta: pd.DataFrame,
        expression: pd.DataFrame,
        g1_features: pd.DataFrame,
        g2_features: pd.DataFrame,
        verbose: bool = True,
    ) -> "HybridInteraction":
        """Fit SVD on double residual, build CF transfer for cold genes.

        Args:
            residuals: (N,) double residual y − μ̂_g − β̂_c.
            cell_ids, gene_ids: (N,) identifiers.
            cold_genes: set of gene symbols with zero training labels.
            gene_meta, pathway_meta: metadata DataFrames.
            expression: N_cells × P_genes z-scored expression DataFrame.
            g1_features, g2_features: gene-level feature DataFrames.
        """
        from sklearn.decomposition import TruncatedSVD

        # ── Build residual matrix ──
        cells = sorted(set(cell_ids))
        warm_genes = sorted(set(gene_ids) - cold_genes)
        self.cell_index_ = pd.Index(cells)
        self.gene_index_ = pd.Index(warm_genes)

        cell_to_idx = {c: i for i, c in enumerate(cells)}
        gene_to_idx = {g: i for i, g in enumerate(warm_genes)}

        n_cells = len(cells)
        n_warm = len(warm_genes)
        self.global_residual_mean_ = float(residuals.mean())

        R = np.full((n_cells, n_warm), self.global_residual_mean_,
                    dtype=np.float64)
        for i in range(len(residuals)):
            c = cell_ids[i]
            g = gene_ids[i]
            if c in cell_to_idx and g in gene_to_idx:
                R[cell_to_idx[c], gene_to_idx[g]] = residuals[i]

        # Center
        R -= R.mean()

        # SVD
        k = min(self.n_components, min(n_cells, n_warm) - 1)
        svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        self.U_ = svd.fit_transform(R).astype(np.float64)       # (n_cells, K)
        self.V_ = svd.components_.T.astype(np.float64)           # (n_warm, K)
        self.singular_values_ = svd.singular_values_.astype(np.float64)
        self.n_components_used_ = k

        # Training R² on observed residual entries
        pred = np.zeros(len(residuals), dtype=np.float64)
        for i in range(len(residuals)):
            c = cell_ids[i]
            g = gene_ids[i]
            if c in cell_to_idx and g in gene_to_idx:
                pred[i] = float(np.dot(self.U_[cell_to_idx[c]],
                                       self.V_[gene_to_idx[g]]))
            else:
                pred[i] = self.global_residual_mean_
        ss_res = np.sum((residuals - pred) ** 2)
        ss_tot = max(np.sum((residuals - residuals.mean()) ** 2), 1e-12)
        self.train_r2_ = 1.0 - ss_res / ss_tot

        if verbose:
            var_explained = float(np.sum(svd.explained_variance_ratio_))
            print(f"  SVD: K={k}, R²={self.train_r2_:.4f} on observed residual, "
                  f"cumulative variance={var_explained:.3f}")

        # ── Build cold gene CF transfer ──
        if cold_genes:
            self._build_cold_transfer(
                cold_genes, warm_genes, cells, cell_to_idx, gene_to_idx,
                gene_meta, pathway_meta, expression, R,
                g1_features, g2_features, verbose,
            )

        return self

    def _build_cold_transfer(
        self,
        cold_genes: set[str],
        warm_genes: list[str],
        cells: list[str],
        cell_to_idx: dict[str, int],
        gene_to_idx: dict[str, int],
        gene_meta: pd.DataFrame,
        pathway_meta: pd.DataFrame,
        expression: pd.DataFrame,
        R: np.ndarray,
        g1_features: pd.DataFrame,
        g2_features: pd.DataFrame,
        verbose: bool,
    ) -> None:
        """Build Ridge(gene_features → V_factors) for cold gene SVD factor prediction.

        Replaces the KNN-based weighted average with Ridge regression:
          V̂_k(cold_g) = Ridge_k(gene_features(cold_g))

        This uses ALL warm genes to learn the feature→V_factor mapping, not just
        K neighbors. Ridge naturally regularizes: genes with similar features
        (same pathway, module, expression profile) get similar V factors.
        This is equivalent to graph-regularized SVD where the graph Laplacian is
        implicitly defined by the gene feature similarity kernel.

        Reference: EMF (Fan et al., 2020), Macau (Zakeri et al., 2018)
        """
        from sklearn.linear_model import RidgeCV
        from sklearn.neighbors import NearestNeighbors

        # Build PW140 gene features
        from .baselines import build_pw140_membership_features
        pw140 = build_pw140_membership_features(gene_meta, pathway_meta)

        # Combine pathway + expression profile + coexpression features
        gene_feats = g1_features.join(g2_features, how="inner")
        pw140_aligned = pw140.reindex(gene_feats.index).fillna(0.0)

        # Build feature matrix for warm genes
        warm_list = sorted(set(warm_genes) & set(gene_feats.index)
                          & set(pw140_aligned.index))
        cold_list = sorted(cold_genes & set(gene_feats.index)
                          & set(pw140_aligned.index))

        if not warm_list or not cold_list:
            if verbose:
                print(f"  CF: insufficient common genes for transfer "
                      f"(warm={len(warm_list)}, cold={len(cold_list)})")
            return

        # Feature matrix: concatenate PW140 + G1 + G2
        X_warm_pw = pw140_aligned.reindex(warm_list).fillna(0).to_numpy(dtype=np.float64)
        X_warm_gf = gene_feats.reindex(warm_list).fillna(0).to_numpy(dtype=np.float64)
        X_warm = np.column_stack([X_warm_pw, X_warm_gf])
        X_warm = np.nan_to_num(X_warm, nan=0.0)

        X_cold_pw = pw140_aligned.reindex(cold_list).fillna(0).to_numpy(dtype=np.float64)
        X_cold_gf = gene_feats.reindex(cold_list).fillna(0).to_numpy(dtype=np.float64)
        X_cold = np.column_stack([X_cold_pw, X_cold_gf])
        X_cold = np.nan_to_num(X_cold, nan=0.0)

        # ── Method 1: Ridge regression for V factor prediction ──
        # For each SVD component, train Ridge(gene_features → V_k)
        n_comp = self.n_components_used_
        V_warm_matrix = np.column_stack([
            self.V_[gene_to_idx[g]] for g in warm_list
        ]).T  # (n_warm, K)

        ridge_r2s = []
        for k in range(n_comp):
            v_k = V_warm_matrix[:, k]
            ridge_k = RidgeCV(alphas=(0.01, 0.1, 1.0, 10.0, 100.0),
                            store_cv_results=False)
            ridge_k.fit(X_warm, v_k)
            pred_k = ridge_k.predict(X_warm)
            ss_res = np.sum((v_k - pred_k) ** 2)
            ss_tot = np.sum((v_k - np.mean(v_k)) ** 2)
            r2_k = 1.0 - ss_res / max(ss_tot, 1e-12)
            ridge_r2s.append(r2_k)

            # Predict V factor for cold genes
            v_cold_k = ridge_k.predict(X_cold)
            for i, cold_g in enumerate(cold_list):
                if cold_g not in self.cold_gene_factors_:
                    self.cold_gene_factors_[cold_g] = np.zeros(n_comp, dtype=np.float64)
                self.cold_gene_factors_[cold_g][k] = v_cold_k[i]

        mean_ridge_r2 = np.mean(ridge_r2s) if ridge_r2s else 0.0

        # ── Method 2: KNN similarity for cold→warm map (backup) ──
        k_eff = min(self.cf_knn, len(warm_list))
        nbrs = NearestNeighbors(n_neighbors=k_eff, metric="cosine")
        nbrs.fit(X_warm)
        distances, indices = nbrs.kneighbors(X_cold)

        for i, cold_g in enumerate(cold_list):
            sims = []
            for j in range(k_eff):
                w_g = warm_list[indices[i, j]]
                sim = max(1.0 - distances[i, j], 1e-6)
                sims.append((w_g, sim))
            total = sum(s for _, s in sims)
            if total > 0:
                self.cold_to_warm_[cold_g] = [(g, s / total) for g, s in sims]

        # ── Fallback: for genes with no Ridge prediction, use KNN weighted avg ──
        for cold_g in cold_list:
            if cold_g not in self.cold_gene_factors_:
                # Fallback to KNN weighted average
                sims = self.cold_to_warm_.get(cold_g, [])
                v_cold = np.zeros(n_comp, dtype=np.float64)
                weight_sum = 0.0
                for w_g, sim in sims:
                    if w_g in gene_to_idx:
                        v_cold += sim * self.V_[gene_to_idx[w_g]]
                        weight_sum += sim
                if weight_sum > 0:
                    v_cold /= weight_sum
                self.cold_gene_factors_[cold_g] = v_cold

        if verbose:
            n_with_cf = len(self.cold_to_warm_)
            n_ridge = sum(1 for g in cold_list
                         if g in self.cold_gene_factors_)
            print(f"  CF: {n_with_cf}/{len(cold_list)} cold genes have warm "
                  f"neighbors (K={k_eff})")
            print(f"  Ridge V-factor prediction: mean R²={mean_ridge_r2:.4f} "
                  f"(best={max(ridge_r2s):.4f}, worst={min(ridge_r2s):.4f}) "
                  f"across {n_comp} components")

    # ── Predict ────────────────────────────────────────────────────────────

    def predict(
        self,
        cell_ids: np.ndarray,
        gene_ids: np.ndarray,
    ) -> np.ndarray:
        """Predict SVD interaction values for (cell, gene) pairs.

        Warm genes: direct SVD dot product U_c · V_g.
        Cold genes: dot product U_c · V̂_g (predicted from warm neighbors).
        """
        n = len(cell_ids)
        pred = np.full(n, self.global_residual_mean_, dtype=np.float32)

        if self.U_ is None or self.V_ is None:
            return pred

        cell_to_idx = {c: i for i, c in enumerate(self.cell_index_)}
        gene_to_idx = {g: i for i, g in enumerate(self.gene_index_)}

        for i in range(n):
            c = cell_ids[i]
            g = gene_ids[i]
            ci = cell_to_idx.get(c)
            if ci is None:
                continue

            if g in gene_to_idx:
                # Warm gene: direct SVD
                gi = gene_to_idx[g]
                pred[i] = float(np.dot(self.U_[ci], self.V_[gi]))
            elif g in self.cold_gene_factors_:
                # Cold gene: predicted V factor
                pred[i] = float(np.dot(self.U_[ci],
                                       self.cold_gene_factors_[g]))

        return pred

    # ── Interpretation ─────────────────────────────────────────────────────

    def formula_str(self, top_n: int = 10) -> str:
        """Return human-readable formula string."""
        lines = [
            "I(c,g) = Σ_k σ_k · u_k(c) · v_k(g)",
            f"  Components: {self.n_components_used_}",
            f"  Training R² = {self.train_r2_:.4f} (on double residual)",
        ]
        if self.singular_values_ is not None:
            sv = self.singular_values_
            var_exp = sv ** 2 / max(np.sum(sv ** 2), 1e-12)
            lines.append(f"  Top singular values: "
                        f"{', '.join(f'{sv[i]:.4f}' for i in range(min(5, len(sv))))}")
            lines.append(f"  Variance explained: "
                        f"{', '.join(f'{var_exp[i]:.3f}' for i in range(min(5, len(var_exp))))}")
        if self.cold_to_warm_:
            lines.append(f"  Cold genes with CF transfer: {len(self.cold_to_warm_)}")
        return "\n".join(lines)

    def get_top_gene_loadings(self, component: int = 0, top_n: int = 20
                             ) -> list[tuple[str, float]]:
        """Return genes with largest |loading| in a given SVD component."""
        if self.V_ is None or component >= self.V_.shape[1]:
            return []
        idx = np.argsort(-np.abs(self.V_[:, component]))[:top_n]
        return [(self.gene_index_[i], float(self.V_[i, component]))
                for i in idx]


# Full Formula Training Pipeline (Empirical Bayes)
# ═══════════════════════════════════════════════════════════════════════════════

def train_formula_models(
    y: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    gene_static_features: pd.DataFrame,
    gene_expr_profile_features: pd.DataFrame,
    cell_features: pd.DataFrame,
    cold_genes: set[str] | None = None,
    config: dict[str, Any] | None = None,
    expression: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Train formula-based models with empirical Bayes smooth transition.

    Sequential fitting (NO hard warm/cold distinction):
      1. ShrinkageGeneFormula → μ̂_g = w_g·x̄_g + (1-w_g)·Φ(g)
      2. ShrinkageCellFormula → β̂_c = v_c·r̄_c + (1-v_c)·Ψ(c)
      3. StructuredInteraction → I(c,g) with biological interaction terms
      4. (Optional) IMC residual on remaining signal

    Args:
        y: label array.
        cell_ids, gene_ids: identifiers for each row.
        gene_static_features: G1 features indexed by gene.
        gene_expr_profile_features: G2 features indexed by gene.
        cell_features: G3 features indexed by cell.
        cold_genes: set of cold gene symbols (n_g=0).
        config: configuration dict.
        expression: N_cells × P_genes expression DataFrame (z-scored).
                    Required for StructuredInteraction pair features.

    Returns:
        dict with shrinkage models, interaction models, and metadata.
    """
    if config is None:
        config = {}
    fm_cfg = config.get("prediction", {}).get("formula", {})
    if cold_genes is None:
        cold_genes = set()

    print("\n" + "=" * 70)
    print("FORMULA-BASED PREDICTION (Empirical Bayes Smooth Transition)")
    print("=" * 70)

    n = len(y)
    unique_cells = sorted(set(cell_ids))
    unique_genes = sorted(set(gene_ids))

    # ── Build gene feature matrix ──
    gene_feat_df = gene_static_features.join(gene_expr_profile_features, how="inner")

    # ── Step 1: Shrinkage Gene Formula (empirical Bayes) ──
    print("\n[Step 1] Empirical Bayes Gene Shrinkage")
    print("  μ̂_g = w_g·x̄_g + (1-w_g)·Φ(g),  w_g = n_g/(n_g+λ)")
    labels_df = pd.DataFrame({
        "cell_line_id": cell_ids,
        "perturbation_gene": gene_ids,
        "label": y,
    })
    shrink_gene = ShrinkageGeneFormula(
        lambda_grid=fm_cfg.get("lambda_grid", [1, 3, 10, 30, 100, 300, 1000, 3000]),
        random_state=42,
    )
    shrink_gene.fit(gene_feat_df, labels_df)
    mu_g = shrink_gene.mu_g_  # shrunk gene essentiality (all genes)

    # ── Step 2: Shrinkage Cell Formula (empirical Bayes) ──
    print("\n[Step 2] Empirical Bayes Cell Shrinkage")
    print("  β̂_c = v_c·r̄_c + (1-v_c)·Ψ(c),  v_c = m_c/(m_c+λ)")
    r1 = y.copy().astype(np.float64)
    for i in range(n):
        r1[i] -= shrink_gene.predict(gene_ids[i])

    shrink_cell = ShrinkageCellFormula(
        lambda_grid=fm_cfg.get("cell_lambda_grid", [1, 3, 10, 30, 100, 300, 1000]),
    )
    shrink_cell.fit(cell_features, r1, cell_ids, gene_ids)
    beta_c = shrink_cell.beta_c_

    # ── Step 3: Double residual ──
    print("\n[Step 3] Variance Decomposition")
    r2 = r1.copy()
    for i in range(n):
        r2[i] -= shrink_cell.predict(cell_ids[i])

    mu_var = np.var([shrink_gene.predict(g) for g in gene_ids])
    beta_var = np.var([shrink_cell.predict(c) for c in cell_ids])
    print(f"    σ²(y)           = {np.var(y):.6f}")
    print(f"    σ²(μ̂_g)         = {mu_var:.6f}")
    print(f"    σ²(β̂_c)         = {beta_var:.6f}")
    print(f"    σ²(r₂)          = {np.var(r2):.6f} "
          f"({100 * np.var(r2) / max(np.var(y), 1e-12):.1f}% remaining)")

    # ── Build feature matrices for interaction ──
    cell_feat_df_aligned = cell_features.reindex(
        pd.Index(cell_ids)).fillna(0.0)
    gene_feat_df_aligned = gene_feat_df.reindex(
        pd.Index(gene_ids)).fillna(0.0)

    # ── Step 4: Hybrid SVD + Gene-Similarity CF Interaction ──
    hybrid_cfg = fm_cfg.get("hybrid_interaction", {})
    print(f"\n[Step 4] Hybrid SVD + Gene-Similarity CF Interaction")
    print(f"  I(c,g) = Σ_k σ_k · u_k(c) · v_k(g)  [SVD, warm genes]")
    print(f"  I(c,g) = Σ_w sim(g,w) · I_SVD(c,w)  [CF, cold genes]")

    # Load metadata for CF
    from pathlib import Path
    from ..utils import load_config as _load_config
    data_dir = Path(config.get("paths", {}).get("data_dir", "数据文件"))
    gene_meta = pd.read_csv(data_dir / "metadata" / "gene_metadata.csv")
    pathway_meta = pd.read_csv(data_dir / "metadata" / "pathway_metadata.csv")

    hybrid_interaction = HybridInteraction(
        n_components=hybrid_cfg.get("n_components", 30),
        cf_knn=hybrid_cfg.get("cf_knn", 30),
        random_state=42,
    )
    hybrid_interaction.fit(
        r2, cell_ids, gene_ids, cold_genes,
        gene_meta, pathway_meta,
        expression if expression is not None else pd.DataFrame(),
        gene_static_features, gene_expr_profile_features,
    )

    # ── Step 5: Multi-Output Gene Profile Predictor ──
    profile_cfg = fm_cfg.get("profile", {})
    profile_predictor = None
    if profile_cfg.get("enabled", False) and cold_genes:
        print(f"\n[Step 5] Multi-Output Gene Profile Predictor")
        print(f"  ŷ_profile(g,c) = Σ_k PC_score_k(g) · PC_loading_k(c)")
        from .profile_predictor import MultiOutputGeneProfile
        # Build enriched gene features: G1 + G2 + PW140
        from pathlib import Path
        from ..utils import load_config as _load_config
        data_dir = Path(config.get("paths", {}).get("data_dir", "数据文件"))
        gene_meta_p = pd.read_csv(data_dir / "metadata" / "gene_metadata.csv")
        pathway_meta_p = pd.read_csv(data_dir / "metadata" / "pathway_metadata.csv")
        from .baselines import build_pw140_membership_features
        pw140_p = build_pw140_membership_features(gene_meta_p, pathway_meta_p)

        profile_gene_feats = gene_feat_df.join(pw140_p, how="inner")
        profile_predictor = MultiOutputGeneProfile(
            n_components=profile_cfg.get("n_components", 50),
            random_state=42,
        )
        profile_predictor.fit(profile_gene_feats, labels_df)

        # Compute profile predictions for training genes
        profile_preds_train = profile_predictor.predict(
            profile_gene_feats, cell_ids, gene_ids,
        )

        # Blend for cold genes in training data
        profile_blend_weight = profile_cfg.get("blend_weight", 0.3)
        n_profile_blended = 0
        for i in range(n):
            if gene_ids[i] in cold_genes:
                full_preds[i] = (1.0 - profile_blend_weight) * full_preds[i] \
                    + profile_blend_weight * profile_preds_train[i]
                n_profile_blended += 1
        if n_profile_blended > 0:
            print(f"  Profile blended for {n_profile_blended:,} cold-gene "
                  f"predictions (weight={profile_blend_weight:.2f})")
            print(f"  {profile_predictor.formula_str().split(chr(10))[0]}")
            print(f"  {profile_predictor.formula_str().split(chr(10))[2]}")

    # ── Step 6b: Gene-Similarity CF for cold genes ──
    cf_cold_predictions: dict[tuple, float] = {}
    cf_weight = hybrid_cfg.get("cf_weight", 0.5)
    if cold_genes:
        from .baselines import build_gene_similarity_cf
        labels_df_for_cf = pd.DataFrame({
            "cell_line_id": cell_ids,
            "perturbation_gene": gene_ids,
            "label": y,
        })
        cf_cold_predictions = build_gene_similarity_cf(
            labels_df_for_cf, gene_meta, pathway_meta,
            expression if expression is not None else pd.DataFrame(),
            cold_genes, k=hybrid_cfg.get("cf_knn", 30),
            multi_kernel=hybrid_cfg.get("cf_multi_kernel", True),
            power=hybrid_cfg.get("cf_power", 2.0),
        )
        print(f"  CF: {len(cf_cold_predictions):,} cold (cell,gene) pairs with "
              f"predictions (multi-kernel={hybrid_cfg.get('cf_multi_kernel', True)})")

    # ── Compute full predictions ──
    mu_arr = shrink_gene.predict_batch(list(gene_ids))
    beta_arr = shrink_cell.predict_batch(list(cell_ids))
    i_arr = hybrid_interaction.predict(cell_ids, gene_ids)
    full_preds = mu_arr + beta_arr + i_arr

    # Blend CF predictions for cold genes
    if cf_cold_predictions:
        cf_blend_count = 0
        for i in range(n):
            key = (cell_ids[i], gene_ids[i])
            if key in cf_cold_predictions and gene_ids[i] in cold_genes:
                cf_val = cf_cold_predictions[key]
                # Blend: (1-w)·formula + w·CF for cold genes
                full_preds[i] = (1.0 - cf_weight) * full_preds[i] \
                    + cf_weight * cf_val
                cf_blend_count += 1
        print(f"  CF blended for {cf_blend_count:,} cold-gene predictions "
              f"(weight={cf_weight:.2f})")

    # ── Step 7: Per-cell isotonic calibration ──
    cal_cfg = fm_cfg.get("calibration", {})
    calibrator = None
    if cal_cfg.get("enabled", True):
        print(f"\n[Step 6] Per-Cell Isotonic Calibration")
        print(f"  f_c: pred → truth (monotonic, per-cell PAV isotonic regression)")
        print(f"  min_samples={cal_cfg.get('min_samples', 200)} "
              f"(cells with fewer warm genes use global calibration)")
        from .calibration import PerCellIsotonicCalibrator
        calibrator = PerCellIsotonicCalibrator(
            y_min=cal_cfg.get("y_min", -5.0),
            y_max=cal_cfg.get("y_max", 5.0),
            min_samples=cal_cfg.get("min_samples", 200),
        )
        calibrator.fit(full_preds, y, cell_ids, gene_ids, cold_genes)
        full_preds_cal = calibrator.transform(full_preds, cell_ids)

        # Report calibration effect
        cal_rmse_before = np.sqrt(np.mean((full_preds - y) ** 2))
        cal_rmse_after = np.sqrt(np.mean((full_preds_cal - y) ** 2))
        print(f"  RMSE: {cal_rmse_before:.6f} → {cal_rmse_after:.6f} "
              f"({(cal_rmse_before - cal_rmse_after) / max(cal_rmse_before, 1e-12) * 100:+.1f}%)")
        full_preds = full_preds_cal

    # ── Print formula summary ──
    _print_hybrid_summary(shrink_gene, shrink_cell, hybrid_interaction,
                          len(cf_cold_predictions))

    return {
        "shrink_gene": shrink_gene,
        "shrink_cell": shrink_cell,
        "hybrid_interaction": hybrid_interaction,
        "cell_features": cell_features,
        "gene_features": gene_feat_df,
        "mu_g": shrink_gene.mu_g_,
        "beta_c": shrink_cell.beta_c_,
        "cold_genes": cold_genes,
        "full_preds": full_preds,
        "cf_cold_predictions": cf_cold_predictions,
        "cf_weight": cf_weight,
        "calibrator": calibrator,
        "quantile_align_cfg": fm_cfg.get("quantile_align", {}),
        "profile_predictor": profile_predictor,
        "profile_cfg": profile_cfg,
        "profile_gene_feats": (
            profile_gene_feats if profile_predictor is not None else None
        ),
    }


def predict_formula(
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    cold_genes: set[str] | None = None,
    models: dict[str, Any] | None = None,
    add_jitter: bool = True,
    expression: pd.DataFrame | None = None,
) -> np.ndarray:
    """Generate predictions using trained formula models.

    Unified formula for ALL genes (no hard warm/cold distinction):
      ŷ(c,g) = μ̂_g + β̂_c + I_structured(c,g) [+ I_imc_residual(c,g)]

    where μ̂_g and β̂_c come from the empirical Bayes shrinkage models,
    and the structured interaction uses biologically meaningful features.
    """
    if models is None:
        return np.zeros(len(cell_ids), dtype=np.float32)
    if cold_genes is None:
        cold_genes = set()

    n = len(cell_ids)

    # ── Gene essentiality (shrinkage model) ──
    shrink_gene = models.get("shrink_gene")
    if shrink_gene is not None:
        mu_arr = shrink_gene.predict_batch(list(gene_ids))
    else:
        mu_g = models.get("mu_g", {})
        mu_arr = np.array([mu_g.get(g, 0.0) for g in gene_ids], dtype=np.float32)

    # ── Cell vulnerability (shrinkage model) ──
    shrink_cell = models.get("shrink_cell")
    if shrink_cell is not None:
        beta_arr = shrink_cell.predict_batch(list(cell_ids))
    else:
        beta_c = models.get("beta_c", {})
        beta_arr = np.array([beta_c.get(c, 0.0) for c in cell_ids], dtype=np.float32)

    # ── Hybrid SVD + CF Interaction ──
    hybrid_interaction = models.get("hybrid_interaction")
    if hybrid_interaction is not None:
        i_arr = hybrid_interaction.predict(cell_ids, gene_ids)
    else:
        # Fallback: try legacy interaction models
        si_interaction = models.get("si_interaction")
        imc_interaction = models.get("imc_interaction")
        i_arr = np.zeros(n, dtype=np.float32)
        if si_interaction is not None:
            cell_features_df = models.get("cell_features")
            gene_features_df = models.get("gene_features")
            if cell_features_df is not None and gene_features_df is not None \
               and expression is not None:
                pair_z, pair_pct = _compute_pair_expr_features(
                    expression, cell_ids, gene_ids,
                )
                cell_df = pd.DataFrame(
                    cell_features_df.reindex(cell_ids).to_numpy(dtype=np.float64),
                    columns=cell_features_df.columns,
                ).fillna(0.0)
                gene_df = pd.DataFrame(
                    gene_features_df.reindex(gene_ids).to_numpy(dtype=np.float64),
                    columns=gene_features_df.columns,
                ).fillna(0.0)
                i_arr += si_interaction.predict(cell_df, gene_df, pair_z, pair_pct)
        if imc_interaction is not None:
            cell_features_df = models.get("cell_features")
            gene_features_df = models.get("gene_features")
            if cell_features_df is not None and gene_features_df is not None:
                Xc = cell_features_df.reindex(cell_ids).to_numpy(dtype=np.float64)
                Xc = np.nan_to_num(Xc, nan=0.0)
                Xg = gene_features_df.reindex(gene_ids).to_numpy(dtype=np.float64)
                Xg = np.nan_to_num(Xg, nan=0.0)
                i_arr += imc_interaction.predict(Xc, Xg)

    # ── Final prediction ──
    final = mu_arr + beta_arr + i_arr

    # ── Multi-Output Profile blending for cold genes ──
    profile_predictor = models.get("profile_predictor")
    if profile_predictor is not None and cold_genes:
        profile_cfg = models.get("profile_cfg", {})
        profile_blend_weight = profile_cfg.get("blend_weight", 0.3)
        profile_gene_feats = models.get("profile_gene_feats")
        if profile_gene_feats is not None:
            try:
                profile_preds = profile_predictor.predict(
                    profile_gene_feats, cell_ids, gene_ids,
                )
                n_profile = 0
                for i in range(n):
                    if gene_ids[i] in cold_genes:
                        final[i] = (1.0 - profile_blend_weight) * final[i] \
                            + profile_blend_weight * profile_preds[i]
                        n_profile += 1
                if n_profile > 0:
                    print(f"  Profile blended for {n_profile:,} cold-gene "
                          f"predictions (weight={profile_blend_weight:.2f})")
            except Exception:
                pass  # Graceful fallback

    # ── Gene-Similarity CF blending for cold genes ──
    cf_cold_predictions = models.get("cf_cold_predictions", {})
    if cf_cold_predictions:
        cf_weight = models.get("cf_weight", 0.5)
        n_blended = 0
        for i in range(n):
            key = (cell_ids[i], gene_ids[i])
            if key in cf_cold_predictions and gene_ids[i] in cold_genes:
                cf_val = cf_cold_predictions[key]
                final[i] = (1.0 - cf_weight) * final[i] + cf_weight * cf_val
                n_blended += 1
        if n_blended > 0:
            print(f"  CF blended for {n_blended:,} cold-gene predictions "
                  f"(weight={cf_weight:.2f})")

    if add_jitter:
        rng = np.random.RandomState(42)
        final += rng.rand(len(final)).astype(np.float32) * 1e-6

    # ── Per-cell isotonic calibration ──
    calibrator = models.get("calibrator")
    if calibrator is not None:
        final = calibrator.transform(final, cell_ids)

    # ── Per-cell quantile alignment for cold genes ──
    qa_cfg = models.get("quantile_align_cfg")
    if qa_cfg and qa_cfg.get("enabled", True) and cold_genes:
        from .calibration import PerCellQuantileAligner
        aligner = PerCellQuantileAligner(
            min_warm=qa_cfg.get("min_warm", 20),
            min_cold=qa_cfg.get("min_cold", 5),
        )
        final = aligner.align(final, cell_ids, gene_ids, cold_genes)

    return final.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _compute_pair_expr_features(
    expression: pd.DataFrame,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute z_cg and expression percentile for (cell, gene) pairs.

    Args:
        expression: N_cells × P_genes z-scored expression DataFrame.
        cell_ids: (N,) cell identifiers.
        gene_ids: (N,) gene identifiers.

    Returns:
        (z_cg, expr_percentile): both (N,) float64 arrays.
    """
    cell_to_idx = {c: i for i, c in enumerate(expression.index)}
    gene_list = expression.columns.tolist()
    gene_to_idx = {g: i for i, g in enumerate(gene_list)}
    expr_arr = expression.to_numpy(dtype=np.float64)

    n = len(cell_ids)
    z_cg = np.zeros(n, dtype=np.float64)
    expr_pct = np.zeros(n, dtype=np.float64)

    # Process unique cells for efficiency
    for cell in np.unique(cell_ids):
        ci = cell_to_idx.get(cell)
        if ci is None:
            continue
        mask = cell_ids == cell
        cell_genes = gene_ids[mask]
        g_indices = np.array([gene_to_idx.get(g, -1) for g in cell_genes])

        valid = g_indices >= 0
        if not valid.any():
            continue

        cell_expr = expr_arr[ci]
        z_cg[mask] = np.where(valid, cell_expr[g_indices], 0.0)
        # Percentile: fraction of genes with lower expression in this cell
        z_vals = cell_expr[g_indices[valid]]
        expr_pct[mask] = np.where(
            valid,
            (cell_expr[np.newaxis, :] < z_vals[:, np.newaxis]).mean(axis=1),
            0.5,
        )

    return z_cg, expr_pct


# ═══════════════════════════════════════════════════════════════════════════════
# Formula Printout
# ═══════════════════════════════════════════════════════════════════════════════

def _print_hybrid_summary(
    shrink_gene: "ShrinkageGeneFormula",
    shrink_cell: "ShrinkageCellFormula",
    hybrid: "HybridInteraction",
    n_cf_pairs: int = 0,
) -> None:
    """Print the human-readable formula with hybrid SVD+CF details."""
    print("\n" + "=" * 70)
    print("COMPLETE PREDICTION FORMULA (Empirical Bayes + Hybrid SVD/CF)")
    print("=" * 70)
    print(f"""
ŷ(c,g) = [w_g·x̄_g + (1-w_g)·Φ(g)] + [v_c·r̄_c + (1-v_c)·Ψ(c)]
       + I(c,g)

SHRINKAGE PARAMETERS:
  λ_gene = {shrink_gene.lambda_:.1f}  (gene prior strength)
  λ_cell = {shrink_cell.lambda_:.1f}  (cell prior strength)

HYBRID INTERACTION:
  Warm genes: I(c,g) = Σ_k σ_k · u_k(c) · v_k(g)  [SVD, {hybrid.n_components_used_} components]
  Cold genes: I(c,g) = Σ_w sim(g,w) · I_SVD(c,w)  [CF transfer]
  Training R² = {hybrid.train_r2_:.4f} (on double residual)
  Cold genes with CF neighbors: {len(hybrid.cold_to_warm_)}
  Cold gene CF label predictions: {n_cf_pairs:,} pairs
""")
    print()
    print("─" * 70)
    print("GENE PRIOR Φ(g):")
    print(shrink_gene.gene_formula_.formula_str(top_n=6))
    print()
    print("─" * 70)
    print("CELL PRIOR Ψ(c):")
    print(shrink_cell.cell_formula_.formula_str(top_n=6))
    print("=" * 70)
