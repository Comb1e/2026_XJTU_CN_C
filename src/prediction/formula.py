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
) -> dict[str, Any]:
    """Train formula-based models with empirical Bayes smooth transition.

    Sequential fitting (NO hard warm/cold distinction):
      1. ShrinkageGeneFormula → μ̂_g = w_g·x̄_g + (1-w_g)·Φ(g)
      2. ShrinkageCellFormula → β̂_c = v_c·r̄_c + (1-v_c)·Ψ(c)
      3. SVDInteraction → I(c,g) = Σ_k σ_k · u_k(c) · v_k(g)
      4. ColdGeneTransfer for genes with n_g=0

    Args:
        y: label array.
        cell_ids, gene_ids: identifiers for each row.
        gene_static_features: G1 features indexed by gene.
        gene_expr_profile_features: G2 features indexed by gene.
        cell_features: G3 features indexed by cell.
        cold_genes: set of cold gene symbols (n_g=0).
        config: configuration dict.

    Returns:
        dict with shrinkage models, SVD interaction, and metadata.
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

    # ── Build feature matrices for IMC interaction ──
    # X_cell: (N, f_c) from cell_features, indexed by cell_ids
    # X_gene: (N, f_g) from gene_feat_df, indexed by gene_ids
    X_cell_arr = cell_features.reindex(cell_ids).to_numpy(dtype=np.float64)
    X_cell_arr = np.nan_to_num(X_cell_arr, nan=0.0)
    cell_feature_names = list(cell_features.columns)

    X_gene_arr = gene_feat_df.reindex(gene_ids).to_numpy(dtype=np.float64)
    X_gene_arr = np.nan_to_num(X_gene_arr, nan=0.0)
    gene_feature_names = list(gene_feat_df.columns)

    # ── Step 4: IMC Bilinear Interaction ──
    print(f"\n[Step 4] IMC Bilinear Interaction (rank={fm_cfg.get('imc', {}).get('rank', 10)})")
    print("  r̂(c,g) = x_c^T W H^T y_g")
    imc_cfg = fm_cfg.get("imc", {})
    imc_interaction = IMCInteraction(
        rank=imc_cfg.get("rank", 10),
        lambda_w=imc_cfg.get("lambda_w", 1.0),
        lambda_h=imc_cfg.get("lambda_h", 1.0),
        max_iter=imc_cfg.get("max_iter", 30),
    )
    imc_interaction.fit(
        X_cell_arr, X_gene_arr, r2,
        cell_feature_names=cell_feature_names,
        gene_feature_names=gene_feature_names,
    )

    # ── Compute full predictions ──
    mu_arr = shrink_gene.predict_batch(list(gene_ids))
    beta_arr = shrink_cell.predict_batch(list(cell_ids))
    i_arr = imc_interaction.predict(X_cell_arr, X_gene_arr)
    full_preds = mu_arr + beta_arr + i_arr

    # ── Print formula summary ──
    _print_imc_summary(shrink_gene, shrink_cell, imc_interaction)

    return {
        "shrink_gene": shrink_gene,
        "shrink_cell": shrink_cell,
        "imc_interaction": imc_interaction,
        "cell_features": cell_features,
        "gene_features": gene_feat_df,
        "mu_g": shrink_gene.mu_g_,
        "beta_c": shrink_cell.beta_c_,
        "cold_genes": cold_genes,
        "full_preds": full_preds,
    }


def predict_formula(
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    cold_genes: set[str] | None = None,
    models: dict[str, Any] | None = None,
    add_jitter: bool = True,
) -> np.ndarray:
    """Generate predictions using trained formula models.

    Unified formula for ALL genes (no hard warm/cold distinction):
      ŷ(c,g) = μ̂_g + β̂_c + x_c^T W H^T y_g

    where μ̂_g and β̂_c come from the empirical Bayes shrinkage models,
    and the IMC bilinear interaction uses cell/gene features directly.
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

    # ── IMC Bilinear Interaction ──
    imc_interaction = models.get("imc_interaction")
    if imc_interaction is not None:
        # Build feature matrices from stored DataFrames
        cell_features_df = models.get("cell_features")
        gene_features_df = models.get("gene_features")
        if cell_features_df is not None and gene_features_df is not None:
            Xc = cell_features_df.reindex(cell_ids).to_numpy(dtype=np.float64)
            Xc = np.nan_to_num(Xc, nan=0.0)
            Xg = gene_features_df.reindex(gene_ids).to_numpy(dtype=np.float64)
            Xg = np.nan_to_num(Xg, nan=0.0)
            i_arr = imc_interaction.predict(Xc, Xg)
        else:
            i_arr = np.zeros(n, dtype=np.float32)
    else:
        i_arr = np.zeros(n, dtype=np.float32)

    # ── Final prediction ──
    final = mu_arr + beta_arr + i_arr

    if add_jitter:
        rng = np.random.RandomState(42)
        final += rng.rand(len(final)).astype(np.float32) * 1e-6

    return final.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Formula Printout
# ═══════════════════════════════════════════════════════════════════════════════

def _print_imc_summary(
    shrink_gene: "ShrinkageGeneFormula",
    shrink_cell: "ShrinkageCellFormula",
    imc_interaction: "IMCInteraction",
) -> None:
    """Print the human-readable formula with IMC interaction details."""
    print("\n" + "=" * 70)
    print("COMPLETE PREDICTION FORMULA (Empirical Bayes + IMC)")
    print("=" * 70)
    print(f"""
ŷ(c,g) = [w_g·x̄_g + (1-w_g)·Φ(g)] + [v_c·r̄_c + (1-v_c)·Ψ(c)]
       + x_c^T W H^T y_g

SHRINKAGE PARAMETERS:
  λ_gene = {shrink_gene.lambda_:.1f}  (gene prior strength)
  λ_cell = {shrink_cell.lambda_:.1f}  (cell prior strength)
  w_g = n_g/(n_g+λ_gene)  (gene evidence weight, 0=cold → 1=warm)
  v_c = m_c/(m_c+λ_cell)  (cell evidence weight)

IMC INTERACTION:
  rank = {imc_interaction.rank}
  Training R² = {imc_interaction.train_r2_:.4f} (on double residual)
""")
    top = imc_interaction.get_top_interactions(top_n=10)
    if top:
        print("  Top feature×feature interactions:")
        for ci, gj, z in top[:5]:
            print(f"    Z[{ci}, {gj}] = {z:+.4f}")

    print()
    print("─" * 70)
    print("GENE PRIOR Φ(g):")
    print(shrink_gene.gene_formula_.formula_str(top_n=6))
    print()
    print("─" * 70)
    print("CELL PRIOR Ψ(c):")
    print(shrink_cell.cell_formula_.formula_str(top_n=6))
    print("=" * 70)
