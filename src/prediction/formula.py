"""Formula-based interpretable gene dependency prediction.

Architecture (Empirical Bayes smooth transition):

    ŷ(c,g) = μ̂_g + β̂_c + Blend[I_mod, I_expr, I_match, I_ew]

where:
  - μ̂_g = w_g·x̄_g + (1-w_g)·Φ(g)  ← empirical Bayes shrinkage
    w_g = n_g/(n_g+λ),  Φ(g) = Ridge(G1+G2 → gene mean)
  - β̂_c = v_c·r̄_c + (1-v_c)·Ψ(c)  ← same shrinkage for cells
    v_c = m_c/(m_c+λ_cell), Ψ(c) = Ridge(G3 → cell mean)
  - I_mod : Module×Indicator interaction (14 named coeffs)
  - I_expr : Asymmetric expression→dependency curve (4 coeffs)
  - I_match : Module-weighted expression percentile (14 coeffs)
  - I_ew : Evidence-weighted expression coupling (1 coeff)

Key innovation: NO hard warm/cold distinction. Every gene uses the SAME formula;
the evidence weight w_g smoothly transitions from 0 (cold: pure prior) to 1
(warm: data-dominated). This is the same principle used by Chronos (hierarchical
kernel prior), CERES (hierarchical prior), and ashr (adaptive shrinkage).

References:
  - Chronos: r_cg = R*_cg/R_c − 1 (Dempster et al., Genome Biology 2021)
  - CERES: hierarchical prior + partial pooling (Meyers et al., Nat Genet 2017)
  - ashr: adaptive shrinkage (Stephens, Biostatistics 2017)
  - James-Stein estimator (1961)
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
# Formula 3: Module×Indicator Interaction
#   I_mod(c,g) = Σ_m η_m · Module_m(g) · Indicator_m(c)
# ═══════════════════════════════════════════════════════════════════════════════

class ModuleInteractionFormula:
    """Module membership × Cell indicator multiplicative interaction.

    I_mod(c,g) = Σ_{m=0}^{13} η_m · Module_m(g) · Indicator_m(c)

    This is the KEY mechanistic insight derived from the Chronos model:
    gene essentiality depends on the cell's activity in the gene's module.

    Each η_m measures: "how much does cell-level module-m activity amplify
    the dependency of module-m genes?"

    14 named coefficients, one per mitochondrial module:
      OXPHOS_CI(0), OXPHOS_CII_CIII(1), OXPHOS_CIV_CV(2), TCA_PYRUVATE(3),
      FAO_LIPID(4), AA_COFACTOR(5), MITO_RIBOSOME(6), mtDNA_RNA(7),
      PROTEIN_IMPORT(8), TRANSPORT(9), REDOX_DETOX(10), MITO_DYNAMICS(11),
      CELL_DEATH(12), SIGNALING(13)
    """

    MODULE_NAMES = [
        "OXPHOS_CI", "OXPHOS_CII_CIII", "OXPHOS_CIV_CV", "TCA_PYRUVATE",
        "FAO_LIPID", "AA_COFACTOR", "MITO_RIBOSOME", "mtDNA_RNA",
        "PROTEIN_IMPORT", "TRANSPORT", "REDOX_DETOX", "MITO_DYNAMICS",
        "CELL_DEATH", "SIGNALING",
    ]

    def __init__(self, alpha: float = 1.0, n_modules: int = 14):
        self.alpha = alpha
        self.n_modules = n_modules
        self.model_: Ridge | None = None
        self.coefficients_: np.ndarray | None = None  # (n_modules,)
        self.intercept_: float = 0.0

    def fit(
        self,
        cell_ids: np.ndarray,               # (N,) cell identifiers
        gene_ids: np.ndarray,                # (N,) gene identifiers
        residuals: np.ndarray,               # (N,) double residual y - μ̂_g - β̂_c
        module_membership: np.ndarray,        # (N, 14) one-hot module membership per pair
        cell_indicators: np.ndarray,          # (N, 14) cell indicator values per pair
    ) -> "ModuleInteractionFormula":
        """Fit Ridge on the 14 interaction features."""
        # Construct: X[:, m] = module_membership[:, m] * cell_indicators[:, m]
        X = (module_membership.astype(np.float64) *
             cell_indicators.astype(np.float64))
        y = residuals.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)

        ridge_cv = RidgeCV(
            alphas=np.logspace(-2, 3, 20),
            fit_intercept=True,
            store_cv_results=False,
        )
        ridge_cv.fit(X, y)

        self.model_ = Ridge(alpha=ridge_cv.alpha_, fit_intercept=True)
        self.model_.fit(X, y)
        self.coefficients_ = self.model_.coef_
        self.intercept_ = float(self.model_.intercept_)

        # Report
        nz = int(np.sum(np.abs(self.coefficients_) > 1e-6))
        top_idx = np.argmax(np.abs(self.coefficients_))
        print(f"  Module interaction: {nz}/{self.n_modules} nonzero, "
              f"α={ridge_cv.alpha_:.2f}, "
              f"top={self.MODULE_NAMES[top_idx]}({self.coefficients_[top_idx]:.4f})")
        return self

    def predict(
        self,
        module_membership: np.ndarray,       # (N, 14)
        cell_indicators: np.ndarray,          # (N, 14)
    ) -> np.ndarray:
        """Predict interaction values for (cell, gene) pairs."""
        if self.model_ is None:
            return np.zeros(len(module_membership), dtype=np.float32)
        X = (module_membership.astype(np.float64) *
             cell_indicators.astype(np.float64))
        X = np.nan_to_num(X, nan=0.0)
        return self.model_.predict(X).astype(np.float32)

    def get_coefficients(self) -> list[tuple[str, float]]:
        """Return named module interaction coefficients."""
        if self.coefficients_ is None:
            return []
        return [(self.MODULE_NAMES[m], float(self.coefficients_[m]))
                for m in range(self.n_modules)]

    def formula_str(self) -> str:
        """Return human-readable formula string."""
        coefs = self.get_coefficients()
        parts = ["I_mod(c,g) = Σ_m η_m · Module_m(g) · Indicator_m(c)"]
        parts.append(f"  intercept = {self.intercept_:.4f}")
        for name, coef in coefs:
            if abs(coef) > 1e-6:
                sign = "+" if coef >= 0 else "-"
                parts.append(f"  η_{name} = {sign} {abs(coef):.4f}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Formula 4: Asymmetric Expression→Dependency Effect
#   I_expr(c,g) = θ₁·z + θ₂·|z| + θ₃·max(0,z) + θ₄·min(0,z)
# ═══════════════════════════════════════════════════════════════════════════════

class ExpressionEffectFormula:
    """Piecewise-linear expression→dependency mapping.

    I_expr(c,g) = θ₁·z + θ₂·|z| + θ₃·max(0,z) + θ₄·min(0,z)

    where z = z_{c,g} is the expression z-score of gene g in cell c.

    This captures asymmetric, nonlinear effects:
      - θ₁ : linear trend (higher expression → more/less dependency)
      - θ₂ : magnitude effect (extreme expression in EITHER direction matters)
      - θ₃ : positive tail (overexpression-specific effect)
      - θ₄ : negative tail (underexpression-specific effect)

    4 interpretable coefficients.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.model_: Ridge | None = None
        self.scaler_: StandardScaler | None = None
        self.coefficients_: np.ndarray | None = None  # [θ₁, θ₂, θ₃, θ₄]
        self.intercept_: float = 0.0

    def fit(
        self,
        z_cg: np.ndarray,                    # (N,) expression z-scores
        residuals: np.ndarray,               # (N,) double residual
    ) -> "ExpressionEffectFormula":
        """Fit Ridge on the 4 basis functions of z."""
        z = z_cg.astype(np.float64).reshape(-1, 1)
        y = residuals.astype(np.float64)

        # Build basis functions
        X = np.column_stack([
            z.ravel(),                       # θ₁: linear
            np.abs(z.ravel()),               # θ₂: magnitude
            np.maximum(0, z.ravel()),         # θ₃: positive tail
            np.minimum(0, z.ravel()),         # θ₄: negative tail
        ])
        X = np.nan_to_num(X, nan=0.0)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        ridge_cv = RidgeCV(
            alphas=np.logspace(-3, 2, 20),
            fit_intercept=True,
            store_cv_results=False,
        )
        ridge_cv.fit(X_scaled, y)

        self.model_ = Ridge(alpha=ridge_cv.alpha_, fit_intercept=True)
        self.model_.fit(X_scaled, y)
        self.coefficients_ = self.model_.coef_
        self.intercept_ = float(self.model_.intercept_)

        from sklearn.metrics import r2_score
        y_pred = self.model_.predict(X_scaled)
        r2 = r2_score(y, y_pred)
        print(f"  Expression effect: R²={r2:.4f}, "
              f"θ=[{', '.join(f'{c:.4f}' for c in self.coefficients_)}], "
              f"α={ridge_cv.alpha_:.2f}")
        return self

    def predict(self, z_cg: np.ndarray) -> np.ndarray:
        """Predict expression-driven dependency effect."""
        if self.model_ is None:
            return np.zeros(len(z_cg), dtype=np.float32)
        z = z_cg.astype(np.float64).reshape(-1, 1)
        X = np.column_stack([
            z.ravel(), np.abs(z.ravel()),
            np.maximum(0, z.ravel()), np.minimum(0, z.ravel()),
        ])
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled).astype(np.float32)

    def formula_str(self) -> str:
        """Return human-readable formula string."""
        if self.coefficients_ is None:
            return "I_expr(c,g) = 0"
        names = ["z", "|z|", "max(0,z)", "min(0,z)"]
        parts = [f"I_expr(c,g) = {self.intercept_:.4f}"]
        for name, coef in zip(names, self.coefficients_):
            sign = "+" if coef >= 0 else "-"
            parts.append(f"  {sign} {abs(coef):.4f} · {name}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Formula 5: Module-Weighted Expression Percentile
#   I_match(c,g) = Σ_m ζ_m · Module_m(g) · ExprPercentile(c,g)
# ═══════════════════════════════════════════════════════════════════════════════

class ModuleMatchFormula:
    """Gene module membership weighted by expression percentile.

    I_match(c,g) = Σ_m ζ_m · Module_m(g) · ExprPercentile(c,g)

    Captures: "for genes in module m, how much does their expression
    percentile (relative to other genes in the same cell) matter?"

    14 named coefficients, one per module.
    """

    MODULE_NAMES = ModuleInteractionFormula.MODULE_NAMES

    def __init__(self, alpha: float = 1.0, n_modules: int = 14):
        self.alpha = alpha
        self.n_modules = n_modules
        self.model_: Ridge | None = None
        self.coefficients_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(
        self,
        module_membership: np.ndarray,        # (N, 14)
        expr_percentile: np.ndarray,          # (N,)
        residuals: np.ndarray,                # (N,)
    ) -> "ModuleMatchFormula":
        """Fit Ridge on module×percentile features."""
        pct = expr_percentile.astype(np.float64).reshape(-1, 1)
        X = module_membership.astype(np.float64) * pct  # broadcasting: (N,14) * (N,1)
        y = residuals.astype(np.float64)
        X = np.nan_to_num(X, nan=0.0)

        ridge_cv = RidgeCV(
            alphas=np.logspace(-2, 3, 20),
            fit_intercept=True,
            store_cv_results=False,
        )
        ridge_cv.fit(X, y)

        self.model_ = Ridge(alpha=ridge_cv.alpha_, fit_intercept=True)
        self.model_.fit(X, y)
        self.coefficients_ = self.model_.coef_
        self.intercept_ = float(self.model_.intercept_)

        nz = int(np.sum(np.abs(self.coefficients_) > 1e-6))
        if nz > 0:
            top_idx = np.argmax(np.abs(self.coefficients_))
            top_name = self.MODULE_NAMES[top_idx]
        else:
            top_name = "none"
        print(f"  Module match: {nz}/{self.n_modules} nonzero, "
              f"α={ridge_cv.alpha_:.2f}, top={top_name}")
        return self

    def predict(
        self,
        module_membership: np.ndarray,        # (N, 14)
        expr_percentile: np.ndarray,          # (N,)
    ) -> np.ndarray:
        """Predict module-weighted expression percentile effect."""
        if self.model_ is None:
            return np.zeros(len(module_membership), dtype=np.float32)
        pct = expr_percentile.astype(np.float64).reshape(-1, 1)
        X = module_membership.astype(np.float64) * pct
        X = np.nan_to_num(X, nan=0.0)
        return self.model_.predict(X).astype(np.float32)

    def formula_str(self) -> str:
        """Return human-readable formula string."""
        if self.coefficients_ is None:
            return "I_match(c,g) = 0"
        parts = ["I_match(c,g) = Σ_m ζ_m · Module_m(g) · ExprPercentile(c,g)"]
        for m in range(self.n_modules):
            coef = self.coefficients_[m]
            if abs(coef) > 1e-6:
                sign = "+" if coef >= 0 else "-"
                parts.append(f"  ζ_{self.MODULE_NAMES[m]} = {sign} {abs(coef):.4f}")
        return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Formula 6: Evidence-Weighted Expression Coupling
#   I_ew(c,g) = ω · EvidenceWeight(g) · z_cg
# ═══════════════════════════════════════════════════════════════════════════════

class EvidenceWeightedFormula:
    """Evidence-weighted expression coupling.

    I_ew(c,g) = ω · EvidenceWeight(g) · z(c,g)

    Hypothesis: genes with stronger mitochondrial evidence (MitoCarta
    confidence) show tighter coupling between expression and dependency.

    Single coefficient ω — the simplest and most interpretable formula.
    """

    def __init__(self):
        self.omega_: float = 0.0
        self.intercept_: float = 0.0

    def fit(
        self,
        evidence_weight: np.ndarray,          # (N,) gene evidence weights
        z_cg: np.ndarray,                      # (N,) expression z-scores
        residuals: np.ndarray,                 # (N,) double residual
    ) -> "EvidenceWeightedFormula":
        """Fit single-feature OLS."""
        X = (evidence_weight * z_cg).astype(np.float64)
        y = residuals.astype(np.float64)
        mask = np.isfinite(X) & np.isfinite(y)
        X = X[mask].reshape(-1, 1)
        y = y[mask]

        if len(y) < 10:
            return self

        # Single-feature Ridge
        ridge = Ridge(alpha=0.01)
        ridge.fit(X, y)
        self.omega_ = float(ridge.coef_[0])
        self.intercept_ = float(ridge.intercept_)

        from sklearn.metrics import r2_score
        y_pred = ridge.predict(X)
        r2 = r2_score(y, y_pred)
        print(f"  Evidence-weighted: R²={r2:.4f}, ω={self.omega_:.4f}")
        return self

    def predict(
        self,
        evidence_weight: np.ndarray,
        z_cg: np.ndarray,
    ) -> np.ndarray:
        """Predict evidence-weighted expression effect."""
        X = (evidence_weight * z_cg).astype(np.float32)
        X = np.nan_to_num(X, nan=0.0)
        return (self.intercept_ + self.omega_ * X).astype(np.float32)

    def formula_str(self) -> str:
        return f"I_ew(c,g) = {self.intercept_:.4f} + {self.omega_:.4f} · EvidenceWeight(g) · z(c,g)"


# ═══════════════════════════════════════════════════════════════════════════════
# Interaction Blend: RidgeCV combination of interaction formulas
# ═══════════════════════════════════════════════════════════════════════════════

class InteractionBlend:
    """RidgeCV blend of multiple interaction formulas.

    Learns: I_blend(c,g) = α₀ + Σ_j α_j · I_j(c,g)

    where I_j are the individual interaction formula predictions.
    The blend weights α_j are the ONLY learned weights in the interaction
    term — everything else is fixed by the explicit formulas.
    """

    def __init__(self, alphas: list[float] | None = None):
        if alphas is None:
            alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        self.alphas = alphas
        self.model_: RidgeCV | None = None
        self.coefficients_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.component_names_: list[str] = []

    def fit(
        self,
        interaction_preds: list[np.ndarray],  # list of (N,) arrays
        residuals: np.ndarray,                # (N,) double residual
        component_names: list[str] | None = None,
    ) -> "InteractionBlend":
        """Fit RidgeCV to find optimal blend weights."""
        if component_names is not None:
            self.component_names_ = list(component_names)
        else:
            self.component_names_ = [f"I_{i}" for i in range(len(interaction_preds))]

        X = np.column_stack([p.astype(np.float64) for p in interaction_preds])
        X = np.nan_to_num(X, nan=0.0)
        y = residuals.astype(np.float64)

        self.model_ = RidgeCV(
            alphas=self.alphas,
            fit_intercept=True,
            store_cv_results=False,
        )
        self.model_.fit(X, y)
        self.coefficients_ = self.model_.coef_
        self.intercept_ = float(self.model_.intercept_)

        from sklearn.metrics import r2_score
        y_pred = self.model_.predict(X)
        r2 = r2_score(y, y_pred)
        weights = list(zip(self.component_names_, self.coefficients_))
        print(f"  Interaction blend: R²={r2:.4f}, α={self.model_.alpha_:.2f}")
        for name, w in weights:
            print(f"    {name}: {w:+.4f}")
        return self

    def predict(self, interaction_preds: list[np.ndarray]) -> np.ndarray:
        """Blend interaction predictions."""
        if self.model_ is None:
            return np.zeros(len(interaction_preds[0]), dtype=np.float32)
        X = np.column_stack([p.astype(np.float64) for p in interaction_preds])
        X = np.nan_to_num(X, nan=0.0)
        return self.model_.predict(X).astype(np.float32)

    def get_weights(self) -> list[tuple[str, float]]:
        """Return named blend weights."""
        if self.coefficients_ is None:
            return []
        return list(zip(self.component_names_, self.coefficients_))


# ═══════════════════════════════════════════════════════════════════════════════
# Cold Gene Transfer: pathway-similarity KNN for interaction term
# ═══════════════════════════════════════════════════════════════════════════════

class ColdGeneTransfer:
    """Transfer cell-specific interaction from pathway-similar warm genes.

    For a cold gene g_cold in cell c:
      I_transfer(c, g_cold) = Σ_{g' ∈ KNN(g_cold)} sim(g_cold, g') · I(c, g') / Σ sim

    Uses MitoCarta3.0 pathway annotations (149-dim) for similarity.
    The transfer is applied to the INTERACTION term only — μ̂_g and β̂_c
    are already available for all genes/cells via their explicit formulas.
    """

    def __init__(self, k: int = 20):
        self.k = k
        self.warm_genes_: list[str] = []
        self.cold_to_warm_: dict[str, list[tuple[str, float]]] = {}

    def fit(
        self,
        cold_genes: set[str],
        warm_genes: set[str],
        pathway_features: pd.DataFrame,      # index=gene, columns=pathway membership
    ) -> "ColdGeneTransfer":
        """Build KNN mapping from cold to warm genes via pathway similarity."""
        from sklearn.neighbors import NearestNeighbors

        warm_list = sorted(warm_genes & set(pathway_features.index))
        cold_list = sorted(cold_genes & set(pathway_features.index))

        if not warm_list or not cold_list:
            print(f"  Cold transfer: no pathway features available "
                  f"(warm={len(warm_list)}, cold={len(cold_list)})")
            return self

        pw_warm = pathway_features.reindex(warm_list).fillna(0).to_numpy(dtype=np.float32)
        pw_cold = pathway_features.reindex(cold_list).fillna(0).to_numpy(dtype=np.float32)

        k_actual = min(self.k, len(warm_list))
        nbrs = NearestNeighbors(n_neighbors=k_actual, metric="cosine")
        nbrs.fit(pw_warm)
        distances, indices = nbrs.kneighbors(pw_cold)

        for i, cold_gene in enumerate(cold_list):
            sims = []
            for j in range(k_actual):
                warm_gene = warm_list[indices[i, j]]
                sim = max(1.0 - distances[i, j], 1e-6)  # cosine → similarity
                sims.append((warm_gene, sim))
            total = sum(s for _, s in sims)
            if total > 0:
                self.cold_to_warm_[cold_gene] = [(g, s / total) for g, s in sims]

        self.warm_genes_ = warm_list
        print(f"  Cold transfer: {len(self.cold_to_warm_)} cold genes → "
              f"{k_actual} warm neighbors each (via 149-dim pathway similarity)")
        return self

    def predict(
        self,
        cell_ids: np.ndarray,
        gene_ids: np.ndarray,
        cold_genes: set[str],
        warm_interaction: dict[tuple[str, str], float],  # (cell, warm_gene) → I
    ) -> np.ndarray:
        """Transfer interaction values from warm to cold genes."""
        preds = np.zeros(len(cell_ids), dtype=np.float32)
        for i, (c, g) in enumerate(zip(cell_ids, gene_ids)):
            if g not in cold_genes or g not in self.cold_to_warm_:
                continue
            sims = self.cold_to_warm_[g]
            weighted_sum = 0.0
            weight_total = 0.0
            for warm_g, sim in sims:
                val = warm_interaction.get((c, warm_g))
                if val is not None:
                    weighted_sum += sim * val
                    weight_total += sim
            if weight_total > 0:
                preds[i] = float(weighted_sum / weight_total)
        return preds


# ═══════════════════════════════════════════════════════════════════════════════
# Full Formula Training Pipeline (Empirical Bayes)
# ═══════════════════════════════════════════════════════════════════════════════

def train_formula_models(
    X: pd.DataFrame,
    y: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    expression: pd.DataFrame,
    gene_static_features: pd.DataFrame,
    gene_expr_profile_features: pd.DataFrame,
    cell_features: pd.DataFrame,
    gene_bl: dict[str, float] | None = None,
    cell_bl: dict[str, float] | None = None,
    svd_dot: dict[tuple[str, str], float] | None = None,
    cf_predictions: dict[tuple[str, str], float] | None = None,
    cold_genes: set[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train formula-based models with empirical Bayes smooth transition.

    Sequential fitting (NO hard warm/cold distinction):
      1. ShrinkageGeneFormula → μ̂_g = w_g·x̄_g + (1-w_g)·Φ(g)
      2. ShrinkageCellFormula → β̂_c = v_c·r̄_c + (1-v_c)·Ψ(c)
      3. Four interaction formulas on double residual
      4. InteractionBlend via RidgeCV
      5. ColdGeneTransfer for genes with n_g=0

    Args:
        X: full feature DataFrame.
        y: label array.
        cell_ids, gene_ids: identifiers for each row.
        expression: N_cells × P_genes expression DataFrame.
        gene_static_features: G1 features indexed by gene.
        gene_expr_profile_features: G2 features indexed by gene.
        cell_features: G3 features indexed by cell.
        gene_bl, cell_bl: unused (kept for interface compatibility).
        svd_dot: unused.
        cf_predictions: unused.
        cold_genes: set of cold gene symbols (n_g=0).
        config: configuration dict.

    Returns:
        dict with shrinkage models, formulas, blend, and metadata.
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

    # ── Build interaction feature arrays ──
    n_modules = 14
    indicator_names = [
        "OXPHOS_CI", "OXPHOS_CII_CIII", "OXPHOS_CIV_CV", "TCA_PYRUVATE",
        "FAO_LIPID", "AA_COFACTOR", "MITO_RIBOSOME", "mtDNA_RNA",
        "PROTEIN_IMPORT", "TRANSPORT", "REDOX_DETOX", "MITO_DYNAMICS",
        "CELL_DEATH", "SIGNALING",
    ]

    module_membership = np.zeros((n, n_modules), dtype=np.float32)
    for m in range(n_modules):
        g1_col = f"gene_module_{m:02d}"
        if g1_col in gene_static_features.columns:
            gene_to_val = gene_static_features[g1_col].to_dict()
            for i, g in enumerate(gene_ids):
                if g in gene_to_val:
                    module_membership[i, m] = float(gene_to_val[g])

    cell_indicators = np.zeros((n, n_modules), dtype=np.float32)
    for m, name in enumerate(indicator_names):
        col = f"cell_indicator_{name}"
        if col in cell_features.columns:
            cell_to_val = cell_features[col].to_dict()
            for i, c in enumerate(cell_ids):
                if c in cell_to_val:
                    cell_indicators[i, m] = float(cell_to_val[c])

    z_cg = np.zeros(n, dtype=np.float32)
    cell_to_expr_row = {c: i for i, c in enumerate(expression.index)}
    gene_to_expr_col = {g: i for i, g in enumerate(expression.columns)}
    for i, (c, g) in enumerate(zip(cell_ids, gene_ids)):
        if c in cell_to_expr_row and g in gene_to_expr_col:
            z_cg[i] = expression.iloc[cell_to_expr_row[c], gene_to_expr_col[g]]

    expr_percentile = np.zeros(n, dtype=np.float32)
    expr_arr = expression.to_numpy(dtype=np.float32)
    for i, (c, g) in enumerate(zip(cell_ids, gene_ids)):
        if c in cell_to_expr_row and g in gene_to_expr_col:
            cell_row = expr_arr[cell_to_expr_row[c]]
            expr_percentile[i] = float((cell_row < z_cg[i]).mean())

    evidence_weight = np.ones(n, dtype=np.float32)
    if "gene_evidence_weight" in gene_static_features.columns:
        ew_dict = gene_static_features["gene_evidence_weight"].to_dict()
        for i, g in enumerate(gene_ids):
            if g in ew_dict:
                evidence_weight[i] = float(ew_dict[g])

    # ── Step 4: Interaction Formulas ──
    interaction_preds = []
    interaction_names = []

    print("\n[Step 4a] Module×Indicator Interaction")
    i_mod_formula = ModuleInteractionFormula(
        alpha=fm_cfg.get("interaction_alpha", 1.0), n_modules=n_modules,
    )
    i_mod_formula.fit(cell_ids, gene_ids, r2, module_membership, cell_indicators)
    interaction_preds.append(i_mod_formula.predict(module_membership, cell_indicators))
    interaction_names.append("I_mod")

    print("\n[Step 4b] Expression→Dependency Effect")
    i_expr_formula = ExpressionEffectFormula(alpha=fm_cfg.get("expr_alpha", 1.0))
    i_expr_formula.fit(z_cg, r2)
    interaction_preds.append(i_expr_formula.predict(z_cg))
    interaction_names.append("I_expr")

    print("\n[Step 4c] Module-Weighted Expression Percentile")
    i_match_formula = ModuleMatchFormula(
        alpha=fm_cfg.get("match_alpha", 1.0), n_modules=n_modules,
    )
    i_match_formula.fit(module_membership, expr_percentile, r2)
    interaction_preds.append(i_match_formula.predict(module_membership, expr_percentile))
    interaction_names.append("I_match")

    print("\n[Step 4d] Evidence-Weighted Expression Coupling")
    i_ew_formula = EvidenceWeightedFormula()
    i_ew_formula.fit(evidence_weight, z_cg, r2)
    interaction_preds.append(i_ew_formula.predict(evidence_weight, z_cg))
    interaction_names.append("I_ew")

    # ── Step 5: Interaction Blend ──
    print("\n[Step 5] Interaction Blend via RidgeCV")
    blend = InteractionBlend(
        alphas=fm_cfg.get("blend_alphas", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]),
    )
    blend.fit(interaction_preds, r2, component_names=interaction_names)

    # ── Step 6: Cold Gene Transfer (for n_g=0 genes only) ──
    cold_transfer = None
    if cold_genes:
        print("\n[Step 6] Cold Gene Transfer (pathway-similarity KNN)")
        from .baselines import build_pw140_membership_features
        from pathlib import Path
        gene_meta = pd.read_csv(
            Path(config["paths"]["data_dir"]) / "metadata" / "gene_metadata.csv",
        )
        pathway_meta = pd.read_csv(
            Path(config["paths"]["data_dir"]) / "metadata" / "pathway_metadata.csv",
        )
        pw_features = build_pw140_membership_features(gene_meta, pathway_meta)
        warm_genes = set(shrink_gene.n_g_.keys()) - cold_genes

        cold_transfer = ColdGeneTransfer(k=fm_cfg.get("cold_knn", 20))
        cold_transfer.fit(cold_genes, warm_genes, pw_features)

    # ── Compute full predictions ──
    mu_arr = shrink_gene.predict_batch(list(gene_ids))
    beta_arr = shrink_cell.predict_batch(list(cell_ids))
    i_blend_arr = blend.predict(interaction_preds)
    full_preds = mu_arr + beta_arr + i_blend_arr

    # Cold gene interaction transfer (blend with formula interaction)
    if cold_transfer is not None and cold_genes:
        warm_interaction: dict[tuple[str, str], float] = {}
        for i in range(n):
            if gene_ids[i] not in cold_genes:
                warm_interaction[(cell_ids[i], gene_ids[i])] = float(i_blend_arr[i])

        i_transfer = cold_transfer.predict(cell_ids, gene_ids, cold_genes, warm_interaction)
        cold_mask = np.array([g in cold_genes for g in gene_ids])
        has_transfer = cold_mask & (i_transfer != 0)
        if has_transfer.any():
            full_preds[has_transfer] = (
                mu_arr[has_transfer] + beta_arr[has_transfer] + i_transfer[has_transfer]
            )

    # ── Print formula summary ──
    _print_shrinkage_summary(shrink_gene, shrink_cell,
                             i_mod_formula, i_expr_formula, i_match_formula, i_ew_formula,
                             blend)

    return {
        "shrink_gene": shrink_gene,
        "shrink_cell": shrink_cell,
        "i_mod_formula": i_mod_formula,
        "i_expr_formula": i_expr_formula,
        "i_match_formula": i_match_formula,
        "i_ew_formula": i_ew_formula,
        "interaction_blend": blend,
        "cold_transfer": cold_transfer,
        "mu_g": shrink_gene.mu_g_,
        "beta_c": shrink_cell.beta_c_,
        "cold_genes": cold_genes,
        "full_preds": full_preds,
    }


def predict_formula(
    X: pd.DataFrame,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    gene_bl: dict[str, float] | None = None,
    cell_bl: dict[str, float] | None = None,
    svd_dot: dict[tuple[str, str], float] | None = None,
    cf_predictions: dict[tuple[str, str], float] | None = None,
    cold_genes: set[str] | None = None,
    models: dict[str, Any] | None = None,
    add_jitter: bool = True,
) -> np.ndarray:
    """Generate predictions using trained formula models.

    Unified formula for ALL genes (no hard warm/cold distinction):
      ŷ(c,g) = μ̂_g + β̂_c + Blend[I_mod, I_expr, I_match, I_ew]

    where μ̂_g and β̂_c come from the empirical Bayes shrinkage models,
    which smoothly transition from data-dominated (warm) to prior-dominated (cold).
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

    # ── Interaction formulas ──
    n_modules = 14
    indicator_names = [
        "OXPHOS_CI", "OXPHOS_CII_CIII", "OXPHOS_CIV_CV", "TCA_PYRUVATE",
        "FAO_LIPID", "AA_COFACTOR", "MITO_RIBOSOME", "mtDNA_RNA",
        "PROTEIN_IMPORT", "TRANSPORT", "REDOX_DETOX", "MITO_DYNAMICS",
        "CELL_DEATH", "SIGNALING",
    ]

    module_membership = np.zeros((n, n_modules), dtype=np.float32)
    for m in range(n_modules):
        col = f"g1_gene_module_{m:02d}"
        if col in X.columns:
            module_membership[:, m] = X[col].to_numpy(dtype=np.float32)

    cell_indicators = np.zeros((n, n_modules), dtype=np.float32)
    for m, name in enumerate(indicator_names):
        col = f"g3_cell_indicator_{name}"
        if col in X.columns:
            cell_indicators[:, m] = X[col].to_numpy(dtype=np.float32)

    z_cg = np.zeros(n, dtype=np.float32)
    if "g4_pair_z_cg" in X.columns:
        z_cg = X["g4_pair_z_cg"].to_numpy(dtype=np.float32)

    expr_percentile = np.zeros(n, dtype=np.float32)
    if "g4_pair_expr_percentile" in X.columns:
        expr_percentile = X["g4_pair_expr_percentile"].to_numpy(dtype=np.float32)

    evidence_weight = np.ones(n, dtype=np.float32)
    if "g1_gene_evidence_weight" in X.columns:
        evidence_weight = X["g1_gene_evidence_weight"].to_numpy(dtype=np.float32)

    # Predict interaction formulas
    interaction_preds = []
    for key in ["i_mod_formula", "i_expr_formula", "i_match_formula", "i_ew_formula"]:
        formula = models.get(key)
        if formula is None:
            interaction_preds.append(np.zeros(n, dtype=np.float32))
        elif key == "i_mod_formula":
            interaction_preds.append(formula.predict(module_membership, cell_indicators))
        elif key == "i_expr_formula":
            interaction_preds.append(formula.predict(z_cg))
        elif key == "i_match_formula":
            interaction_preds.append(formula.predict(module_membership, expr_percentile))
        elif key == "i_ew_formula":
            interaction_preds.append(formula.predict(evidence_weight, z_cg))

    # Blend interactions
    blend = models.get("interaction_blend")
    i_blend = blend.predict(interaction_preds) if blend is not None else np.zeros(n, dtype=np.float32)

    # Cold gene transfer for n_g=0 genes
    cold_transfer = models.get("cold_transfer")
    if cold_transfer is not None and cold_genes:
        warm_interaction: dict[tuple[str, str], float] = {}
        for i in range(n):
            if gene_ids[i] not in cold_genes:
                warm_interaction[(cell_ids[i], gene_ids[i])] = float(i_blend[i])

        i_transfer = cold_transfer.predict(cell_ids, gene_ids, cold_genes, warm_interaction)
        cold_mask = np.array([g in cold_genes for g in gene_ids])
        has_transfer = cold_mask & (i_transfer != 0)
        if has_transfer.any():
            i_blend[has_transfer] = i_transfer[has_transfer]

    # ── Final prediction ──
    final = mu_arr + beta_arr + i_blend

    if add_jitter:
        rng = np.random.RandomState(42)
        final += rng.rand(len(final)).astype(np.float32) * 1e-6

    return final.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Formula Printout
# ═══════════════════════════════════════════════════════════════════════════════

def _print_shrinkage_summary(
    shrink_gene: "ShrinkageGeneFormula",
    shrink_cell: "ShrinkageCellFormula",
    i_mod: ModuleInteractionFormula,
    i_expr: ExpressionEffectFormula,
    i_match: ModuleMatchFormula,
    i_ew: EvidenceWeightedFormula,
    blend: InteractionBlend,
) -> None:
    """Print the human-readable formula with shrinkage details."""
    print("\n" + "=" * 70)
    print("COMPLETE PREDICTION FORMULA (Empirical Bayes)")
    print("=" * 70)
    print(f"""
ŷ(c,g) = [w_g·x̄_g + (1-w_g)·Φ(g)] + [v_c·r̄_c + (1-v_c)·Ψ(c)]
       + Blend[I_mod, I_expr, I_match, I_ew]

SHRINKAGE PARAMETERS:
  λ_gene = {shrink_gene.lambda_:.1f}  (gene prior strength)
  λ_cell = {shrink_cell.lambda_:.1f}  (cell prior strength)
  w_g = n_g/(n_g+λ_gene)  (gene evidence weight, 0=cold → 1=warm)
  v_c = m_c/(m_c+λ_cell)  (cell evidence weight)
""")
    print("─" * 70)
    print("GENE PRIOR Φ(g):")
    print(shrink_gene.gene_formula_.formula_str(top_n=6))
    print()
    print("─" * 70)
    print("CELL PRIOR Ψ(c):")
    print(shrink_cell.cell_formula_.formula_str(top_n=6))
    print()
    print("─" * 70)
    print("MODULE×INDICATOR INTERACTION I_mod:")
    print(i_mod.formula_str())
    print()
    print("─" * 70)
    print("EXPRESSION EFFECT I_expr:")
    print(i_expr.formula_str())
    print()
    print("─" * 70)
    print("INTERACTION BLEND:")
    for name, w in blend.get_weights():
        print(f"  α_{name} = {w:+.4f}")
    print(f"  intercept = {blend.intercept_:.4f}")
    print("=" * 70)
