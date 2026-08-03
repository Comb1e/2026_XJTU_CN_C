"""Formula-based interpretable gene dependency prediction.

Replaces all matrix-factorization and latent-variable models with explicit,
human-readable formulas where every coefficient has a named biological meaning.

Architecture (Chronos-inspired multiplicative decomposition):

    ŷ(c,g) = μ̂_g + β̂_c + Blend[I_mod, I_expr, I_match, I_ew]

where:
  - μ̂_g : Gene essentiality (Ridge on G1+G2 gene features)
  - β̂_c : Cell mitochondrial vulnerability (Ridge on G3 cell features)
  - I_mod : Module×Indicator interaction (14 coeffs, one per module)
  - I_expr : Asymmetric expression→dependency curve (4 coeffs)
  - I_match : Module-weighted expression percentile (14 coeffs)
  - I_ew : Evidence-weighted expression coupling (1 coeff)

Cold-start genes: μ̂_g from gene features (available for all genes) +
pathway-similarity KNN transfer for the interaction term.

Key references:
  - Chronos: r_cg = R*_cg/R_c − 1 (Dempster et al., Genome Biology 2021)
  - AC-Chronos: global median normalization (DepMap 23Q2+)
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
# Full Formula Training Pipeline
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
    gene_bl: dict[str, float],
    cell_bl: dict[str, float],
    svd_dot: dict[tuple[str, str], float] | None = None,
    cf_predictions: dict[tuple[str, str], float] | None = None,
    cold_genes: set[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train all formula-based models.

    Sequential fitting pipeline:
      1. GeneEssentialityFormula → μ̂_g
      2. CellVulnerabilityFormula → β̂_c (on residual y - μ̂_g)
      3. Four interaction formulas on double residual y - μ̂_g - β̂_c
      4. InteractionBlend via RidgeCV
      5. ColdGeneTransfer for cold-start genes

    Args:
        X: full feature DataFrame with g1_/g2_/g3_/g4_/g5_ prefixed columns.
        y: label array.
        cell_ids, gene_ids: identifiers for each row.
        expression: N_cells × P_genes expression DataFrame.
        gene_static_features: G1 features indexed by gene.
        gene_expr_profile_features: G2 features indexed by gene.
        cell_features: G3 features indexed by cell.
        gene_bl: gene → baseline dict (for comparison).
        cell_bl: cell → bias dict (for comparison).
        svd_dot: unused (formula model doesn't need SVD).
        cf_predictions: unused (formula model has its own cold-start).
        cold_genes: set of cold gene symbols.
        config: configuration dict.

    Returns:
        dict with trained formulas, blend model, and metadata.
    """
    if config is None:
        config = {}
    fm_cfg = config.get("prediction", {}).get("formula", {})
    if cold_genes is None:
        cold_genes = set()

    print("\n" + "=" * 70)
    print("FORMULA-BASED PREDICTION MODEL")
    print("=" * 70)

    n = len(y)
    unique_cells = sorted(set(cell_ids))
    unique_genes = sorted(set(gene_ids))
    cell_to_idx = {c: i for i, c in enumerate(unique_cells)}
    gene_to_idx = {g: i for i, g in enumerate(unique_genes)}

    # ── Extract feature columns by group ──
    all_feature_cols = [c for c in X.columns
                        if c not in ("cell_line_id", "perturbation_gene")]
    g1_cols = [c for c in all_feature_cols if c.startswith("g1_")]
    g2_cols = [c for c in all_feature_cols if c.startswith("g2_")]
    g3_cols = [c for c in all_feature_cols if c.startswith("g3_")]
    g4_cols = [c for c in all_feature_cols if c.startswith("g4_")]

    # ── Step 1: Gene Essentiality Formula ──
    print("\n[Formula 1] Gene Essentiality μ̂_g = f(gene_features)")
    # Build gene-level feature matrix from G1 + G2
    # G1 and G2 are gene-indexed DataFrames; join them
    gene_feat_df = gene_static_features.join(gene_expr_profile_features, how="inner")

    # Compute per-gene mean labels from training data
    gene_mean_labels: dict[str, float] = {}
    for g in unique_genes:
        mask = gene_ids == g
        if mask.sum() > 0:
            gene_mean_labels[g] = float(y[mask].mean())

    warm_genes_set = set(gene_mean_labels.keys()) - cold_genes if cold_genes else set(gene_mean_labels.keys())

    gene_formula = GeneEssentialityFormula(
        alpha=fm_cfg.get("gene_alpha", 1.0),
    )
    gene_formula.fit(gene_feat_df, gene_mean_labels, feature_names=list(gene_feat_df.columns))

    # Predict μ̂_g for ALL genes (warm + cold)
    all_genes_list = sorted(set(unique_genes) | cold_genes)
    gene_feat_all = gene_feat_df.reindex(all_genes_list).fillna(0.0)
    mu_g = gene_formula.predict(gene_feat_all)  # dict gene → μ̂_g

    # Fill missing genes
    for g in all_genes_list:
        if g not in mu_g:
            mu_g[g] = 0.0

    # ── Step 2: Cell Vulnerability Formula ──
    print("\n[Formula 2] Cell Vulnerability β̂_c = f(cell_features) on residual")
    # Compute residual after subtracting gene essentiality
    r1 = y.copy().astype(np.float64)
    for i in range(n):
        r1[i] -= mu_g.get(gene_ids[i], 0.0)

    # Compute per-cell mean residual
    cell_mean_residual: dict[str, float] = {}
    for c in unique_cells:
        mask = cell_ids == c
        if mask.sum() > 0:
            cell_mean_residual[c] = float(r1[mask].mean())

    cell_formula = CellVulnerabilityFormula(
        alpha=fm_cfg.get("cell_alpha", 10.0),
    )
    cell_formula.fit(cell_features, cell_mean_residual, feature_names=list(cell_features.columns))

    # Predict β̂_c for all cells
    beta_c = cell_formula.predict(cell_features)

    # Fill missing cells
    for c in unique_cells:
        if c not in beta_c:
            beta_c[c] = 0.0

    # ── Step 3: Double residual ──
    r2 = r1.copy()
    for i in range(n):
        r2[i] -= beta_c.get(cell_ids[i], 0.0)

    print(f"\n  Variance decomposition:")
    print(f"    σ²(y)           = {np.var(y):.6f}")
    print(f"    σ²(μ̂_g)         = {np.var([mu_g.get(g, 0.0) for g in gene_ids]):.6f}")
    print(f"    σ²(β̂_c)         = {np.var([beta_c.get(c, 0.0) for c in cell_ids]):.6f}")
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

    # Module membership: (N, 14) from gene_static_features (G1)
    module_membership = np.zeros((n, n_modules), dtype=np.float32)
    for m in range(n_modules):
        g1_col = f"gene_module_{m:02d}"
        if g1_col in gene_static_features.columns:
            gene_to_val = gene_static_features[g1_col].to_dict()
            for i, g in enumerate(gene_ids):
                if g in gene_to_val:
                    module_membership[i, m] = float(gene_to_val[g])

    # Cell indicators: (N, 14) from cell_features (G3)
    cell_indicators = np.zeros((n, n_modules), dtype=np.float32)
    for m, name in enumerate(indicator_names):
        col = f"cell_indicator_{name}"
        if col in cell_features.columns:
            cell_to_val = cell_features[col].to_dict()
            for i, c in enumerate(cell_ids):
                if c in cell_to_val:
                    cell_indicators[i, m] = float(cell_to_val[c])

    # Expression z-score: (N,) from expression matrix
    z_cg = np.zeros(n, dtype=np.float32)
    cell_to_expr_row = {c: i for i, c in enumerate(expression.index)}
    gene_to_expr_col = {g: i for i, g in enumerate(expression.columns)}
    for i, (c, g) in enumerate(zip(cell_ids, gene_ids)):
        if c in cell_to_expr_row and g in gene_to_expr_col:
            z_cg[i] = expression.iloc[cell_to_expr_row[c], gene_to_expr_col[g]]

    # Expression percentile: fraction of genes with lower expression in same cell
    expr_percentile = np.zeros(n, dtype=np.float32)
    expr_arr = expression.to_numpy(dtype=np.float32)
    for i, (c, g) in enumerate(zip(cell_ids, gene_ids)):
        if c in cell_to_expr_row and g in gene_to_expr_col:
            cell_row = expr_arr[cell_to_expr_row[c]]
            val = z_cg[i]
            expr_percentile[i] = float((cell_row < val).mean())

    # Evidence weight: (N,) from gene_static_features
    evidence_weight = np.ones(n, dtype=np.float32)
    if "gene_evidence_weight" in gene_static_features.columns:
        ew_dict = gene_static_features["gene_evidence_weight"].to_dict()
        for i, g in enumerate(gene_ids):
            if g in ew_dict:
                evidence_weight[i] = float(ew_dict[g])

    # ── Step 4: Interaction Formulas ──
    interaction_preds = []
    interaction_names = []

    # 4a: Module × Indicator Interaction
    print("\n[Formula 3a] Module×Indicator Interaction")
    i_mod_formula = ModuleInteractionFormula(
        alpha=fm_cfg.get("interaction_alpha", 1.0),
        n_modules=n_modules,
    )
    i_mod_formula.fit(cell_ids, gene_ids, r2, module_membership, cell_indicators)
    i_mod_preds = i_mod_formula.predict(module_membership, cell_indicators)
    interaction_preds.append(i_mod_preds)
    interaction_names.append("I_mod")

    # 4b: Expression Effect
    print("\n[Formula 3b] Expression→Dependency Effect")
    i_expr_formula = ExpressionEffectFormula(
        alpha=fm_cfg.get("expr_alpha", 1.0),
    )
    i_expr_formula.fit(z_cg, r2)
    i_expr_preds = i_expr_formula.predict(z_cg)
    interaction_preds.append(i_expr_preds)
    interaction_names.append("I_expr")

    # 4c: Module-Weighted Expression Percentile
    print("\n[Formula 3c] Module-Weighted Expression Percentile")
    i_match_formula = ModuleMatchFormula(
        alpha=fm_cfg.get("match_alpha", 1.0),
        n_modules=n_modules,
    )
    i_match_formula.fit(module_membership, expr_percentile, r2)
    i_match_preds = i_match_formula.predict(module_membership, expr_percentile)
    interaction_preds.append(i_match_preds)
    interaction_names.append("I_match")

    # 4d: Evidence-Weighted Expression
    print("\n[Formula 3d] Evidence-Weighted Expression Coupling")
    i_ew_formula = EvidenceWeightedFormula()
    i_ew_formula.fit(evidence_weight, z_cg, r2)
    i_ew_preds = i_ew_formula.predict(evidence_weight, z_cg)
    interaction_preds.append(i_ew_preds)
    interaction_names.append("I_ew")

    # ── Step 5: Interaction Blend ──
    print("\n[Formula 4] Interaction Blend via RidgeCV")
    blend = InteractionBlend(
        alphas=fm_cfg.get("blend_alphas", [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]),
    )
    blend.fit(interaction_preds, r2, component_names=interaction_names)

    # ── Step 6: Cold Gene Transfer ──
    cold_transfer = None
    if cold_genes:
        print("\n[Formula 5] Cold Gene Transfer (pathway-similarity KNN)")
        from .baselines import build_pw140_membership_features
        from pathlib import Path
        gene_meta = pd.read_csv(
            Path(config["paths"]["data_dir"]) / "metadata" / "gene_metadata.csv",
        )
        pathway_meta = pd.read_csv(
            Path(config["paths"]["data_dir"]) / "metadata" / "pathway_metadata.csv",
        )
        pw_features = build_pw140_membership_features(gene_meta, pathway_meta)

        cold_transfer = ColdGeneTransfer(k=fm_cfg.get("cold_knn", 20))
        cold_transfer.fit(cold_genes, warm_genes_set, pw_features)

    # ── Compute full predictions ──
    mu_arr = np.array([mu_g.get(g, 0.0) for g in gene_ids], dtype=np.float32)
    beta_arr = np.array([beta_c.get(c, 0.0) for c in cell_ids], dtype=np.float32)
    i_blend_arr = blend.predict(interaction_preds)

    full_preds = mu_arr + beta_arr + i_blend_arr

    # Cold gene correction: replace interaction with transfer
    if cold_transfer is not None and cold_genes:
        # Build warm interaction lookup
        warm_interaction: dict[tuple[str, str], float] = {}
        for i in range(n):
            if gene_ids[i] not in cold_genes:
                warm_interaction[(cell_ids[i], gene_ids[i])] = float(i_blend_arr[i])

        i_transfer = cold_transfer.predict(
            cell_ids, gene_ids, cold_genes, warm_interaction,
        )
        cold_mask = np.array([g in cold_genes for g in gene_ids])
        has_transfer = cold_mask & (i_transfer != 0)
        if has_transfer.any():
            # Blend: for cold genes with transfer, use transfer instead of formula interaction
            full_preds[has_transfer] = (
                mu_arr[has_transfer] + beta_arr[has_transfer] + i_transfer[has_transfer]
            )

    # ── Print full formula ──
    _print_formula_summary(gene_formula, cell_formula, i_mod_formula,
                           i_expr_formula, i_match_formula, i_ew_formula,
                           blend)

    return {
        "gene_formula": gene_formula,
        "cell_formula": cell_formula,
        "i_mod_formula": i_mod_formula,
        "i_expr_formula": i_expr_formula,
        "i_match_formula": i_match_formula,
        "i_ew_formula": i_ew_formula,
        "interaction_blend": blend,
        "cold_transfer": cold_transfer,
        "mu_g": mu_g,
        "beta_c": beta_c,
        "cold_genes": cold_genes,
        "full_preds": full_preds,
    }


def predict_formula(
    X: pd.DataFrame,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    gene_bl: dict[str, float],
    cell_bl: dict[str, float],
    svd_dot: dict[tuple[str, str], float] | None = None,
    cf_predictions: dict[tuple[str, str], float] | None = None,
    cold_genes: set[str] | None = None,
    models: dict[str, Any] | None = None,
    add_jitter: bool = True,
) -> np.ndarray:
    """Generate predictions using trained formula models.

    Prediction formula:
      ŷ(c,g) = μ̂_g + β̂_c + Blend[I_mod, I_expr, I_match, I_ew]

    For cold genes: interaction term transferred from pathway-similar warm genes.
    """
    if models is None:
        return np.zeros(len(cell_ids), dtype=np.float32)
    if cold_genes is None:
        cold_genes = set()

    n = len(cell_ids)

    # ── Gene essentiality ──
    mu_g = models.get("mu_g", {})
    mu_arr = np.array([mu_g.get(g, 0.0) for g in gene_ids], dtype=np.float32)

    # ── Cell vulnerability ──
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

    # Module membership: from G1 columns in X
    module_membership = np.zeros((n, n_modules), dtype=np.float32)
    for m in range(n_modules):
        col = f"g1_gene_module_{m:02d}"
        if col in X.columns:
            module_membership[:, m] = X[col].to_numpy(dtype=np.float32)

    # Cell indicators: from G3 columns in X (named by module, not index)
    cell_indicators = np.zeros((n, n_modules), dtype=np.float32)
    for m, name in enumerate(indicator_names):
        col = f"g3_cell_indicator_{name}"
        if col in X.columns:
            cell_indicators[:, m] = X[col].to_numpy(dtype=np.float32)

    # Expression z-score
    z_cg = np.zeros(n, dtype=np.float32)
    if "g4_pair_z_cg" in X.columns:
        z_cg = X["g4_pair_z_cg"].to_numpy(dtype=np.float32)

    # Expression percentile
    expr_percentile = np.zeros(n, dtype=np.float32)
    if "g4_pair_expr_percentile" in X.columns:
        expr_percentile = X["g4_pair_expr_percentile"].to_numpy(dtype=np.float32)

    # Evidence weight
    evidence_weight = np.ones(n, dtype=np.float32)
    if "g1_gene_evidence_weight" in X.columns:
        evidence_weight = X["g1_gene_evidence_weight"].to_numpy(dtype=np.float32)

    # Predict each interaction formula
    interaction_preds = []

    i_mod = models.get("i_mod_formula")
    if i_mod is not None:
        interaction_preds.append(i_mod.predict(module_membership, cell_indicators))
    else:
        interaction_preds.append(np.zeros(n, dtype=np.float32))

    i_expr = models.get("i_expr_formula")
    if i_expr is not None:
        interaction_preds.append(i_expr.predict(z_cg))
    else:
        interaction_preds.append(np.zeros(n, dtype=np.float32))

    i_match = models.get("i_match_formula")
    if i_match is not None:
        interaction_preds.append(i_match.predict(module_membership, expr_percentile))
    else:
        interaction_preds.append(np.zeros(n, dtype=np.float32))

    i_ew = models.get("i_ew_formula")
    if i_ew is not None:
        interaction_preds.append(i_ew.predict(evidence_weight, z_cg))
    else:
        interaction_preds.append(np.zeros(n, dtype=np.float32))

    # Blend interactions
    blend = models.get("interaction_blend")
    if blend is not None:
        i_blend = blend.predict(interaction_preds)
    else:
        i_blend = np.zeros(n, dtype=np.float32)

    # Cold gene transfer
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
        jitter = rng.rand(len(final)).astype(np.float32) * 1e-6
        final += jitter

    return final.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Formula Printout
# ═══════════════════════════════════════════════════════════════════════════════

def _print_formula_summary(
    gene_formula: GeneEssentialityFormula,
    cell_formula: CellVulnerabilityFormula,
    i_mod: ModuleInteractionFormula,
    i_expr: ExpressionEffectFormula,
    i_match: ModuleMatchFormula,
    i_ew: EvidenceWeightedFormula,
    blend: InteractionBlend,
) -> None:
    """Print the complete human-readable prediction formula."""
    print("\n" + "=" * 70)
    print("COMPLETE PREDICTION FORMULA")
    print("=" * 70)
    print("""
ŷ(c,g) = μ̂_g + β̂_c + Blend[I_mod, I_expr, I_match, I_ew]

where:
""")
    print("─" * 70)
    print("GENE ESSENTIALITY μ̂_g:")
    print(gene_formula.formula_str(top_n=6))
    print()
    print("─" * 70)
    print("CELL VULNERABILITY β̂_c:")
    print(cell_formula.formula_str(top_n=6))
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
    print("MODULE-MATCH I_match:")
    print(i_match.formula_str())
    print()
    print("─" * 70)
    print("EVIDENCE-WEIGHTED I_ew:")
    print(i_ew.formula_str())
    print()
    print("─" * 70)
    print("INTERACTION BLEND:")
    weights = blend.get_weights()
    for name, w in weights:
        print(f"  α_{name} = {w:+.4f}")
    print(f"  intercept = {blend.intercept_:.4f}")
    print("=" * 70)
