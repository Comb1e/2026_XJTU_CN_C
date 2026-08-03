"""White-box interpretable models for gene dependency prediction.

Replaces black-box tree ensembles (HGBR, XGBoost, LightGBM) with
fully interpretable linear/algebraic models:

  Component 1 — Factor Analysis on residual label matrix
  Component 2 — Sparse ElasticNet on pair features (L1+L2)
  Component 3 — PCA-Ridge on full feature set
  Component 4 — Canonical Correlation Analysis (CCA)
  Component 5 — Spline-GAM for key nonlinear features
  Component 6 — RidgeCV blend of all components

All components produce interpretable coefficients, loadings, or curves.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.linalg import svd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import FactorAnalysis, PCA, TruncatedSVD
from sklearn.linear_model import ElasticNetCV, Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════════
# Component 1: Factor Analysis on residual label matrix
# ═══════════════════════════════════════════════════════════════════════════════

class FactorAnalysisModel:
    """Factor Analysis on the residual label matrix R(c,g) = y - ŷ_base.

    Decomposes cell×gene interactions into K interpretable latent factors.
    Each factor k has:
      - cell_scores[:, k]: how strongly each cell expresses factor k
      - gene_loadings[:, k]: how strongly each gene loads on factor k
      - component_variance[k]: variance explained by factor k

    Prediction: ŷ_FA(c,g) = Σ_k cell_score(c,k) × gene_loading(g,k)
    """

    def __init__(
        self,
        n_components: int = 20,
        random_state: int = 42,
        max_iter: int = 2000,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.max_iter = max_iter
        self.fa_: FactorAnalysis | None = None
        self.cell_scores_: np.ndarray | None = None       # (n_cells, K)
        self.gene_loadings_: np.ndarray | None = None     # (n_genes, K)
        self.cell_index_: pd.Index | None = None
        self.gene_index_: pd.Index | None = None
        self.global_mean_: float = 0.0

    def fit(
        self,
        residuals: np.ndarray,            # (n_cells, n_genes) residual matrix
        cell_index: pd.Index,
        gene_index: pd.Index,
        cold_genes: set[str] | None = None,
        gene_static_features: pd.DataFrame | None = None,
        gene_expr_profile_features: pd.DataFrame | None = None,
    ) -> "FactorAnalysisModel":
        """Fit FA to the residual matrix. Cold genes have loadings imputed."""
        n_cells, n_genes = residuals.shape
        K = min(self.n_components, min(n_cells, n_genes) - 1, 50)

        # Standardize residuals per gene (gene-wise centering)
        self.global_mean_ = float(np.nanmean(residuals))
        R = residuals - self.global_mean_
        # Fill NaN with 0 (missing pairs)
        R = np.nan_to_num(R, nan=0.0)

        # FA fits to genes as "samples" and cells as "features" (transposed)
        # We want gene loadings, so genes are "features" dimension
        self.fa_ = FactorAnalysis(
            n_components=K, random_state=self.random_state,
            max_iter=self.max_iter, tol=1e-4,
        )
        # Fit on cell×gene: cells=samples, genes=features
        self.gene_loadings_ = self.fa_.fit_transform(R.T).astype(np.float32)  # (n_genes, K)
        # Cell scores via minimum mean squared error estimation
        # R ≈ C @ L^T → C ≈ R @ L @ (L^T @ L)^{-1}
        L = self.gene_loadings_
        LtL = L.T @ L
        LtL_inv = np.linalg.pinv(LtL + 1e-4 * np.eye(K))
        self.cell_scores_ = (R @ L @ LtL_inv).astype(np.float32)  # (n_cells, K)

        self.cell_index_ = cell_index
        self.gene_index_ = gene_index

        # Impute loadings for cold genes via Ridge
        if cold_genes and gene_static_features is not None and gene_expr_profile_features is not None:
            self._impute_cold_gene_loadings(
                cold_genes, gene_static_features, gene_expr_profile_features,
            )

        return self

    def _impute_cold_gene_loadings(
        self,
        cold_genes: set[str],
        gene_static_features: pd.DataFrame,
        gene_expr_profile_features: pd.DataFrame,
    ) -> None:
        """Impute factor loadings for cold-start genes using gene features."""
        warm_genes = sorted(set(self.gene_index_) - cold_genes)
        cold_list = sorted(cold_genes & set(gene_static_features.index))

        if not warm_genes or not cold_list:
            return

        gene_feats = gene_static_features.join(gene_expr_profile_features, how="inner")
        X_warm = gene_feats.reindex(warm_genes).to_numpy(dtype=np.float32)
        X_cold = gene_feats.reindex(cold_list).to_numpy(dtype=np.float32)
        X_warm = np.nan_to_num(X_warm, nan=0.0)
        X_cold = np.nan_to_num(X_cold, nan=0.0)

        warm_to_row = {g: i for i, g in enumerate(warm_genes)}
        gene_to_row = {g: i for i, g in enumerate(self.gene_index_)}
        Y_warm = np.array(
            [self.gene_loadings_[gene_to_row[g]] for g in warm_genes],
            dtype=np.float32,
        )

        K = self.gene_loadings_.shape[1]
        preds = np.zeros((len(cold_list), K), dtype=np.float32)
        for k in range(K):
            ridge = Ridge(alpha=1.0)
            ridge.fit(X_warm, Y_warm[:, k])
            preds[:, k] = ridge.predict(X_cold).astype(np.float32)

        # Append to existing arrays
        old_gene_index = list(self.gene_index_)
        self.gene_loadings_ = np.vstack([self.gene_loadings_, preds])
        self.gene_index_ = pd.Index(old_gene_index + cold_list)

    def predict(
        self,
        cell_ids: list[str],
        gene_ids: list[str],
    ) -> np.ndarray:
        """Predict FA component for (cell, gene) pairs."""
        cell_to_row = {c: i for i, c in enumerate(self.cell_index_)}
        gene_to_row = {g: i for i, g in enumerate(self.gene_index_)}

        preds = np.full(len(cell_ids), self.global_mean_, dtype=np.float32)
        for i, (c, g) in enumerate(zip(cell_ids, gene_ids)):
            if c in cell_to_row and g in gene_to_row:
                preds[i] = float(np.dot(
                    self.cell_scores_[cell_to_row[c]],
                    self.gene_loadings_[gene_to_row[g]],
                )) + self.global_mean_

        return preds

    def get_top_genes_per_factor(self, k: int, top_n: int = 20) -> list[tuple[str, float]]:
        """Return top-N genes by absolute loading for factor k."""
        if self.gene_loadings_ is None or self.gene_index_ is None:
            return []
        loadings = self.gene_loadings_[:, k]
        idx = np.argsort(-np.abs(loadings))[:top_n]
        return [(self.gene_index_[i], float(loadings[i])) for i in idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Component 2: Sparse ElasticNet on pair features
# ═══════════════════════════════════════════════════════════════════════════════

class SparseElasticNetModel:
    """L1+L2 regularized linear model on pair features (G4).

    Replaces tree-based Model A with an interpretable linear model.
    L1 penalty drives sparsity → only the most important features get
    nonzero coefficients, making the model directly interpretable.

    Uses ElasticNetCV for automatic hyperparameter selection.
    """

    def __init__(
        self,
        l1_ratio: float = 0.5,
        n_alphas: int = 50,
        max_iter: int = 5000,
        random_state: int = 42,
    ):
        self.l1_ratio = l1_ratio
        self.n_alphas = n_alphas
        self.max_iter = max_iter
        self.random_state = random_state
        self.model_: ElasticNetCV | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.n_nonzero_: int = 0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> "SparseElasticNetModel":
        """Fit sparse ElasticNet to training data."""
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        alphas = np.logspace(-3, 1, self.n_alphas)
        self.model_ = ElasticNetCV(
            l1_ratio=[self.l1_ratio],
            alphas=alphas,
            max_iter=self.max_iter,
            random_state=self.random_state,
            cv=3,
            n_jobs=-1,
        )
        self.model_.fit(X_scaled, y)

        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]

        self.n_nonzero_ = int(np.sum(np.abs(self.model_.coef_) > 1e-6))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using sparse ElasticNet."""
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return self.model_.predict(X).astype(np.float32)

    def get_top_features(self, top_n: int = 30) -> list[tuple[str, float]]:
        """Return features with largest absolute coefficients."""
        if self.model_ is None:
            return []
        coef = self.model_.coef_
        idx = np.argsort(-np.abs(coef))[:top_n]
        return [(self.feature_names_[i], float(coef[i])) for i in idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Component 3: PCA-Ridge regression
# ═══════════════════════════════════════════════════════════════════════════════

class PCARidgeModel:
    """PCA dimensionality reduction followed by Ridge regression.

    Reduces the ~220 feature space to K principal components,
    then fits Ridge regression on those components.
    Fully linear and interpretable: coefficients can be mapped back
    to original feature space via PCA loadings.

    ŷ_PCA = w_ridge · PCA(X)
    ∂ŷ/∂x_j = Σ_k w_k · PCA_loading(k, j)
    """

    def __init__(
        self,
        n_components: int = 50,
        ridge_alpha: float = 1.0,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.ridge_alpha = ridge_alpha
        self.random_state = random_state
        self.pca_: PCA | None = None
        self.ridge_: Ridge | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.feature_importance_: np.ndarray | None = None  # back-mapped to original space

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "PCARidgeModel":
        """Fit PCA + Ridge."""
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]

        n_comp = min(self.n_components, X.shape[0], X.shape[1])
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.pca_ = PCA(n_components=n_comp, random_state=self.random_state)
        X_pca = self.pca_.fit_transform(X_scaled)

        self.ridge_ = Ridge(alpha=self.ridge_alpha)
        self.ridge_.fit(X_pca, y)

        # Back-map feature importance to original space
        # imp_j = || Σ_k w_k · PCA_loading(k,j) ||  (RMS across PCs weighted by Ridge coef)
        loadings = self.pca_.components_  # (K, D)
        coef = self.ridge_.coef_         # (K,)
        self.feature_importance_ = np.abs(loadings.T @ coef)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using PCA + Ridge."""
        if self.scaler_ is not None:
            X_scaled = self.scaler_.transform(X)
        else:
            X_scaled = X
        X_pca = self.pca_.transform(X_scaled)
        return self.ridge_.predict(X_pca).astype(np.float32)

    def get_top_features(self, top_n: int = 30) -> list[tuple[str, float]]:
        """Return features with largest back-mapped importance."""
        if self.feature_importance_ is None:
            return []
        idx = np.argsort(-self.feature_importance_)[:top_n]
        return [(self.feature_names_[i], float(self.feature_importance_[i])) for i in idx]

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        """PCA explained variance ratios."""
        if self.pca_ is None:
            return np.array([])
        return self.pca_.explained_variance_ratio_


# ═══════════════════════════════════════════════════════════════════════════════
# Component 4: Partial Least Squares (PLS) Regression
# ═══════════════════════════════════════════════════════════════════════════════

class PLSModel:
    """Partial Least Squares regression combining cell+gene features.

    Unlike CCA (which maximizes feature-feature correlation), PLS directly
    optimizes latent dimensions for covariance with the target variable.
    Each PLS component is a linear combination of input features that
    maximally covaries with the residual dependency score.

    ŷ_PLS = PLS(cell_features || gene_features → residual)
    """

    def __init__(
        self,
        n_components: int = 20,
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.random_state = random_state
        self.pls_: PLSRegression | None = None
        self.scaler_: StandardScaler | None = None
        self.feature_names_: list[str] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "PLSModel":
        """Fit PLS regression to predict residuals from features."""
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]

        n_comp = min(self.n_components, X.shape[0] - 1, X.shape[1])
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.pls_ = PLSRegression(
            n_components=n_comp,
            scale=False,  # already scaled
            max_iter=1000,
            tol=1e-6,
        )
        self.pls_.fit(X_scaled, y.reshape(-1, 1))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using PLS."""
        if self.scaler_ is not None:
            X = self.scaler_.transform(X)
        return self.pls_.predict(X).ravel().astype(np.float32)

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance as ||coef_|| across PLS components."""
        if self.pls_ is None:
            return np.array([])
        # coef_ shape: (n_features, 1) after fitting
        return np.abs(self.pls_.coef_.ravel())

    @property
    def n_components_used(self) -> int:
        if self.pls_ is None:
            return 0
        return self.pls_.n_components


# ═══════════════════════════════════════════════════════════════════════════════
# Component 5: Alternating Least Squares (ALS) Matrix Factorization
# ═══════════════════════════════════════════════════════════════════════════════

class ALSModel:
    """Alternating Least Squares matrix factorization on residual matrix.

    Decomposes R(c,g) ≈ U_c · V_g + b_c + b_g + μ with L2 regularization.
    Unlike SVD (which requires complete matrix), ALS only fits observed entries,
    making it naturally robust to missing (cell, gene) pairs.

    Each latent dimension k has:
      - cell_factors[:, k]: cell c's score on factor k
      - gene_factors[:, k]: gene g's loading on factor k
      - cell_bias[c]: per-cell offset
      - gene_bias[g]: per-gene offset

    Fully interpretable: predictions are ŷ(c,g) = Σ_k U_{c,k}·V_{g,k} + b_c + b_g + μ
    """

    def __init__(
        self,
        n_factors: int = 50,
        regularization: float = 0.1,
        max_iter: int = 30,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.regularization = regularization
        self.max_iter = max_iter
        self.random_state = random_state
        self.cell_factors_: np.ndarray | None = None
        self.gene_factors_: np.ndarray | None = None
        self.cell_bias_: np.ndarray | None = None
        self.gene_bias_: np.ndarray | None = None
        self.global_mean_: float = 0.0
        self.cell_index_: pd.Index | None = None
        self.gene_index_: pd.Index | None = None

    def fit(
        self,
        residuals: np.ndarray,
        cell_index: pd.Index,
        gene_index: pd.Index,
    ) -> "ALSModel":
        """Fit ALS to the residual matrix using alternating Ridge regression."""
        n_cells, n_genes = residuals.shape
        K = min(self.n_factors, min(n_cells, n_genes) - 1)
        rng = np.random.RandomState(self.random_state)

        # Fill NaN with 0 for initialization (missing pairs)
        R = np.nan_to_num(residuals, nan=0.0).astype(np.float64)
        # Create mask of observed entries
        mask = ~np.isnan(residuals)

        self.global_mean_ = float(np.nanmean(residuals))
        R_centered = R - self.global_mean_

        # Initialize factors randomly
        U = rng.randn(n_cells, K).astype(np.float64) * 0.01
        V = rng.randn(n_genes, K).astype(np.float64) * 0.01
        b_c = np.zeros(n_cells, dtype=np.float64)
        b_g = np.zeros(n_genes, dtype=np.float64)

        lam = self.regularization

        for iteration in range(self.max_iter):
            # Update gene factors V (Ridge: solve for each gene)
            for g in range(n_genes):
                observed = mask[:, g]
                if observed.sum() < 2:
                    continue
                U_obs = U[observed]
                R_obs = R_centered[observed, g] - b_c[observed] - b_g[g]
                A = U_obs.T @ U_obs + lam * np.eye(K)
                b_vec = U_obs.T @ R_obs
                try:
                    V[g] = np.linalg.solve(A, b_vec)
                except np.linalg.LinAlgError:
                    V[g] = np.linalg.lstsq(A, b_vec, rcond=None)[0]

            # Update cell factors U (Ridge: solve for each cell)
            for c in range(n_cells):
                observed = mask[c, :]
                if observed.sum() < 2:
                    continue
                V_obs = V[observed]
                R_obs = R_centered[c, observed] - b_c[c] - b_g[observed]
                A = V_obs.T @ V_obs + lam * np.eye(K)
                b_vec = V_obs.T @ R_obs
                try:
                    U[c] = np.linalg.solve(A, b_vec)
                except np.linalg.LinAlgError:
                    U[c] = np.linalg.lstsq(A, b_vec, rcond=None)[0]

            # Update biases
            for c in range(n_cells):
                observed = mask[c, :]
                if observed.sum() > 0:
                    b_c[c] = np.mean(
                        R_centered[c, observed] - U[c] @ V[observed].T - b_g[observed]
                    )
            for g in range(n_genes):
                observed = mask[:, g]
                if observed.sum() > 0:
                    b_g[g] = np.mean(
                        R_centered[observed, g] - U[observed] @ V[g] - b_c[observed]
                    )

        self.cell_factors_ = U.astype(np.float32)
        self.gene_factors_ = V.astype(np.float32)
        self.cell_bias_ = b_c.astype(np.float32)
        self.gene_bias_ = b_g.astype(np.float32)
        self.cell_index_ = cell_index
        self.gene_index_ = gene_index

        return self

    def predict(
        self,
        cell_ids: list[str],
        gene_ids: list[str],
    ) -> np.ndarray:
        """Predict ALS component for (cell, gene) pairs."""
        cell_to_row = {c: i for i, c in enumerate(self.cell_index_)}
        gene_to_row = {g: i for i, g in enumerate(self.gene_index_)}

        preds = np.full(len(cell_ids), self.global_mean_, dtype=np.float32)
        for i, (c, g) in enumerate(zip(cell_ids, gene_ids)):
            if c in cell_to_row and g in gene_to_row:
                ci = cell_to_row[c]
                gi = gene_to_row[g]
                preds[i] = float(
                    np.dot(self.cell_factors_[ci], self.gene_factors_[gi])
                    + self.cell_bias_[ci] + self.gene_bias_[gi]
                    + self.global_mean_
                )
        return preds

    def get_top_genes_per_factor(self, k: int, top_n: int = 20) -> list[tuple[str, float]]:
        """Return top-N genes by absolute loading for factor k."""
        if self.gene_factors_ is None or self.gene_index_ is None:
            return []
        loadings = self.gene_factors_[:, k]
        idx = np.argsort(-np.abs(loadings))[:top_n]
        return [(self.gene_index_[i], float(loadings[i])) for i in idx]


# ═══════════════════════════════════════════════════════════════════════════════
# Component 6: Spline-GAM for key nonlinear features
# ═══════════════════════════════════════════════════════════════════════════════

class SplineGAMModel:
    """Spline-based Generalized Additive Model for key nonlinear features.

    Fits univariate smoothing splines for the most important features
    and combines them additively: ŷ = Σ f_j(x_j).

    Each spline f_j can be plotted to show the (potentially nonlinear)
    relationship between feature j and gene dependency.
    """

    def __init__(
        self,
        max_features: int = 10,
        spline_smooth: float = 0.5,
        random_state: int = 42,
    ):
        self.max_features = max_features
        self.spline_smooth = spline_smooth
        self.random_state = random_state
        self.selected_features_: list[int] = []
        self.splines_: list[UnivariateSpline | None] = []
        self.feature_means_: np.ndarray | None = None
        self.feature_stds_: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self.intercept_: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str] | None = None,
    ) -> "SplineGAMModel":
        """Select top features by correlation with target, fit univariate splines."""
        if feature_names is not None:
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"feat_{i}" for i in range(X.shape[1])]

        n, d = X.shape
        # Standardize features for stable correlation computation
        self.feature_means_ = X.mean(axis=0)
        self.feature_stds_ = X.std(axis=0)
        self.feature_stds_[self.feature_stds_ < 1e-12] = 1.0

        # Select features by absolute Pearson correlation with target
        X_centered = X - self.feature_means_
        corrs = np.abs(np.dot(y - y.mean(), X_centered) /
                       (np.std(y) * self.feature_stds_ * n))
        self.selected_features_ = list(np.argsort(-corrs)[:self.max_features])

        # Fit univariate splines
        residuals = y.copy().astype(np.float64)
        for feat_idx in self.selected_features_:
            x = X[:, feat_idx].astype(np.float64)
            # Sort for spline fitting
            order = np.argsort(x)
            x_sorted = x[order]
            r_sorted = residuals[order]

            try:
                spline = UnivariateSpline(
                    x_sorted, r_sorted,
                    s=self.spline_smooth * len(x),
                    k=3,  # cubic
                    ext='const',
                )
                residuals -= spline(x)
                self.splines_.append(spline)
            except Exception:
                self.splines_.append(None)
                continue

        # Fit remaining residual as intercept
        self.intercept_ = float(np.mean(residuals))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using additive spline model."""
        preds = np.full(len(X), self.intercept_, dtype=np.float32)
        for feat_idx, spline in zip(self.selected_features_, self.splines_):
            if spline is not None:
                preds += spline(X[:, feat_idx].astype(np.float64)).astype(np.float32)
        return preds

    def get_partial_dependence(
        self, feat_idx: int, x_range: tuple[float, float] | None = None,
        n_points: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (x_grid, y_values) for partial dependence of feature feat_idx."""
        if feat_idx not in self.selected_features_:
            return np.array([]), np.array([])
        pos = self.selected_features_.index(feat_idx)
        spline = self.splines_[pos]
        if spline is None:
            return np.array([]), np.array([])

        if x_range is None:
            # Use training data range
            x_range = (-3.0, 3.0)  # fallback for standardized features
        x_grid = np.linspace(x_range[0], x_range[1], n_points)
        y_vals = spline(x_grid) + self.intercept_ / max(len(self.selected_features_), 1)
        return x_grid.astype(np.float32), y_vals.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Component 6: RidgeCV Blend
# ═══════════════════════════════════════════════════════════════════════════════

class RidgeBlend:
    """Optimal RidgeCV blending of multiple white-box components.

    Learns: ŷ = α₀ + Σ_j α_j · ŷ_component_j

    The Ridge coefficients α_j give the relative importance of each
    white-box component. This is the ONLY learned weighting — no
    black-box transformations are involved.

    Unlike the previous ensemble which blended in rank space with
    arbitrary α values, this learns the optimal linear combination
    that minimizes RMSE (which also optimizes for ranking via the
    monotone relationship between scale and rank).
    """

    def __init__(
        self,
        alphas: list[float] | None = None,
        fit_intercept: bool = True,
    ):
        if alphas is None:
            alphas = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
        self.alphas = alphas
        self.fit_intercept = fit_intercept
        self.model_: RidgeCV | None = None
        self.component_names_: list[str] = []
        self.coefficients_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(
        self,
        component_preds: list[np.ndarray],  # list of (N,) arrays, one per component
        y: np.ndarray,
        component_names: list[str] | None = None,
    ) -> "RidgeBlend":
        """Fit RidgeCV to find optimal component weights."""
        if component_names is not None:
            self.component_names_ = list(component_names)
        else:
            self.component_names_ = [f"comp_{i}" for i in range(len(component_preds))]

        X = np.column_stack([p.astype(np.float64) for p in component_preds])
        self.model_ = RidgeCV(
            alphas=self.alphas,
            fit_intercept=self.fit_intercept,
            store_cv_results=False,
        )
        self.model_.fit(X, y.astype(np.float64))

        self.coefficients_ = self.model_.coef_
        self.intercept_ = float(self.model_.intercept_) if self.fit_intercept else 0.0
        return self

    def predict(self, component_preds: list[np.ndarray]) -> np.ndarray:
        """Blend component predictions."""
        X = np.column_stack([p.astype(np.float64) for p in component_preds])
        return self.model_.predict(X).astype(np.float32)

    def get_component_weights(self) -> list[tuple[str, float]]:
        """Return component names and their learned weights."""
        if self.coefficients_ is None:
            return []
        return list(zip(self.component_names_, self.coefficients_))

    @property
    def alpha_(self) -> float:
        """Selected Ridge alpha."""
        if self.model_ is None:
            return 0.0
        return float(self.model_.alpha_)


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration: Quantile mapping to fix prediction scale
# ═══════════════════════════════════════════════════════════════════════════════

class QuantileCalibrator:
    """Monotone quantile mapping to match prediction and label distributions.

    Maps predictions through the empirical CDF^{-1} of training labels.
    This is monotone → zero risk to ranking metrics (Spearman, NDCG, Precision),
    while directly optimizing the RMSE component by matching the label marginal.
    """

    def __init__(self, n_quantiles: int = 1000):
        self.n_quantiles = n_quantiles
        self.quantiles_: np.ndarray | None = None
        self.label_quantiles_: np.ndarray | None = None

    def fit(self, y_pred: np.ndarray, y_true: np.ndarray) -> "QuantileCalibrator":
        """Learn the quantile mapping."""
        self.quantiles_ = np.linspace(0.0, 1.0, self.n_quantiles)
        self.label_quantiles_ = np.quantile(y_true, self.quantiles_)
        # Ensure monotonicity (paranoid)
        self.label_quantiles_ = np.maximum.accumulate(self.label_quantiles_)
        return self

    def transform(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply quantile mapping."""
        if self.quantiles_ is None:
            return y_pred
        return np.interp(
            np.clip(y_pred, 0.0, 1.0),
            # Map preds to [0,1] first via their own quantiles
            np.quantile(y_pred, self.quantiles_),
            self.label_quantiles_,
        ).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# Full white-box training pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def train_whitebox_models(
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
    svd_dot: dict[tuple[str, str], float],
    cf_predictions: dict[tuple[str, str], float] | None = None,
    cold_genes: set[str] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train all white-box models and blend them.

    Args:
        X: full feature DataFrame.
        y: label array.
        cell_ids, gene_ids: identifiers for each row.
        expression: N_cells × P_genes expression DataFrame.
        gene_static_features: G1 features indexed by gene.
        gene_expr_profile_features: G2 features indexed by gene.
        cell_features: G3 features indexed by cell.
        gene_bl: gene → baseline dict.
        cell_bl: cell → bias dict.
        svd_dot: (cell, gene) → SVD dot product dict.
        cf_predictions: (cell, gene) → CF prediction dict.
        cold_genes: set of cold gene symbols.
        config: configuration dict.

    Returns:
        dict with trained components, blend model, and metadata.
    """
    if config is None:
        config = {}
    wb_cfg = config.get("prediction", {}).get("whitebox", {})
    if cold_genes is None:
        cold_genes = set()

    n = len(y)
    components = {}
    component_preds = []

    # ── Extract feature groups ──
    g1_cols = [c for c in X.columns if c.startswith("g1_")]
    g2_cols = [c for c in X.columns if c.startswith("g2_")]
    g3_cols = [c for c in X.columns if c.startswith("g3_")]
    g4_cols = [c for c in X.columns if c.startswith("g4_")]
    g5_cols = [c for c in X.columns if c.startswith("g5_")]

    X_g1g2 = X[g1_cols + g2_cols].to_numpy(dtype=np.float32)
    X_g3 = X[g3_cols].to_numpy(dtype=np.float32)
    X_g4 = X[g4_cols].to_numpy(dtype=np.float32)
    X_full = X[[c for c in X.columns if c not in ("cell_line_id", "perturbation_gene")]].to_numpy(dtype=np.float32)
    X_full = np.nan_to_num(X_full, nan=0.0)
    feature_names = [c for c in X.columns if c not in ("cell_line_id", "perturbation_gene")]

    # ── Additive baseline (Component 0) ──
    print("  [WB-0] Computing additive baseline...")
    from .baselines import predict_additive
    gene_arr = np.array([gene_bl.get(g, 0.0) for g in gene_ids], dtype=np.float32)
    cell_arr = np.array([cell_bl.get(c, 0.0) for c in cell_ids], dtype=np.float32)
    base_preds = gene_arr + cell_arr

    svd_vals = np.array([
        svd_dot.get((c, g), 0.0)
        for c, g in zip(cell_ids, gene_ids)
    ], dtype=np.float32)

    cf_vals = np.full(n, np.nan, dtype=np.float32)
    if cf_predictions is not None:
        for i, (c, g) in enumerate(zip(cell_ids, gene_ids)):
            if (c, g) in cf_predictions:
                cf_vals[i] = cf_predictions[(c, g)]

    add_preds = predict_additive_from_arrays(
        gene_arr, cell_arr, svd_vals, cf_vals,
        cell_ids, gene_ids, cold_genes,
        svd_weight=wb_cfg.get("svd_weight", 0.3),
        cf_weight=wb_cfg.get("cf_weight", 0.6),
    )
    components["additive"] = {"preds": add_preds}
    component_preds.append(add_preds)

    # ── Factor Analysis (Component 1) ──
    print("  [WB-1] Fitting Factor Analysis on residual matrix...")
    fa_cfg = wb_cfg.get("factor_analysis", {})
    residuals = y - add_preds

    # Build residual matrix
    unique_cells = sorted(set(cell_ids))
    unique_genes = sorted(set(gene_ids))
    cell_to_row = {c: i for i, c in enumerate(unique_cells)}
    gene_to_col = {g: i for i, g in enumerate(unique_genes)}
    R = np.full((len(unique_cells), len(unique_genes)), np.nan, dtype=np.float64)
    for i in range(n):
        R[cell_to_row[cell_ids[i]], gene_to_col[gene_ids[i]]] = residuals[i]

    fa_model = FactorAnalysisModel(
        n_components=fa_cfg.get("n_components", 20),
        random_state=42,
        max_iter=fa_cfg.get("max_iter", 2000),
    )
    fa_model.fit(
        R,
        cell_index=pd.Index(unique_cells),
        gene_index=pd.Index(unique_genes),
        cold_genes=cold_genes,
        gene_static_features=gene_static_features,
        gene_expr_profile_features=gene_expr_profile_features,
    )
    fa_preds = fa_model.predict(list(cell_ids), list(gene_ids))
    components["factor_analysis"] = {"preds": fa_preds, "model": fa_model}
    component_preds.append(fa_preds)

    # ── Sparse ElasticNet (Component 2) ──
    print("  [WB-2] Fitting Sparse ElasticNet on pair features...")
    enet_cfg = wb_cfg.get("elasticnet", {})
    enet_model = SparseElasticNetModel(
        l1_ratio=enet_cfg.get("l1_ratio", 0.5),
        n_alphas=enet_cfg.get("n_alphas", 50),
        max_iter=enet_cfg.get("max_iter", 5000),
        random_state=42,
    )
    # Fit on residuals after additive model
    feature_cols_for_enet = g4_cols
    X_enet = X[feature_cols_for_enet].to_numpy(dtype=np.float64)
    enet_model.fit(X_enet, residuals, feature_names=feature_cols_for_enet)
    enet_preds = enet_model.predict(X_enet)
    enet_nz = enet_model.n_nonzero_
    print(f"    ElasticNet: {enet_nz}/{len(feature_cols_for_enet)} nonzero coefficients "
          f"(alpha={enet_model.model_.alpha_:.4f})")
    components["elasticnet"] = {"preds": enet_preds, "model": enet_model}
    component_preds.append(enet_preds)

    # ── PCA-Ridge (Component 3) ──
    print("  [WB-3] Fitting PCA-Ridge on full features...")
    pca_cfg = wb_cfg.get("pca_ridge", {})
    pca_ridge = PCARidgeModel(
        n_components=pca_cfg.get("n_components", 50),
        ridge_alpha=pca_cfg.get("ridge_alpha", 1.0),
        random_state=42,
    )
    pca_ridge.fit(X_full, residuals, feature_names=feature_names)
    pca_preds = pca_ridge.predict(X_full)
    pca_var = pca_ridge.explained_variance_ratio_.sum()
    print(f"    PCA-Ridge: {pca_ridge.n_components} components, "
          f"cumulative var={pca_var:.3f}")
    components["pca_ridge"] = {"preds": pca_preds, "model": pca_ridge}
    component_preds.append(pca_preds)

    # ── PLS (Component 4) ──
    print("  [WB-4] Fitting PLS on combined features...")
    pls_cfg = wb_cfg.get("pls", {})
    pls_model = PLSModel(
        n_components=pls_cfg.get("n_components", 20),
        random_state=42,
    )
    pls_model.fit(X_full, residuals, feature_names=feature_names)
    pls_preds = pls_model.predict(X_full)
    print(f"    PLS: {pls_model.n_components_used} components used")
    components["pls"] = {"preds": pls_preds, "model": pls_model}
    component_preds.append(pls_preds)

    # ── ALS Matrix Factorization (Component 5) ──
    print("  [WB-5] Fitting ALS on residual matrix...")
    als_cfg = wb_cfg.get("als", {})
    als_model = ALSModel(
        n_factors=als_cfg.get("n_factors", 50),
        regularization=als_cfg.get("regularization", 0.1),
        max_iter=als_cfg.get("max_iter", 30),
        random_state=42,
    )
    als_model.fit(R, cell_index=pd.Index(unique_cells), gene_index=pd.Index(unique_genes))
    als_preds = als_model.predict(list(cell_ids), list(gene_ids))
    components["als"] = {"preds": als_preds, "model": als_model}
    component_preds.append(als_preds)

    # ── Spline-GAM (Component 6) ──
    print("  [WB-6] Fitting Spline-GAM on key features...")
    spline_cfg = wb_cfg.get("spline_gam", {})
    # Select most important features across groups
    top_feat_indices = _select_top_features_for_spline(
        X_full, residuals, feature_names, max_features=spline_cfg.get("max_features", 10),
    )
    X_top = X_full[:, top_feat_indices]
    top_names = [feature_names[i] for i in top_feat_indices]
    spline_model = SplineGAMModel(
        max_features=len(top_feat_indices),
        spline_smooth=spline_cfg.get("smooth", 0.5),
        random_state=42,
    )
    spline_model.fit(X_top, residuals, feature_names=top_names)
    spline_preds = spline_model.predict(X_top)
    components["spline_gam"] = {"preds": spline_preds, "model": spline_model}
    component_preds.append(spline_preds)

    # ── Ridge Blend (Component 7: final combiner) ──
    print("  [WB-7] Blending all white-box components via RidgeCV...")
    # Ensure no NaN in component predictions before blending
    component_preds_clean = [np.nan_to_num(p, nan=0.0) for p in component_preds]
    blend = RidgeBlend(
        alphas=wb_cfg.get("blend_alphas", [0.1, 1.0, 10.0, 100.0, 1000.0]),
        fit_intercept=True,
    )
    comp_names = ["additive", "factor_analysis", "elasticnet", "pca_ridge", "pls", "als", "spline_gam"]
    blend.fit(component_preds_clean, y, component_names=comp_names)
    weights = blend.get_component_weights()
    print(f"    Blend weights: {[(n, f'{w:.4f}') for n, w in weights]}")
    print(f"    Ridge alpha: {blend.alpha_:.4f}")

    # Full training prediction
    blended_preds = blend.predict(component_preds_clean)

    # Calibration: quantile map blended predictions to label distribution
    print("  Fitting quantile calibration...")
    calibrator = QuantileCalibrator(n_quantiles=1000)
    calibrator.fit(blended_preds, y)
    calibrated_preds = calibrator.transform(blended_preds)
    print(f"    Calibrated: mean {calibrated_preds.mean():.4f} → {y.mean():.4f} (target)")

    return {
        "components": components,
        "blend": blend,
        "calibrator": calibrator,
        "component_names": comp_names,
        "blended_preds": blended_preds,
        "calibrated_preds": calibrated_preds,
        "cold_genes": cold_genes,
    }


def predict_whitebox(
    X: pd.DataFrame,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    gene_bl: dict[str, float],
    cell_bl: dict[str, float],
    svd_dot: dict[tuple[str, str], float],
    cf_predictions: dict[tuple[str, str], float] | None,
    cold_genes: set[str],
    models: dict[str, Any],
    add_jitter: bool = True,
) -> np.ndarray:
    """Generate predictions using trained white-box models.

    Args:
        X: full feature DataFrame.
        cell_ids, gene_ids: identifiers for each row.
        gene_bl, cell_bl: baseline dicts.
        svd_dot: SVD dot product dict.
        cf_predictions: CF prediction dict.
        cold_genes: set of cold gene symbols.
        models: dict from train_whitebox_models().
        add_jitter: add tiny deterministic jitter to avoid ties.

    Returns:
        Array of predicted labels.
    """
    wb_cfg = {}
    n = len(cell_ids)

    # Extract feature matrices
    g1_cols = [c for c in X.columns if c.startswith("g1_")]
    g2_cols = [c for c in X.columns if c.startswith("g2_")]
    g3_cols = [c for c in X.columns if c.startswith("g3_")]
    g4_cols = [c for c in X.columns if c.startswith("g4_")]
    feature_names = [c for c in X.columns if c not in ("cell_line_id", "perturbation_gene")]
    X_full = X[feature_names].to_numpy(dtype=np.float32)
    # Fill NaN with 0 (missing features treated as neutral)
    X_full = np.nan_to_num(X_full, nan=0.0)

    components = models["components"]
    comp_preds = []

    # Component 0: Additive baseline
    gene_arr = np.array([gene_bl.get(g, 0.0) for g in gene_ids], dtype=np.float32)
    cell_arr = np.array([cell_bl.get(c, 0.0) for c in cell_ids], dtype=np.float32)
    svd_vals = np.array([
        svd_dot.get((c, g), 0.0)
        for c, g in zip(cell_ids, gene_ids)
    ], dtype=np.float32)
    cf_vals = np.full(n, np.nan, dtype=np.float32)
    if cf_predictions is not None:
        for i, (c, g) in enumerate(zip(cell_ids, gene_ids)):
            if (c, g) in cf_predictions:
                cf_vals[i] = cf_predictions[(c, g)]
    add_preds = predict_additive_from_arrays(
        gene_arr, cell_arr, svd_vals, cf_vals,
        cell_ids, gene_ids, cold_genes,
    )
    comp_preds.append(add_preds)

    # Component 1: Factor Analysis
    fa_model = components.get("factor_analysis", {}).get("model")
    if fa_model is not None:
        fa_preds = fa_model.predict(list(cell_ids), list(gene_ids))
    else:
        fa_preds = np.zeros(n, dtype=np.float32)
    comp_preds.append(fa_preds)

    # Component 2: ElasticNet
    enet_model = components.get("elasticnet", {}).get("model")
    if enet_model is not None:
        X_enet = X[g4_cols].to_numpy(dtype=np.float64)
        X_enet = np.nan_to_num(X_enet, nan=0.0)
        enet_preds = enet_model.predict(X_enet)
    else:
        enet_preds = np.zeros(n, dtype=np.float32)
    comp_preds.append(enet_preds)

    # Component 3: PCA-Ridge
    pca_ridge = components.get("pca_ridge", {}).get("model")
    if pca_ridge is not None:
        pca_preds = pca_ridge.predict(X_full)
    else:
        pca_preds = np.zeros(n, dtype=np.float32)
    comp_preds.append(pca_preds)

    # Component 4: PLS
    pls_model = components.get("pls", {}).get("model")
    if pls_model is not None:
        pls_preds = pls_model.predict(X_full)
    else:
        pls_preds = np.zeros(n, dtype=np.float32)
    comp_preds.append(pls_preds)

    # Component 5: ALS
    als_model = components.get("als", {}).get("model")
    if als_model is not None:
        als_preds = als_model.predict(list(cell_ids), list(gene_ids))
    else:
        als_preds = np.zeros(n, dtype=np.float32)
    comp_preds.append(als_preds)

    # Component 6: Spline-GAM
    spline_model = components.get("spline_gam", {}).get("model")
    if spline_model is not None and spline_model.selected_features_:
        top_indices = spline_model.selected_features_
        X_top = X_full[:, top_indices]
        spline_preds = spline_model.predict(X_top)
    else:
        spline_preds = np.zeros(n, dtype=np.float32)
    comp_preds.append(spline_preds)

    # Blend — ensure no NaN in component predictions
    comp_preds_clean = [np.nan_to_num(p, nan=0.0) for p in comp_preds]
    blend = models["blend"]
    final = blend.predict(comp_preds_clean)

    # Calibration: match label distribution
    calibrator = models.get("calibrator")
    if calibrator is not None:
        final = calibrator.transform(final)

    # Add deterministic jitter to avoid ties
    if add_jitter:
        rng = np.random.RandomState(42)
        jitter = rng.rand(len(final)).astype(np.float32) * 1e-6
        final += jitter

    return final.astype(np.float32)


# ── Helpers ─────────────────────────────────────────────────────────────────

def predict_additive_from_arrays(
    gene_vals: np.ndarray,
    cell_vals: np.ndarray,
    svd_vals: np.ndarray,
    cf_vals: np.ndarray,
    cell_ids: np.ndarray,
    gene_ids: np.ndarray,
    cold_genes: set[str],
    svd_weight: float = 0.3,
    cf_weight: float = 0.6,
) -> np.ndarray:
    """Predict using additive model from pre-computed arrays."""
    preds = gene_vals + cell_vals

    # Add SVD correction
    preds += svd_weight * svd_vals

    # Blend CF for cold genes
    cold_mask = np.array([g in cold_genes for g in gene_ids])
    has_cf = ~np.isnan(cf_vals)
    blend_mask = cold_mask & has_cf
    if blend_mask.any():
        preds[blend_mask] = (1.0 - cf_weight) * preds[blend_mask] + cf_weight * cf_vals[blend_mask]

    return preds


def _select_top_features_for_spline(
    X: np.ndarray,
    residuals: np.ndarray,
    feature_names: list[str],
    max_features: int = 10,
) -> list[int]:
    """Select features most correlated with residuals for spline fitting."""
    n, d = X.shape
    # Pearson correlation
    X_centered = X - X.mean(axis=0)
    r_centered = residuals - residuals.mean()
    X_std = X.std(axis=0)
    X_std[X_std < 1e-12] = 1.0
    corrs = np.abs(np.dot(r_centered, X_centered) /
                   (np.std(residuals) * X_std * n))
    return list(np.argsort(-corrs)[:max_features])
