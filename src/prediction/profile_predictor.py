"""Multi-Output Gene Profile Predictor for cold-start gene dependency.

Core idea (Chang & Zhang, 2023):
  Instead of predicting a scalar gene mean Φ(g) (cell-invariant),
  predict the FULL dependency profile across all cells.

  ŷ_profile(g) = X_gene(g) @ B    where B ∈ R^{f_g × n_cells}

  Cold genes get cell-specific predictions directly from gene features,
  breaking the cell-invariant bottleneck that limits cold-gene ranking.

Architecture:
  1. PCA on label matrix Y (genes × cells) → reduce to n_components PCs
  2. Multi-output Ridge: gene_features → PC scores
  3. Inverse PCA transform → full profile predictions
  4. Blend: ŷ_final = (1-λ)·ŷ_formula + λ·ŷ_profile

References:
  - Chang & Zhang (2023): "Ridge regression baseline outperforms DeepDEP"
    Multi-output Ridge ρ=0.88 vs DeepDEP ρ=0.87, per-gene ρ=0.276 vs 0.137
  - Ahlmann-Eltze et al. (2024): Linear models match/beat deep learning
    for unseen gene perturbation prediction
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler


class MultiOutputGeneProfile:
    """Multi-output Ridge predicting full gene dependency profiles.

    y_profile(g, c) = Σ_k PC_score_k(g) · PC_loading_k(c)

    where PC_score_k(g) = Ridge_k(gene_features(g)).

    This gives cold genes CELL-SPECIFIC predictions from gene features alone.
    """

    def __init__(
        self,
        n_components: int = 50,
        ridge_alphas: tuple = (0.1, 1.0, 10.0, 100.0, 1000.0),
        random_state: int = 42,
    ):
        """
        Args:
            n_components: number of PCA components for output compression.
            ridge_alphas: alpha grid for RidgeCV per component.
            random_state: random seed.
        """
        self.n_components = n_components
        self.ridge_alphas = ridge_alphas
        self.random_state = random_state

        # Fitted state
        self.pca_: PCA | None = None
        self.ridge_models_: list[RidgeCV] = []  # one per PC
        self.scaler_X_: StandardScaler | None = None
        self.scaler_Y_: StandardScaler | None = None
        self.feature_names_: list[str] = []
        self.gene_index_: pd.Index | None = None
        self.cell_index_: pd.Index | None = None
        self.global_mean_: float = 0.0

        # Performance
        self.oof_r2_: float = 0.0
        self.per_component_r2_: list[float] = []

    def fit(
        self,
        gene_features: pd.DataFrame,
        labels: pd.DataFrame,
        n_folds: int = 5,
        verbose: bool = True,
    ) -> "MultiOutputGeneProfile":
        """Fit multi-output profile predictor.

        Args:
            gene_features: DataFrame indexed by gene_symbol, G1+G2+PW140 features.
            labels: DataFrame with [cell_line_id, perturbation_gene, label].
            n_folds: CV folds for OOF evaluation.
            verbose: print progress.
        """
        # ── Build gene×cell label matrix ──
        cells = sorted(labels["cell_line_id"].unique())
        genes_with_labels = sorted(set(labels["perturbation_gene"].unique()))

        self.cell_index_ = pd.Index(cells)
        self.global_mean_ = float(labels["label"].mean())

        # Build label matrix: genes × cells
        cell_to_idx = {c: i for i, c in enumerate(cells)}
        n_cells = len(cells)
        n_genes_labeled = len(genes_with_labels)

        Y = np.full((n_genes_labeled, n_cells), np.nan, dtype=np.float64)
        gene_to_row = {g: i for i, g in enumerate(genes_with_labels)}

        for _, row in labels.iterrows():
            c = row["cell_line_id"]
            g = row["perturbation_gene"]
            if c in cell_to_idx and g in gene_to_row:
                Y[gene_to_row[g], cell_to_idx[c]] = row["label"]

        # Fill NaN with global mean
        Y = np.where(np.isnan(Y), self.global_mean_, Y)

        # ── Align gene features ──
        common_genes = sorted(set(genes_with_labels) & set(gene_features.index))
        self.gene_index_ = pd.Index(common_genes)
        X = gene_features.reindex(common_genes).to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        Y_aligned = Y[[gene_to_row[g] for g in common_genes]]

        self.feature_names_ = list(gene_features.columns)

        # ── Standardize ──
        self.scaler_X_ = StandardScaler()
        X_scaled = self.scaler_X_.fit_transform(X)

        self.scaler_Y_ = StandardScaler()
        Y_scaled = self.scaler_Y_.fit_transform(Y_aligned)

        # ── PCA on labels ──
        k = min(self.n_components, min(Y_scaled.shape) - 1)
        self.pca_ = PCA(n_components=k, random_state=self.random_state)
        Y_pca = self.pca_.fit_transform(Y_scaled)

        if verbose:
            var_total = np.sum(self.pca_.explained_variance_ratio_)
            print(f"  [Profile] PCA: {k} components, "
                  f"{var_total:.1%} variance explained")

        # ── Multi-output Ridge: one Ridge per PC ──
        from sklearn.model_selection import cross_val_score

        self.ridge_models_ = []
        self.per_component_r2_ = []

        for comp in range(k):
            y_comp = Y_pca[:, comp]
            ridge = RidgeCV(
                alphas=self.ridge_alphas,
                store_cv_results=False,
            )
            ridge.fit(X_scaled, y_comp)

            # OOF R² for this component
            oof_r2 = cross_val_score(
                ridge, X_scaled, y_comp,
                cv=min(n_folds, len(common_genes)),
                scoring="r2",
            ).mean()

            self.ridge_models_.append(ridge)
            self.per_component_r2_.append(float(oof_r2))

        # Overall OOF R²
        Y_pred_pca = np.column_stack([
            model.predict(X_scaled) for model in self.ridge_models_
        ])
        Y_pred_scaled = self.pca_.inverse_transform(Y_pred_pca)
        Y_pred = self.scaler_Y_.inverse_transform(Y_pred_scaled)

        ss_res = np.sum((Y_aligned - Y_pred) ** 2)
        ss_tot = np.sum((Y_aligned - self.global_mean_) ** 2)
        self.oof_r2_ = 1.0 - ss_res / max(ss_tot, 1e-12)

        if verbose:
            mean_pc_r2 = np.mean(self.per_component_r2_)
            print(f"  [Profile] OOF R² = {self.oof_r2_:.4f}, "
                  f"mean per-PC R² = {mean_pc_r2:.4f}")

        return self

    def predict_profile(
        self,
        gene_features: pd.DataFrame,
        genes: list[str],
    ) -> np.ndarray:
        """Predict full dependency profile for given genes.

        Args:
            gene_features: DataFrame indexed by gene_symbol.
            genes: list of gene symbols to predict for.

        Returns:
            array (n_genes, n_cells) of predicted profiles.
        """
        if not self.ridge_models_:
            raise ValueError("Model not fitted. Call fit() first.")

        X = gene_features.reindex(genes).to_numpy(dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)
        X_scaled = self.scaler_X_.transform(X)

        # Predict PC scores
        pc_scores = np.column_stack([
            model.predict(X_scaled) for model in self.ridge_models_
        ])

        # Inverse PCA + scaling
        Y_scaled = self.pca_.inverse_transform(pc_scores)
        Y = self.scaler_Y_.inverse_transform(Y_scaled)

        return Y

    def predict(
        self,
        gene_features: pd.DataFrame,
        cell_ids: np.ndarray,
        gene_ids: np.ndarray,
    ) -> np.ndarray:
        """Predict dependency for specific (cell, gene) pairs.

        Args:
            gene_features: DataFrame indexed by gene_symbol.
            cell_ids: (N,) cell identifiers.
            gene_ids: (N,) gene identifiers.

        Returns:
            (N,) predicted values from profile model.
        """
        unique_genes = sorted(set(gene_ids))
        profiles = self.predict_profile(gene_features, unique_genes)

        gene_to_row = {g: i for i, g in enumerate(unique_genes)}
        cell_to_col = {c: i for i, c in enumerate(self.cell_index_)}

        preds = np.full(len(cell_ids), self.global_mean_, dtype=np.float64)
        for i in range(len(cell_ids)):
            g = gene_ids[i]
            c = cell_ids[i]
            gi = gene_to_row.get(g)
            ci = cell_to_col.get(c)
            if gi is not None and ci is not None:
                preds[i] = profiles[gi, ci]

        return preds

    def get_top_gene_loadings(
        self, component: int = 0, top_n: int = 20,
    ) -> list[tuple[str, float]]:
        """Return genes with largest |loading| in a PCA component of label space."""
        if self.pca_ is None or component >= self.pca_.n_components_:
            return []
        loadings = self.pca_.components_[component]  # (n_cells,)
        idx = np.argsort(-np.abs(loadings))[:top_n]
        return [(self.cell_index_[i], float(loadings[i])) for i in idx]

    def formula_str(self) -> str:
        """Human-readable summary."""
        lines = [
            "Multi-Output Gene Profile Predictor",
            f"  ŷ_profile(g, c) = Σ_k PC_score_k(g) · PC_loading_k(c)",
            f"  PCA components: {len(self.ridge_models_)}",
            f"  OOF R²: {self.oof_r2_:.4f}",
            f"  Per-component OOF R²: "
            f"{', '.join(f'{r:.3f}' for r in self.per_component_r2_[:5])}...",
        ]
        return "\n".join(lines)
