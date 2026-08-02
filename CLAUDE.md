# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

2026 XJTU CN C contest — "Mitochondrial-related gene functional dependency modeling and prediction."
Two problems: (1) build low-dimensional interpretable mitochondrial expression indicators from 1,140 cell lines × 1,123 genes, (2) predict gene dependency scores (Chronos CRISPR fitness effects) for (cell_line, gene) pairs.

**Data lives in `数据文件/` (Chinese dir name), not `data/`.** All paths are relative to this directory, defined in `config.yaml`.

## Commands

```bash
# Problem 1: compute mitochondrial expression module indicators → outputs/
python run.py                              # --task indicators (default)

# Problem 2: train models and generate submission.csv
python run.py --task predict

# Problem 2: cross-validation (group-by-gene + group-by-cell)
python run.py --task validate

# Problem 2: generate interpretability outputs
python run.py --task interpret

# Local validation (train/val split + official scoring script)
python local_validate.py                   # full 80/20 split
python local_validate.py --n-genes 200     # quick test with 200 genes

# Run all tests (106 total)
python -m pytest tests/ -v

# Run specific test files
python -m pytest tests/test_prediction_metrics.py -v
python -m pytest tests/test_prediction_features.py -v
python -m pytest tests/test_pipeline.py -v
```

**Requirements**: `numpy>=1.24`, `scipy>=1.10`, `pandas>=1.5`, `scikit-learn>=1.2`, `pyyaml>=6`, `pytest>=7`. No PyTorch/LightGBM — the competition environment is sklearn-only.

## Architecture

### Problem 1 Pipeline (`src/` — stages in `src/indicators.py::run_pipeline`)

1. **`preprocess.py`**: Loads data, builds gene→module mapping from MitoCarta3.0 pathway annotations (149 hierarchical pathways → 14 modules), computes evidence weights from 6 MitoCarta evidence scores.
2. **`scoring/ewm.py`**: Evidence-Weighted Mean per module. Closed-form knockout delta: Δ_k = −w_g·z_{c,g}/(Σw_j − w_g·𝕀(g∈G_k)).
3. **`scoring/res.py`**: Rank-based Enrichment Score (GSEA running-sum statistic, probit-normalized).
4. **`scoring/spca.py`**: Sparse PCA projection scores with optional gene set refinement.
5. **`ensemble.py`**: Weighted fusion: γ_ewm·EWM + γ_res·RES + γ_spca·SPCA (default 0.50/0.25/0.25). CELL_DEATH module (index 12) boosted by (1+α) factor (α=0.25).
6. **`orthogonalize.py`**: Löwdin symmetric orthogonalization (S = Σ^{−1/2}) — preserves biological meaning (diagonal-dominant transform).
7. **`derived.py`**: 10 pathway imbalance indices computed pre-orthogonalization (OXPHOS_vs_TCA, BCL2 balance, mtDNA transcription-replication coupling, etc.). Lineage-conditioned z-scores per OncotreeLineage.
8. **`knockout.py`**: Gene knockout response Δ_k(g,c) — sampled computation for validation.

**14 modules** (defined in `preprocess.py::MODULE_DEFINITIONS`): OXPHOS_CI(0), OXPHOS_CII_CIII(1), OXPHOS_CIV_CV(2), TCA_PYRUVATE(3), FAO_LIPID(4), AA_COFACTOR(5), MITO_RIBOSOME(6), mtDNA_RNA(7), PROTEIN_IMPORT(8), TRANSPORT(9), REDOX_DETOX(10), MITO_DYNAMICS(11), CELL_DEATH(12), SIGNALING(13).

### Problem 2 Pipeline (`src/prediction/` — separate package, Problem 1 code unchanged)

**Feature engineering** (`features.py`): ~220 features in 5 groups:
- **G1** (~43): Gene static — module membership (14-dim one-hot), sub-mitochondrial location, evidence scores, pathway count
- **G2** (~8): Gene expression profile — mean/std/quartiles of z-score across all 1,140 cells
- **G3** (~115): Cell state — Problem 1 outputs (14 indicators + 10 derived + 14 lineage-conditioned + 42 raw EWM/RES/SPCA + 20 pathway PCA + 29 lineage one-hot)
- **G4** (~44): Pair features — z_{c,g}, expression percentile, vectorized Δ_EWM (14), Δ_SPCA (14), module-match (14), lineage×module interaction
- **G5** (~7): Collaborative — gene baseline μ̂_g (shrunk mean or teacher-predicted), cell bias β̂_c, SVD latent dot product, neighbor essentiality (k-NN). **Leakage-controlled**: all out-of-fold.

**Models** (`models.py`):
- **Model A**: `HistGradientBoostingRegressor` on G1-G5 (excluding neighbor_score). Cold-start safe — serves all pairs.
- **Model B**: HGBR with all G5 features. Only for genes with training labels.
- **Model C**: Pairwise `LogisticRegression` (Bradley-Terry ranking). Optional, default off.
- **Blending**: Rank-space weighted average per cell. Cold genes get α_B=0 (Model A only).
- **Calibration**: `RidgeCV` on [1, ŷ, μ̂_g, β̂_c, ŷ²] — decouples ranking (monotone-invariant) from RMSE.

**Cold-start handling** (`baselines.py`): 165 test genes have no training labels. Gene baseline μ̂_g predicted by teacher model (RidgeCV) from G1+G2 features. Module-level curated priors (MITO_RIBOSOME > CELL_DEATH > SIGNALING) provide Bayesian shrinkage for low-evidence genes. Primary CV protocol is group-by-gene 5-fold, which directly measures cold-start generalization.

### Scoring Metric (`src/prediction/metrics.py`)

Exact replica of official `数据文件/calculate_metric.py`:
```
S = 100 × (0.3×Spearman + 0.3×nDCG + 0.25×Precision@K + 0.15×RMSE)
```
All metrics per cell line, macro-averaged. Ranks use `method="average"`, ties broken by row `_order`. nDCG uses graded relevance `rel = max(K − true_rank + 1, 0)`. RMSE normalized by global σ_y (ddof=0). Parity verified to 1e-9.

## Key Design Decisions

- **Config-driven**: All tunable parameters in `config.yaml`. No hardcoded values. The `prediction:` config section controls all Problem 2 hyperparameters.
- **Problem 1 outputs as Problem 2 inputs**: Problem 2 reads CSVs from `outputs/` (produced by Problem 1). If Problem 1 is re-run, Problem 2 picks up the new indicators.
- **Unified entry point**: `run.py --task` dispatches to Problem 1 (`indicators`), Problem 2 (`predict`/`validate`/`interpret`). Default `indicators` preserves backward compatibility.
- **UTF-8 everywhere**: Chinese directory names (`数据文件/`, `参考文献/`) must be handled as raw paths. `src/utils.py::resolve_data_path` uses `Path` which handles this natively.
- **Sub-mitochondrial localization fallback**: ~100 genes lack MitoCarta pathway annotations; assigned to modules via `SUBMITO_FALLBACK` dict in `preprocess.py`.
- **Pathway name conversion**: `preprocess.py::_leaf_to_pw_key()` converts human-readable pathway names ("CI subunits") to metadata keys ("CI_subunits") by replacing spaces with underscores and removing commas.

## Reference Papers (`参考文献/`)

8 papers as Markdown summaries: MitoCarta3.0 [4], Chronos [3], DepMap [2], Mitochondria & cell death [1], Mitochondrial heterogeneity [5], mtDNA transcription [7], Omics approaches [6], Mitochondrial proteome research [8]. Numbers correspond to citations in the competition problem statement.
