"""Main pipeline for computing mitochondrial expression module indicators.

Orchestrates all stages:
  1. Load and validate data
  2. Build gene-module mapping
  3. Compute three scoring methods (EWM, RES, SPCA)
  4. Ensemble fusion
  5. Orthogonalization
  6. Gene knockout response (sampled)
  7. Output CSV files
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .preprocess import (
    MODULE_DEFINITIONS,
    load_all_data,
    validate_data,
    build_gene_module_map,
    compute_evidence_weights,
)
from .scoring.ewm import compute_ewm_scores
from .scoring.res import compute_res_scores
from .scoring.spca import compute_spca_scores
from .ensemble import fuse_ensemble
from .orthogonalize import orthogonalize
from .knockout import compute_knockout_summary


def run_pipeline(config: dict[str, Any]) -> dict[str, Any]:
    """Run the full indicator computation pipeline.

    Args:
        config: configuration dictionary (from config.yaml)

    Returns:
        dict with keys:
            - indicators: DataFrame of orthogonalized indicators (N×K)
            - indicators_raw: DataFrame before orthogonalization
            - ewm_scores, res_scores, spca_scores: individual method scores
            - pathway_scores: DataFrame of 149 pathway-level scores (if enabled)
            - knockout_summary: DataFrame of knockout response samples
            - transform_matrix: S matrix from orthogonalization
            - gene_module_map: gene-to-module mapping dictionary
            - evidence_weights: gene-to-weight dictionary
            - spca_loadings: module-to-gene-loadings dictionary
    """
    # ── Stage 0: Load data ──
    print("Loading data...")
    data = load_all_data(config)
    issues = validate_data(data)
    if issues:
        print("Data validation issues:")
        for issue in issues:
            print(f"  - {issue}")

    expression = data["expression"]
    gene_meta = data["gene_meta"]
    gene_module_map = data["gene_module_map"]
    evidence_weights = data["evidence_weights"]

    n_cells, n_genes = expression.shape
    n_modules = config["modules"]["K"]
    print(f"  Loaded: {n_cells} cell lines × {n_genes} genes, {n_modules} modules")

    # ── Stage 1: Module definitions ──
    print("\nModule definitions:")
    genes_per_module = {i: 0 for i in range(n_modules)}
    for g, info in gene_module_map.items():
        for m in info["modules"]:
            genes_per_module[m] += 1
    for i in range(n_modules):
        mod = MODULE_DEFINITIONS[i]
        print(f"  [{i:02d}] {mod['name']}: {genes_per_module[i]} genes — {mod['description']}")

    # ── Stage 2: Optional gene set refinement ──
    refinement_config = config.get("gene_set_refinement", {})
    if refinement_config.get("enabled", False):
        print("\nStage 2: Gene set refinement enabled (via sparse PCA)")

    # ── Stage 3: Multi-method scoring ──
    print("\nStage 3: Computing scoring methods...")

    print("  Computing EWM scores...")
    ewm_scores = compute_ewm_scores(
        expression, gene_module_map, evidence_weights, n_modules,
    )

    print("  Computing RES scores...")
    res_scores = compute_res_scores(
        expression, gene_module_map, n_modules,
    )

    print("  Computing SPCA scores...")
    spca_scores = compute_spca_scores(
        expression, gene_module_map, n_modules, refinement_config,
    )
    spca_loadings = spca_scores.attrs.get("loadings", {})

    # ── Orient SPCA components by EWM sign ──
    # SparsePCA has sign arbitrariness. Without orientation, 6 of 14 modules
    # have negative EWM-SPCA correlation → fusion subtracts signal.
    for k in range(n_modules):
        spca_col = spca_scores.columns[k]
        ewm_col = ewm_scores.columns[k]
        if np.corrcoef(spca_scores[spca_col], ewm_scores[ewm_col])[0, 1] < 0:
            spca_scores[spca_col] = -spca_scores[spca_col]
            # Also flip stored loadings for consistency
            if k in spca_loadings:
                for gene in spca_loadings[k]:
                    spca_loadings[k][gene] = -spca_loadings[k][gene]

    # ── Ensemble fusion ──
    ensemble_config = config.get("ensemble", {})
    gamma_ewm = ensemble_config.get("gamma_ewm", 0.50)
    gamma_res = ensemble_config.get("gamma_res", 0.25)
    gamma_spca = ensemble_config.get("gamma_spca", 0.25)

    module_names = [m["name"] for m in MODULE_DEFINITIONS[:n_modules]]

    print(f"\n  Fusing ensemble (γ_ewm={gamma_ewm}, γ_res={gamma_res}, γ_spca={gamma_spca})...")

    # Cell death enhancement: boost CELL_DEATH module weight
    cell_death_cfg = config.get("cell_death", {})
    cd_idx = cell_death_cfg.get("module_index", 12)
    cd_alpha = cell_death_cfg.get("enhancement_alpha", 0.25)

    indicators_raw = fuse_ensemble(
        ewm_scores, res_scores, spca_scores,
        gamma_ewm, gamma_res, gamma_spca, module_names,
    )

    # Apply cell death enhancement if configured
    if cd_alpha > 0 and cd_idx < n_modules:
        cd_col = module_names[cd_idx]
        if cd_col in indicators_raw.columns:
            indicators_raw[cd_col] *= (1.0 + cd_alpha)
            print(f"  CELL_DEATH module enhanced by factor {(1.0 + cd_alpha):.2f}")

    # ── Stage 4: Orthogonalization ──
    ortho_config = config.get("orthogonalization", {})
    ortho_method = ortho_config.get("method", "lowdin")
    ortho_threshold = ortho_config.get("partial_corr_threshold", 0.7)

    print(f"\nStage 4: Orthogonalization (method={ortho_method})...")
    indicators, S = orthogonalize(indicators_raw, ortho_method, ortho_threshold)

    if S is not None:
        # Report diagonal dominance of S (biological meaning preservation)
        diag = np.diag(S)
        print(f"  Transform matrix diagonal range: [{diag.min():.3f}, {diag.max():.3f}]")
        print(f"  Mean diagonal: {diag.mean():.3f} (>0.5 = good meaning preservation)")

    # ── Use provider's real pathway-level scores ──
    # Previously, _compute_pathway_scores produced a degenerate output (15 unique
    # columns from 140, all r=1.000 with EWM modules). The provider's
    # cell_pathway_scores_zscore.csv has genuine per-pathway z-scores computed
    # as simple means of available genes per pathway (≥3 expressed genes).
    pathway_scores = None
    if config["modules"].get("include_pathway_level", True):
        print("\n  Loading 140 pathway-level scores from provider file...")
        pathway_scores = data["pathway_scores"]

    # ── Derived indicators (pathway imbalance indices) ──
    print("\n  Computing derived pathway imbalance indicators...")
    derived_cfg = config.get("derived_indicators", {})
    derived_indicators = None
    if derived_cfg.get("enabled", True):
        from .derived import compute_derived_indicators
        derived_indicators = compute_derived_indicators(
            indicators_raw, expression, gene_module_map,
            module_names, config,
        )
        print(f"  Derived indicators: {list(derived_indicators.columns)}")

    # ── Lineage-conditioned indicators ──
    print("\n  Computing lineage-conditioned indicators...")
    lineage_cfg = config.get("lineage_conditioned", {})
    lineage_indicators = None
    if lineage_cfg.get("enabled", True):
        from .derived import compute_lineage_conditioned
        lineage_indicators = compute_lineage_conditioned(
            indicators, data["cell_meta"],
            min_samples_per_lineage=lineage_cfg.get("min_samples_per_lineage", 5),
        )
        print(f"  Lineage-conditioned indicators: {lineage_indicators.shape}")

    # ── Stage 5: Knockout response (sampled) ──
    print("\nStage 5: Computing knockout response samples...")
    knockout_config = config.get("knockout", {})
    ensemble_weights = (gamma_ewm, gamma_res, gamma_spca)

    # Sample 5 cell lines and 20 genes for demonstration
    sample_cells = list(expression.index[:5])
    sample_genes = list(expression.columns[:20])

    knockout_summary = compute_knockout_summary(
        expression, gene_module_map, evidence_weights,
        ensemble_weights, spca_loadings,
        sample_genes=sample_genes,
        sample_cells=sample_cells,
        n_modules=n_modules,
    )
    print(f"  Computed {len(knockout_summary)} knockout response samples")

    return {
        "indicators": indicators,
        "indicators_raw": indicators_raw,
        "ewm_scores": ewm_scores,
        "res_scores": res_scores,
        "spca_scores": spca_scores,
        "pathway_scores": pathway_scores,
        "derived_indicators": derived_indicators,
        "lineage_indicators": lineage_indicators,
        "knockout_summary": knockout_summary,
        "transform_matrix": S,
        "gene_module_map": gene_module_map,
        "evidence_weights": evidence_weights,
        "spca_loadings": spca_loadings,
    }


def _compute_pathway_scores(
    expression: pd.DataFrame,
    pathway_meta: pd.DataFrame,
    gene_module_map: dict[str, dict],
    evidence_weights: dict[str, float],
) -> pd.DataFrame:
    """
    Compute evidence-weighted scores for all 149 MitoCarta3.0 pathways.

    This is a fine-grained supplement to the 12-module indicators.
    Each pathway gets an evidence-weighted mean score.
    """
    gene_list = expression.columns.tolist()
    expr_array = expression.to_numpy(dtype=np.float64)

    # Build pathway gene sets from gene metadata
    # Each gene's pathway annotations link to leaf-level pathway names
    pathway_genes: dict[str, list[int]] = {}
    for pw_name in pathway_meta["pathway_name"]:
        pathway_genes[pw_name] = []

    # For each gene, find which pathways it belongs to
    for gene_idx, gene in enumerate(gene_list):
        if gene not in gene_module_map:
            continue
        gw = evidence_weights.get(gene, 1.0)
        info = gene_module_map[gene]
        # Map module assignments back to pathways (approximate via module pathways)
        for mod_idx in info["modules"]:
            for pw_name in MODULE_DEFINITIONS[mod_idx]["pathways"]:
                if pw_name in pathway_genes:
                    pathway_genes[pw_name].append((gene_idx, gw))

    # Compute weighted mean per pathway
    pw_names = pathway_meta["pathway_name"].tolist()
    scores = np.zeros((len(expression), len(pw_names)), dtype=np.float64)

    for pw_idx, pw_name in enumerate(pw_names):
        gene_weight_pairs = pathway_genes[pw_name]
        if not gene_weight_pairs:
            continue
        indices = np.array([p[0] for p in gene_weight_pairs], dtype=np.int64)
        weights = np.array([p[1] for p in gene_weight_pairs], dtype=np.float64)
        weight_sum = weights.sum()
        if weight_sum > 0:
            scores[:, pw_idx] = (expr_array[:, indices] @ weights) / weight_sum

    result = pd.DataFrame(scores, index=expression.index, columns=pw_names)
    return result


def export_outputs(results: dict[str, Any], config: dict[str, Any]) -> None:
    """Export all outputs to CSV files."""
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Main indicator matrix
    results["indicators"].to_csv(output_dir / "cell_line_indicators.csv")
    print(f"  Saved: cell_line_indicators.csv ({results['indicators'].shape})")

    # Raw (pre-orthogonalization) indicators
    results["indicators_raw"].to_csv(output_dir / "cell_line_indicators_raw.csv")
    print(f"  Saved: cell_line_indicators_raw.csv")

    # Individual method scores
    results["ewm_scores"].to_csv(output_dir / "ewm_scores.csv")
    results["res_scores"].to_csv(output_dir / "res_scores.csv")
    results["spca_scores"].to_csv(output_dir / "spca_scores.csv")
    print("  Saved: ewm_scores.csv, res_scores.csv, spca_scores.csv")

    # Pathway-level scores
    if results["pathway_scores"] is not None:
        results["pathway_scores"].to_csv(output_dir / "pathway_scores_149.csv")
        print(f"  Saved: pathway_scores_149.csv ({results['pathway_scores'].shape})")

    # Knockout response samples
    results["knockout_summary"].to_csv(
        output_dir / "knockout_response_sample.csv", index=False,
    )
    print(f"  Saved: knockout_response_sample.csv")

    # Derived indicators
    if results.get("derived_indicators") is not None:
        results["derived_indicators"].to_csv(output_dir / "derived_indicators.csv")
        print(f"  Saved: derived_indicators.csv ({results['derived_indicators'].shape})")

    # Lineage-conditioned indicators
    if results.get("lineage_indicators") is not None:
        results["lineage_indicators"].to_csv(output_dir / "cell_line_indicators_lineage.csv")
        print(f"  Saved: cell_line_indicators_lineage.csv ({results['lineage_indicators'].shape})")

    # Indicator definition mapping
    _export_indicator_definitions(output_dir, results)

    # Gene-to-module mapping summary
    _export_gene_module_summary(output_dir, results)

    # Transform matrix
    if results["transform_matrix"] is not None:
        np.savetxt(
            output_dir / "transform_matrix.csv",
            results["transform_matrix"],
            delimiter=",",
        )
        print("  Saved: transform_matrix.csv")


def _export_indicator_definitions(
    output_dir: Path, results: dict[str, Any],
) -> None:
    """Export indicator definition table."""
    rows = []
    for i, mod in enumerate(MODULE_DEFINITIONS[:results["indicators"].shape[1]]):
        rows.append({
            "indicator_index": i,
            "indicator_name": mod["name"],
            "description": mod["description"],
            "n_pathways": len(mod["pathways"]),
            "pathways": "|".join(mod["pathways"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "indicator_definition.csv", index=False)


def _export_gene_module_summary(
    output_dir: Path, results: dict[str, Any],
) -> None:
    """Export gene-to-module mapping and evidence weights."""
    gmm = results["gene_module_map"]
    ew = results["evidence_weights"]

    rows = []
    for gene, info in gmm.items():
        rows.append({
            "gene_symbol": gene,
            "modules": "|".join(str(m) for m in info["modules"]),
            "n_modules": info["n_modules"],
            "sub_mito_location": info["sub_mito"],
            "evidence_weight": round(ew.get(gene, 1.0), 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "gene_module_mapping.csv", index=False)
