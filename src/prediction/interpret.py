"""Model interpretability and explainability outputs for Problem 2.

All outputs are derived from interpretable formula-based models:
  - Gene essentiality formula coefficients
  - Cell vulnerability formula coefficients
  - Module×Indicator interaction coefficients
  - Expression→Dependency effect curve
  - Human-readable formula printout
  - Gene baseline decomposition via teacher model
  - Context-specificity analysis (lineage × module)
  - Cell-level module dependency profiles
  - Case studies
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import build_all_features
from .baselines import (
    shrink_gene_means, shrink_cell_means,
    build_collaborative_features,
    train_gene_baseline_teacher,
)
from .formula import (
    GeneEssentialityFormula, CellVulnerabilityFormula,
    ShrinkageGeneFormula, ShrinkageCellFormula,
    ModuleInteractionFormula, ExpressionEffectFormula,
    ModuleMatchFormula, EvidenceWeightedFormula,
    InteractionBlend,
)


def generate_all_interpretations(
    pred_result: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Generate all interpretability outputs from formula-based models.

    Args:
        pred_result: output from run_prediction().
        config: configuration dict.

    Returns:
        dict with paths to all generated files.
    """
    output_dir = Path(config["paths"]["output_dir"]) / "prediction"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = pred_result["models"]

    # Reload data for analysis
    print("  Loading data for interpretation...")
    data_dir = Path(config["paths"]["data_dir"])
    labels = pd.read_csv(data_dir / "labels" / "gene_dependency.csv")

    # Build features
    print("  Building features...")
    X, meta = build_all_features(
        labels[["cell_line_id", "perturbation_gene"]], config,
    )

    results = {}

    # 1. Complete formula printout
    print("  1. Formula printout...")
    results["formula_printout"] = _formula_printout(models, output_dir)

    # 2. Gene essentiality feature importance (from shrinkage model)
    shrink_gene = models.get("shrink_gene")
    if shrink_gene is not None and shrink_gene.gene_formula_ is not None:
        print("  2. Gene essentiality coefficients (prior Φ(g))...")
        results["gene_formula"] = _gene_formula_importance(shrink_gene.gene_formula_, output_dir)
        # Also save shrinkage stats
        _shrinkage_stats(shrink_gene, output_dir)
    else:
        gene_formula = models.get("gene_formula")
        if gene_formula is not None:
            print("  2. Gene essentiality coefficients...")
            results["gene_formula"] = _gene_formula_importance(gene_formula, output_dir)

    # 3. Cell vulnerability feature importance (from shrinkage model)
    shrink_cell = models.get("shrink_cell")
    if shrink_cell is not None and shrink_cell.cell_formula_ is not None:
        print("  3. Cell vulnerability coefficients (prior Ψ(c))...")
        results["cell_formula"] = _cell_formula_importance(shrink_cell.cell_formula_, output_dir)
    else:
        cell_formula = models.get("cell_formula")
        if cell_formula is not None:
            print("  3. Cell vulnerability coefficients...")
            results["cell_formula"] = _cell_formula_importance(cell_formula, output_dir)

    # 4. Module×Indicator interaction coefficients
    i_mod = models.get("i_mod_formula")
    if i_mod is not None:
        print("  4. Module×Indicator interaction coefficients...")
        results["module_interaction"] = _module_interaction_coefficients(i_mod, output_dir)

    # 5. Expression→Dependency effect curve
    i_expr = models.get("i_expr_formula")
    if i_expr is not None:
        print("  5. Expression effect curve...")
        results["expr_effect"] = _expression_effect_curve(i_expr, output_dir)

    # 6. Interaction blend weights
    blend = models.get("interaction_blend")
    if blend is not None:
        print("  6. Interaction blend weights...")
        results["blend_weights"] = _blend_weights(blend, output_dir)

    # 7. Gene baseline decomposition
    print("  7. Gene baseline decomposition...")
    results["gene_baseline"] = _gene_baseline_decomposition(meta, labels, config, output_dir)

    # 8. Context-specificity analysis
    print("  8. Context-specificity analysis...")
    cell_meta = pd.read_csv(data_dir / "metadata" / "cell_line_metadata.csv")
    results["context"] = _context_specificity(pred_result, labels, cell_meta,
                                               meta, output_dir)

    # 9. Cell module dependency profiles
    print("  9. Cell module dependency profiles...")
    results["cell_module"] = _cell_module_profiles(pred_result, labels, meta, output_dir)

    # 10. Case studies
    print("  10. Case studies...")
    results["case_studies"] = _case_studies(pred_result, labels, meta, output_dir)

    return results


# ── Formula model interpretability ──────────────────────────────────────────

def _formula_printout(models: dict, output_dir: Path) -> str:
    """Save complete human-readable formula to file."""
    lines = []
    lines.append("=" * 70)
    lines.append("GENE DEPENDENCY PREDICTION FORMULA (Empirical Bayes)")
    lines.append("=" * 70)
    lines.append("")

    shrink_gene = models.get("shrink_gene")
    shrink_cell = models.get("shrink_cell")

    if shrink_gene is not None and shrink_cell is not None:
        lines.append("ŷ(c,g) = [w_g·x̄_g + (1-w_g)·Φ(g)] + [v_c·r̄_c + (1-v_c)·Ψ(c)]")
        lines.append("       + Blend[I_mod, I_expr, I_match, I_ew]")
        lines.append("")
        lines.append(f"SHRINKAGE: λ_gene={shrink_gene.lambda_:.1f}, λ_cell={shrink_cell.lambda_:.1f}")
        lines.append(f"  w_g = n_g/(n_g+λ_gene) ∈ [0, 1]  (0=cold, 1=warm)")
        lines.append(f"  v_c = m_c/(m_c+λ_cell) ∈ [0, 1]")
        lines.append("")

        gf = shrink_gene.gene_formula_
        if gf is not None:
            lines.append("─" * 70)
            lines.append("GENE PRIOR Φ(g):")
            lines.append(gf.formula_str(top_n=10))
            lines.append("")

        cf = shrink_cell.cell_formula_
        if cf is not None:
            lines.append("─" * 70)
            lines.append("CELL PRIOR Ψ(c):")
            lines.append(cf.formula_str(top_n=10))
            lines.append("")
    else:
        lines.append("ŷ(c,g) = μ̂_g + β̂_c + Blend[I_mod, I_expr, I_match, I_ew]")
        lines.append("")

        gene_formula = models.get("gene_formula")
        if gene_formula is not None:
            lines.append("─" * 70)
            lines.append("GENE ESSENTIALITY μ̂_g:")
            lines.append(gene_formula.formula_str(top_n=10))
            lines.append("")

        cell_formula = models.get("cell_formula")
        if cell_formula is not None:
            lines.append("─" * 70)
            lines.append("CELL VULNERABILITY β̂_c:")
            lines.append(cell_formula.formula_str(top_n=10))
            lines.append("")

    i_mod = models.get("i_mod_formula")
    if i_mod is not None:
        lines.append("─" * 70)
        lines.append("MODULE×INDICATOR INTERACTION I_mod:")
        lines.append(i_mod.formula_str())
        lines.append("")

    i_expr = models.get("i_expr_formula")
    if i_expr is not None:
        lines.append("─" * 70)
        lines.append("EXPRESSION EFFECT I_expr:")
        lines.append(i_expr.formula_str())
        lines.append("")

    i_match = models.get("i_match_formula")
    if i_match is not None:
        lines.append("─" * 70)
        lines.append("MODULE-MATCH I_match:")
        lines.append(i_match.formula_str())
        lines.append("")

    i_ew = models.get("i_ew_formula")
    if i_ew is not None:
        lines.append("─" * 70)
        lines.append("EVIDENCE-WEIGHTED I_ew:")
        lines.append(i_ew.formula_str())
        lines.append("")

    blend = models.get("interaction_blend")
    if blend is not None:
        lines.append("─" * 70)
        lines.append("INTERACTION BLEND:")
        for name, w in blend.get_weights():
            lines.append(f"  α_{name} = {w:+.4f}")
        lines.append(f"  intercept = {blend.intercept_:.4f}")
        lines.append("=" * 70)

    path = output_dir / "prediction_formula.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"    Saved formula to {path}")
    return str(path)


def _gene_formula_importance(
    gene_formula: GeneEssentialityFormula,
    output_dir: Path,
) -> str:
    """Export gene essentiality formula coefficients."""
    top = gene_formula.get_top_features(top_n=50)
    df = pd.DataFrame(top, columns=["feature", "coefficient"])
    df["abs_coefficient"] = np.abs(df["coefficient"])
    df = df.sort_values("abs_coefficient", ascending=False)
    path = output_dir / "gene_essentiality_coefficients.csv"
    df.to_csv(path, index=False)
    for _, row in df.head(10).iterrows():
        print(f"    {row['feature']}: {row['coefficient']:.4f}")
    return str(path)


def _cell_formula_importance(
    cell_formula: CellVulnerabilityFormula,
    output_dir: Path,
) -> str:
    """Export cell vulnerability formula coefficients."""
    top = cell_formula.get_top_features(top_n=50)
    df = pd.DataFrame(top, columns=["feature", "coefficient"])
    df["abs_coefficient"] = np.abs(df["coefficient"])
    df = df.sort_values("abs_coefficient", ascending=False)
    path = output_dir / "cell_vulnerability_coefficients.csv"
    df.to_csv(path, index=False)
    for _, row in df.head(10).iterrows():
        print(f"    {row['feature']}: {row['coefficient']:.4f}")
    return str(path)


def _module_interaction_coefficients(
    i_mod: ModuleInteractionFormula,
    output_dir: Path,
) -> str:
    """Export module×indicator interaction coefficients (14 named values)."""
    coefs = i_mod.get_coefficients()
    df = pd.DataFrame(coefs, columns=["module", "coefficient"])
    df["abs_coefficient"] = np.abs(df["coefficient"])
    df = df.sort_values("abs_coefficient", ascending=False)
    path = output_dir / "module_interaction_coefficients.csv"
    df.to_csv(path, index=False)
    for _, row in df.iterrows():
        if abs(row["coefficient"]) > 1e-6:
            print(f"    {row['module']}: {row['coefficient']:+.4f}")
    return str(path)


def _expression_effect_curve(
    i_expr: ExpressionEffectFormula,
    output_dir: Path,
) -> str:
    """Export expression→dependency effect curve as a table."""
    z_vals = np.linspace(-4, 4, 100)
    preds = i_expr.predict(z_vals)
    df = pd.DataFrame({"z_score": z_vals, "effect": preds})
    path = output_dir / "expression_effect_curve.csv"
    df.to_csv(path, index=False)
    print(f"    θ = [{', '.join(f'{c:.4f}' for c in i_expr.coefficients_)}]")
    return str(path)


def _blend_weights(blend: InteractionBlend, output_dir: Path) -> str:
    """Export interaction blend weights."""
    weights = blend.get_weights()
    df = pd.DataFrame(weights, columns=["component", "weight"])
    df["abs_weight"] = np.abs(df["weight"])
    df = df.sort_values("abs_weight", ascending=False)
    path = output_dir / "interaction_blend_weights.csv"
    df.to_csv(path, index=False)
    for _, row in df.iterrows():
        print(f"    {row['component']}: {row['weight']:+.4f}")
    return str(path)


def _shrinkage_stats(
    shrink_gene: ShrinkageGeneFormula,
    output_dir: Path,
) -> str:
    """Export empirical Bayes shrinkage statistics."""
    weights = shrink_gene.weights
    n_genes = {g: shrink_gene.n_g_.get(g, 0) for g in shrink_gene.mu_g_}

    rows = []
    for g in sorted(shrink_gene.mu_g_.keys()):
        rows.append({
            "gene": g,
            "n_cells": n_genes.get(g, 0),
            "evidence_weight": weights.get(g, 0.0),
            "observed_mean": shrink_gene.x_bar_g_.get(g, 0.0),
            "prior_prediction": shrink_gene.phi_g_.get(g, 0.0),
            "shrunk_estimate": shrink_gene.mu_g_.get(g, 0.0),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("evidence_weight", ascending=False)
    path = output_dir / "gene_shrinkage_stats.csv"
    df.to_csv(path, index=False)

    # Summary
    n_cold = int((df["n_cells"] == 0).sum())
    n_warm = int((df["n_cells"] > 0).sum())
    print(f"    λ={shrink_gene.lambda_:.1f}, {n_warm} warm + {n_cold} cold genes")
    print(f"    Evidence weights: "
          f"warm ∈ [{df[df['n_cells']>0]['evidence_weight'].min():.3f}, "
          f"{df[df['n_cells']>0]['evidence_weight'].max():.3f}], "
          f"cold = 0.0")
    return str(path)


# ── Gene baseline decomposition ──────────────────────────────────────────────

def _gene_baseline_decomposition(
    meta: dict,
    labels: pd.DataFrame,
    config: dict,
    output_dir: Path,
) -> str:
    """Decompose gene baseline predictions into feature contributions."""
    gene_meta = meta.get("gene_meta")
    if gene_meta is None:
        gene_meta = pd.read_csv(
            Path(config["paths"]["metadata_dir"]) / "gene_metadata.csv"
        )

    from .features import build_gene_static_features, build_gene_expression_profile_features

    g1 = build_gene_static_features(
        gene_meta, meta["gene_module_map"], meta["evidence_weights"],
    )
    g2 = build_gene_expression_profile_features(meta["expression"])
    gene_feats = g1.join(g2, how="inner")

    teacher, oof_preds, _ = train_gene_baseline_teacher(g1, g2, labels)

    train_genes = set(labels["perturbation_gene"].unique())
    all_genes = set(gene_feats.index)
    cold_genes = sorted(all_genes - train_genes)

    rows = []
    for gene in cold_genes[:50]:
        if gene not in gene_feats.index:
            continue
        x = gene_feats.loc[gene].to_numpy(dtype=np.float32).reshape(1, -1)
        x = np.nan_to_num(x, nan=0.0)
        pred = teacher.predict(x)[0]
        modules = meta["gene_module_map"].get(gene, {}).get("modules", [])
        rows.append({
            "gene": gene,
            "predicted_baseline": float(pred),
            "n_modules": len(modules),
        })

    df = pd.DataFrame(rows)
    path = output_dir / "gene_baseline_decomposition.csv"
    df.to_csv(path, index=False)
    return str(path)


# ── Context-specificity analysis ────────────────────────────────────────────

def _context_specificity(
    pred_result: dict,
    labels: pd.DataFrame,
    cell_meta: pd.DataFrame,
    meta: dict,
    output_dir: Path,
) -> str:
    """Analyze lineage-specific dependency patterns."""
    submission = pred_result["submission"]
    lineage_map = cell_meta.set_index("cell_line_id")["OncotreeLineage"]

    df = submission.merge(
        labels[["cell_line_id", "perturbation_gene", "label"]].rename(
            columns={"label": "true_label"}),
        on=["cell_line_id", "perturbation_gene"], how="left",
    )
    df["lineage"] = df["cell_line_id"].map(lineage_map)

    gmm = meta.get("gene_module_map", {})
    n_modules = 14

    rows = []
    for lineage in sorted(df["lineage"].dropna().unique()):
        lin_df = df[df["lineage"] == lineage]
        for mod in range(n_modules):
            mod_genes = {g for g, info in gmm.items()
                        if mod in info.get("modules", [])}
            mod_mask = lin_df["perturbation_gene"].isin(mod_genes)
            if mod_mask.sum() > 0:
                mean_pred = lin_df.loc[mod_mask, "label"].mean()
                rows.append({
                    "lineage": lineage,
                    "module": mod,
                    "module_name": _module_name(mod),
                    "mean_dependency": mean_pred,
                    "n_pairs": int(mod_mask.sum()),
                })

    df_out = pd.DataFrame(rows)
    path = output_dir / "lineage_module_dependency.csv"
    df_out.to_csv(path, index=False)

    if len(df_out) > 0:
        csi = df_out.groupby("module")["mean_dependency"].agg(["std", "mean"])
        csi["csi"] = csi["std"] / (csi["mean"].abs() + 1e-8)
        csi_path = output_dir / "context_specificity_index.csv"
        csi.to_csv(csi_path)

    return str(path)


# ── Cell module profiles ────────────────────────────────────────────────────

def _cell_module_profiles(
    pred_result: dict,
    labels: pd.DataFrame,
    meta: dict,
    output_dir: Path,
) -> str:
    """Compute per-cell module dependency profiles."""
    submission = pred_result["submission"]
    gmm = meta.get("gene_module_map", {})
    n_modules = 14

    rows = []
    for cell in submission["cell_line_id"].unique()[:100]:
        cell_df = submission[submission["cell_line_id"] == cell]
        for mod in range(n_modules):
            mod_genes = {g for g, info in gmm.items()
                        if mod in info.get("modules", [])}
            mod_mask = cell_df["perturbation_gene"].isin(mod_genes)
            if mod_mask.sum() > 0:
                mean_pred = cell_df.loc[mod_mask, "label"].mean()
                rows.append({
                    "cell_line_id": cell,
                    "module": mod,
                    "mean_predicted_dependency": mean_pred,
                })

    df = pd.DataFrame(rows)
    path = output_dir / "cell_module_dependency.csv"
    df.to_csv(path, index=False)
    return str(path)


# ── Case studies ────────────────────────────────────────────────────────────

def _case_studies(
    pred_result: dict,
    labels: pd.DataFrame,
    meta: dict,
    output_dir: Path,
) -> str:
    """Generate case studies for representative cell lines."""
    submission = pred_result["submission"]
    example_cells = submission["cell_line_id"].unique()[:3]

    rows = []
    for cell in example_cells:
        cell_df = submission[submission["cell_line_id"] == cell].copy()
        cell_df = cell_df.sort_values("label", ascending=False)
        top15 = cell_df.head(15)
        for i, (_, row) in enumerate(top15.iterrows()):
            gene = row["perturbation_gene"]
            modules = meta.get("gene_module_map", {}).get(gene, {}).get("modules", [])
            rows.append({
                "cell_line_id": cell,
                "rank": i + 1,
                "perturbation_gene": gene,
                "predicted_dependency": row["label"],
                "gene_modules": "|".join(str(m) for m in modules),
                "n_modules": len(modules),
            })

    df = pd.DataFrame(rows)
    path = output_dir / "case_studies.csv"
    df.to_csv(path, index=False)
    return str(path)


def _module_name(mod_idx: int) -> str:
    """Get module name from index."""
    names = [
        "OXPHOS_CI", "OXPHOS_CII_CIII", "OXPHOS_CIV_CV", "TCA_PYRUVATE",
        "FAO_LIPID", "AA_COFACTOR", "MITO_RIBOSOME", "mtDNA_RNA",
        "PROTEIN_IMPORT", "TRANSPORT", "REDOX_DETOX", "MITO_DYNAMICS",
        "CELL_DEATH", "SIGNALING",
    ]
    return names[mod_idx] if mod_idx < len(names) else f"Module_{mod_idx}"
