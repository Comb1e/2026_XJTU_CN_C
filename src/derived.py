"""Derived pathway imbalance and compensation indicators.

These indicators capture biologically meaningful relationships between
primary modules that are lost during orthogonalization. They are computed
from raw (pre-orthogonalization) module scores and are NOT orthogonalized.

References:
  [1] Glover et al. (2024) — BCL-2 family, cell death thresholds
  [5] Granath-Panelo & Kajimura (2024) — mitochondrial heterogeneity
  [7] Tan et al. (2024) — mtDNA transcription-replication coupling
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Gene sets for BCL-2 family classification
# Pro-apoptotic (BH3-only and effectors)
PRO_APOPTOTIC_GENES = {
    "BAX", "BAK1", "BID", "BIM", "BBC3",  # BBC3 = PUMA
    "PMAIP1", "BAD", "BIK", "BMF", "HRK",  # PMAIP1 = NOXA
    "BNIP3", "BNIP3L",
}
# Anti-apoptotic
ANTI_APOPTOTIC_GENES = {
    "BCL2", "BCL2L1", "MCL1", "BCL2L2", "A1",  # BCL2L1 = BCL-XL
    "BCL2A1",
}

# mtDNA transcription-related pathway names
TRANSCRIPTION_PATHWAYS = {
    "Transcription", "Polycistronic_mtRNA_processing",
    "mtRNA_granules", "mt-tRNA_modifications",
    "mt-rRNA_modifications", "mt-mRNA_modifications",
    "mtRNA_stability_and_decay",
}

# mtDNA replication-related pathway names
REPLICATION_PATHWAYS = {
    "mtDNA_replication", "mtDNA_nucleoid",
    "mtDNA_repair", "mtDNA_stability_and_decay",
}

# Fusion vs Fission genes
FUSION_GENES = {"MFN1", "MFN2", "OPA1"}
FISSION_GENES = {"DNM1L", "FIS1", "MFF", "MIEF1", "MIEF2"}  # DNM1L = DRP1

# Mitophagy/autophagy genes
MITOPHAGY_GENES = {"PINK1", "PRKN", "BNIP3", "BNIP3L", "FUNDC1", "SQSTM1", "OPTN"}


def compute_derived_indicators(
    indicators_raw: pd.DataFrame,
    expression: pd.DataFrame,
    gene_module_map: dict,
    module_names: list[str],
    config: dict | None = None,
) -> pd.DataFrame:
    """
    Compute derived pathway imbalance and compensation indicators.

    Args:
        indicators_raw: N_cells × K_modules DataFrame (pre-orthogonalization)
        expression: N_cells × P_genes DataFrame (z-scored)
        gene_module_map: gene → module assignments
        module_names: list of module names (matching indicators_raw columns)
        config: optional configuration dict

    Returns:
        DataFrame of shape (N_cells, D) with derived indicators.
    """
    if config is None:
        config = {}
    derived_cfg = config.get("derived_indicators", {})
    if not derived_cfg.get("enabled", True):
        return pd.DataFrame(index=indicators_raw.index)

    cell_ids = indicators_raw.index.tolist()
    derived_cols = {}

    # Helper: find module index by name
    def _mod_idx(name: str) -> int | None:
        try:
            return module_names.index(name)
        except ValueError:
            return None

    def _mod_score(name: str) -> np.ndarray:
        idx = _mod_idx(name)
        if idx is not None and idx < indicators_raw.shape[1]:
            return indicators_raw.iloc[:, idx].to_numpy(dtype=np.float64)
        return np.zeros(len(cell_ids))

    # Helper: compute weighted mean expression for a gene set
    def _gene_set_score(gene_set: set) -> np.ndarray:
        genes_in_data = [g for g in gene_set if g in expression.columns]
        if not genes_in_data:
            return np.zeros(len(cell_ids))
        return expression[genes_in_data].mean(axis=1).to_numpy(dtype=np.float64)

    # 1. OXPHOS vs TCA balance (ETC capacity vs substrate supply)
    oxphos = (_mod_score("OXPHOS_CI") + _mod_score("OXPHOS_CII_CIII") +
              _mod_score("OXPHOS_CIV_CV")) / 3.0
    tca = _mod_score("TCA_PYRUVATE")
    derived_cols["OXPHOS_vs_TCA_balance"] = oxphos - tca

    # 2. FAO vs TCA preference (lipid vs carbohydrate oxidation)
    fao = _mod_score("FAO_LIPID")
    derived_cols["FAO_vs_TCA_preference"] = fao - tca

    # 3. Fusion vs Fission balance
    fusion_score = _gene_set_score(FUSION_GENES)
    fission_score = _gene_set_score(FISSION_GENES)
    derived_cols["Fusion_vs_Fission_balance"] = fusion_score - fission_score

    # 4. OXPHOS Assembly Stress (assembly factors vs subunits)
    # Uses pathway annotations from gene_metadata to classify genes
    af_genes = set()
    sub_genes = set()
    for gene, info in gene_module_map.items():
        if gene not in expression.columns:
            continue
        pw_raw = info.get("pathways_raw", "").lower()
        if not pw_raw:
            continue
        # Check membership in OXPHOS modules via pathway annotations
        is_oxphos = any(
            kw in pw_raw for kw in ["oxphos", "complex i", "complex ii", "complex iii",
                                     "complex iv", "complex v", "respirasome"]
        )
        if not is_oxphos:
            continue
        is_af = "assembly" in pw_raw
        is_sub = "subunit" in pw_raw and "assembly" not in pw_raw
        if is_af:
            af_genes.add(gene)
        if is_sub:
            sub_genes.add(gene)
    af_score = _gene_set_score(af_genes) if af_genes else np.zeros(len(cell_ids))
    sub_score = _gene_set_score(sub_genes) if sub_genes else np.zeros(len(cell_ids))
    derived_cols["OXPHOS_Assembly_Stress"] = af_score - sub_score

    # 5. mtDNA Transcription vs Replication coupling (POLRMT dual role)
    trans_genes = set()
    repl_genes = set()
    from .preprocess import MODULE_DEFINITIONS
    mtDNA_idx = _mod_idx("mtDNA_RNA")
    if mtDNA_idx is not None:
        mtDNA_mod = MODULE_DEFINITIONS[mtDNA_idx]
        for pw in mtDNA_mod["pathways"]:
            # Classify pathways
            pw_lower = (pw + " ").lower()
            if any(t in pw_lower for t in ["transcrip", "mrna", "rrna", "trna", "rna_proc", "rna_gr", "rna_stab"]):
                # These are transcription-related in mtDNA context
                pass  # We'll use the pathway annotations from gene_metadata
    # Simplified approach: use gene metadata pathways
    for gene, info in gene_module_map.items():
        if gene not in expression.columns:
            continue
        pw_raw = info.get("pathways_raw", "")
        if not pw_raw:
            continue
        pw_lower = pw_raw.lower()
        z = expression[gene].to_numpy(dtype=np.float64)
        in_trans = any(kw in pw_lower for kw in ["transcrip", "mrna", "rrna", "trna", "rna_proc", "rna_gran", "rna_stab"])
        in_repl = any(kw in pw_lower for kw in ["replicat", "nucleoid", "repair", "stability_and_decay"])
        if in_trans and not in_repl:
            trans_genes.add(gene)
        if in_repl and not in_trans:
            repl_genes.add(gene)

    trans_score = _gene_set_score(trans_genes) if trans_genes else np.zeros(len(cell_ids))
    repl_score = _gene_set_score(repl_genes) if repl_genes else np.zeros(len(cell_ids))
    eps = 1e-8
    raw_ratio = trans_score / (repl_score + eps)
    # Clip extreme ratios to [-10, 10] (very low replication scores cause instability)
    derived_cols["mtDNA_TCR_ratio"] = np.clip(raw_ratio, -10.0, 10.0)
    derived_cols["mtDNA_TCR_balance"] = trans_score - repl_score

    # 6. BCL-2 pro-survival vs pro-death balance
    if derived_cfg.get("include_bcl2_balance", True):
        pro_death = _gene_set_score(PRO_APOPTOTIC_GENES)
        pro_survival = _gene_set_score(ANTI_APOPTOTIC_GENES)
        derived_cols["BCL2_prosurvival_vs_prodeath"] = pro_survival - pro_death
        derived_cols["BCL2_prosurvival"] = pro_survival
        derived_cols["BCL2_prodeath"] = pro_death

    # 7. Mitophagy vs Biogenesis balance
    mitophagy_score = _gene_set_score(MITOPHAGY_GENES)
    ribosome_score = _mod_score("MITO_RIBOSOME")
    derived_cols["Mitophagy_vs_Biogenesis"] = mitophagy_score - ribosome_score

    result = pd.DataFrame(derived_cols, index=cell_ids)
    return result


def compute_lineage_conditioned(
    indicators: pd.DataFrame,
    cell_meta: pd.DataFrame,
    min_samples_per_lineage: int = 5,
) -> pd.DataFrame:
    """
    Compute lineage-conditioned z-scores for each indicator.

    For each lineage L:
        M̃_k^(L)(c) = (M_k(c) - μ_k^(L)) / (σ_k^(L) + ε)

    Lineages with fewer than min_samples use global statistics.

    Reference: Granath-Panelo & Kajimura (2024) — tissue-specific
    mitochondrial specialization.

    Args:
        indicators: N_cells × K_modules DataFrame (global indicators)
        cell_meta: cell line metadata with OncotreeLineage column
        min_samples_per_lineage: minimum cells per lineage for local stats

    Returns:
        DataFrame of same shape with lineage-conditioned z-scores.
    """
    # Build lineage mapping
    lineage_map = cell_meta.set_index("cell_line_id")["OncotreeLineage"]
    common_cells = [c for c in indicators.index if c in lineage_map.index]
    lineages = lineage_map.loc[common_cells]

    # Global stats
    global_mean = indicators.loc[common_cells].mean()
    global_std = indicators.loc[common_cells].std()

    # Per-lineage stats for large lineages
    lineage_counts = lineages.value_counts()
    valid_lineages = lineage_counts[lineage_counts >= min_samples_per_lineage].index

    lineage_means = {}
    lineage_stds = {}
    for lin in valid_lineages:
        lin_cells = lineages[lineages == lin].index
        lin_data = indicators.loc[lin_cells]
        lineage_means[lin] = lin_data.mean()
        lineage_stds[lin] = lin_data.std()

    # Compute conditioned scores
    result = indicators.copy()
    for cell in common_cells:
        lin = lineages.loc[cell]
        if lin in lineage_means:
            mu = lineage_means[lin]
            sigma = lineage_stds[lin]
        else:
            mu = global_mean
            sigma = global_std
        result.loc[cell] = (indicators.loc[cell] - mu) / (sigma + 1e-12)

    return result
