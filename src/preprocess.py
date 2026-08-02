"""Data loading, validation, and gene-to-module mapping (Stages 1 & 2).

Key outputs:
- expression_df: N_cells x P_genes matrix (z-scored)
- gene_module_map: dict mapping gene_symbol -> list of module indices
- module_definitions: dict mapping module_idx -> {name, description, gene_set}
- evidence_weights: dict mapping gene_symbol -> composite weight (for EWM)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any


# ── Module definitions ──────────────────────────────────────────────
# Maps MitoCarta3.0 pathway names (from pathway_metadata) to 12 modules.
# Each module has a name, biological description, and list of MitoPathway keys.

MODULE_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "OXPHOS_CI",
        "description": "Respiratory chain Complex I expression state",
        "pathways": [
            "CI_subunits", "CI_assembly_factors",
        ],
    },
    {
        "name": "OXPHOS_CII_CIII",
        "description": "Respiratory chain Complex II & III expression",
        "pathways": [
            "CII_subunits", "CII_assembly_factors",
            "CIII_subunits", "CIII_assembly_factors",
            "Respirasome_assembly",
        ],
    },
    {
        "name": "OXPHOS_CIV_CV",
        "description": "Respiratory chain Complex IV & V expression",
        "pathways": [
            "CIV_subunits", "CIV_assembly_factors",
            "CV_subunits", "CV_assembly_factors",
        ],
    },
    {
        "name": "TCA_PYRUVATE",
        "description": "TCA cycle and central carbon metabolism",
        "pathways": [
            "TCA_cycle", "TCA-associated",
            "Pyruvate_metabolism", "Gluconeogenesis",
            "Malate-aspartate_shuttle",
        ],
    },
    {
        "name": "FAO_LIPID",
        "description": "Fatty acid oxidation and lipid metabolism",
        "pathways": [
            "Fatty_acid_oxidation", "Type_II_fatty_acid_synthesis",
            "Cardiolipin_synthesis", "Phospholipid_metabolism",
            "Cholesterol_bile_acid_steroid_synthesis",
            "Carnitine_synthesis_and_transport", "Carnitine_shuttle",
            "Lipoate_insertion", "Ketone_metabolism",
        ],
    },
    {
        "name": "AA_COFACTOR",
        "description": "Amino acid metabolism and cofactor biosynthesis",
        "pathways": [
            "Branched-chain_amino_acid_metabolism",
            "Branched-chain_amino_acid_dehydrogenase_complex",
            "Lysine_metabolism", "Serine_metabolism",
            "Glycine_metabolism", "Glycine_cleavage_system",
            "Glutamate_metabolism", "Proline_metabolism",
            "Glyoxylate_metabolism", "GABA_metabolism",
            "Catechol_metabolism", "Kynurenine_metabolism",
            "Urea_cycle",
            "Coenzyme_Q_metabolism", "Coenzyme_A_metabolism",
            "Heme_synthesis_and_processing",
            "Fe-S_cluster_biosynthesis", "Fe-S-containing_proteins",
            "Heme-containing_proteins",
            "NAD_biosynthesis_and_metabolism",
            "Iron_homeostasis", "Copper_metabolism",
            "Molybdenum_cofactor_synthesis_and_proteins",
            "Tetrahydrobiopterin_synthesis",
            "Vitamin_A_metabolism", "Vitamin_B2_metabolism",
            "Vitamin_B12_metabolism", "Vitamin_D_metabolism",
            "Folate_and_1-C_metabolism", "Biotin_utilizing_proteins",
            "Choline_and_betaine_metabolism",
        ],
    },
    {
        "name": "MITO_RIBOSOME",
        "description": "Mitochondrial translation machinery",
        "pathways": [
            "Mitochondrial_ribosome", "Mitochondrial_ribosome_assembly",
            "Translation_factors", "mt-tRNA_synthetases",
            "fMet_processing",
        ],
    },
    {
        "name": "mtDNA_RNA",
        "description": "mtDNA maintenance and RNA metabolism",
        "pathways": [
            "mtDNA_replication", "mtDNA_nucleoid",
            "mtDNA_repair", "mtDNA_stability_and_decay",
            "Transcription",
            "mtRNA_granules", "Polycistronic_mtRNA_processing",
            "mt-tRNA_modifications", "mt-rRNA_modifications",
            "mt-mRNA_modifications", "mtRNA_stability_and_decay",
        ],
    },
    {
        "name": "PROTEIN_IMPORT",
        "description": "Protein import, processing and quality control",
        "pathways": [
            "TOM", "SAM", "MIA40",
            "TIM22_carrier_pathway", "TIM23_presequence_pathway",
            "Import_motor", "Preprotein_cleavage",
            "Proteases", "Chaperones",
        ],
    },
    {
        "name": "TRANSPORT",
        "description": "Small molecule and ion transport",
        "pathways": [
            "SLC25A_family", "ABC_transporters",
            "Sideroflexins", "Calcium_uniporter",
            "Nucleotide_import",
        ],
    },
    {
        "name": "REDOX_DETOX",
        "description": "Redox balance and detoxification",
        "pathways": [
            "ROS_and_glutathione_metabolism",
            "Xenobiotic_metabolism",
            "Selenoproteins", "Amidoxime_reducing_complex",
            "Cytochromes", "Q-linked_reactions_other",
            "Sulfur_metabolism",
        ],
    },
    {
        "name": "MITO_DYNAMICS",
        "description": "Mitochondrial dynamics, membrane structure and trafficking",
        "pathways": [
            "Fusion", "Fission",
            "Organelle_contact_sites", "Intramitochondrial_membrane_interactions",
            "Trafficking",
            "Cristae_formation", "MICOS_complex",
        ],
    },
    {
        "name": "CELL_DEATH",
        "description": "Cell death regulation (apoptosis, mitophagy, MPTP) — directly linked to CRISPR fitness phenotypes",
        "pathways": [
            "Apoptosis", "Mitophagy", "Autophagy",
            "Mitochondrial_permeability_transition_pore",
        ],
    },
    {
        "name": "SIGNALING",
        "description": "Mitochondrial signaling (calcium, immune, cAMP-PKA)",
        "pathways": [
            "Calcium_homeostasis", "Calcium_cycle",
            "EF_hand_proteins",
            "Immune_response", "cAMP-PKA_signaling",
        ],
    },
]

# Sub-mitochondrial location -> module fallback mapping
# Used for genes without pathway annotations (~100 genes)
SUBMITO_FALLBACK: dict[str, int] = {
    # Matrix: mostly metabolism and central dogma
    "Matrix": 5,       # AA_COFACTOR (largest metabolic module)
    # MIM: mostly OXPHOS and transport
    "MIM": 2,          # OXPHOS_CIV_CV
    # MOM: dynamics, cell death, and signaling proteins
    "MOM": 12,         # CELL_DEATH (BCL-2 family and apoptosis regulators)
    # IMS: redox and protein import
    "IMS": 10,         # REDOX_DETOX
    # Membrane (general): transport
    "Membrane": 9,     # TRANSPORT
}


def _build_pathway_to_module_map() -> dict[str, int]:
    """Build a mapping from leaf pathway name -> module index."""
    pathway_map: dict[str, int] = {}
    for mod_idx, mod in enumerate(MODULE_DEFINITIONS):
        for pw_name in mod["pathways"]:
            pathway_map[pw_name] = mod_idx
    return pathway_map


def _leaf_to_pw_key(leaf_name: str) -> str:
    """Convert a human-readable leaf pathway name to pathway_metadata key format.

    Examples:
        'CI subunits' -> 'CI_subunits'
        'TCA cycle' -> 'TCA_cycle'
        'Cholesterol, bile acid, steroid synthesis' -> 'Cholesterol_bile_acid_steroid_synthesis'
        'Q-linked reactions, other' -> 'Q-linked_reactions_other'
        'Fe-S-containing proteins' -> 'Fe-S-containing_proteins'
    """
    # Remove commas, replace spaces with underscores
    key = leaf_name.replace(",", "").replace(" ", "_")
    return key


def build_gene_module_map(
    gene_metadata: pd.DataFrame,
    pathway_metadata: pd.DataFrame | None = None,
) -> dict[str, dict]:
    """
    Build gene-to-module mapping from MitoCarta3.0 pathway annotations.

    For each gene, determines which of the 12 modules it belongs to by:
    1. Parsing its hierarchical pathway string from gene_metadata
    2. Extracting leaf pathway names, converting to pathway_metadata key format
    3. Matching against module definitions
    4. Falling back to sub_mito_location for genes with no pathway match

    Returns:
        dict keyed by gene_symbol with:
            - 'modules': list of module indices (0-11)
            - 'sub_mito': sub-mitochondrial location string
            - 'pathways_raw': original pathway string
            - 'n_modules': number of modules assigned
    """
    pw_to_mod = _build_pathway_to_module_map()

    # Build set of valid pathway_metadata keys for validation
    pw_meta_keys: set[str] = set()
    if pathway_metadata is not None:
        pw_meta_keys = set(pathway_metadata["pathway_name"])

    gene_map: dict[str, dict] = {}
    genes_without_module = 0
    total_assignments = 0

    for _, row in gene_metadata.iterrows():
        gene = row["gene_symbol"]
        pathways_raw = str(row.get("pathways", ""))
        sub_mito = str(row.get("sub_mito_location", "unknown"))

        assigned_modules: set[int] = set()

        if pathways_raw and pathways_raw != "nan":
            # Each gene may be annotated to multiple pathways separated by '|'
            for pathway_entry in pathways_raw.split("|"):
                pathway_entry = pathway_entry.strip()
                if not pathway_entry:
                    continue

                # Extract leaf pathway name (last segment after '>')
                parts = [p.strip() for p in pathway_entry.split(">")]
                leaf = parts[-1]

                # Skip artifact entries like "0" (from "MitoCarta3.0")
                if leaf == "0" or leaf.isdigit():
                    continue

                # Convert to pathway_metadata key format
                pw_key = _leaf_to_pw_key(leaf)

                # Match against module definitions
                if pw_key in pw_to_mod:
                    assigned_modules.add(pw_to_mod[pw_key])
                else:
                    # Try matching mid-level pathway names too
                    for part in parts[1:-1]:  # skip root and leaf
                        mid_key = _leaf_to_pw_key(part)
                        if mid_key in pw_to_mod:
                            assigned_modules.add(pw_to_mod[mid_key])

        # Fallback: use sub-mitochondrial location for unassigned genes
        if not assigned_modules:
            genes_without_module += 1
            if sub_mito in SUBMITO_FALLBACK:
                assigned_modules.add(SUBMITO_FALLBACK[sub_mito])
            elif "|" in sub_mito:
                parts = sub_mito.split("|")
                for p in parts:
                    p = p.strip()
                    if p in SUBMITO_FALLBACK:
                        assigned_modules.add(SUBMITO_FALLBACK[p])
                        break
                else:
                    assigned_modules.add(5)  # Default: AA_COFACTOR
            else:
                assigned_modules.add(5)  # Default: AA_COFACTOR

        gene_map[gene] = {
            "modules": sorted(assigned_modules),
            "sub_mito": sub_mito,
            "pathways_raw": pathways_raw if pathways_raw != "nan" else "",
            "n_modules": len(assigned_modules),
        }
        total_assignments += len(assigned_modules)

    return gene_map


def compute_evidence_weights(
    gene_metadata: pd.DataFrame,
    config: dict | None = None,
) -> dict[str, float]:
    """
    Compute composite evidence weight for each gene.

    Uses MitoCarta3.0 evidence scores to weight genes by mitochondrial
    localization confidence. Higher weight = stronger evidence.

    Reference: Rath et al. (2021) MitoCarta3.0 — the same features used
    in the Bayesian integration to determine mitochondrial membership.

    Returns:
        dict mapping gene_symbol -> weight in [0.8, 1.2] range
    """
    if config is None:
        config = {}
    ew_config = config.get("evidence_weights", {})
    if not ew_config.get("enabled", True):
        return {g: 1.0 for g in gene_metadata["gene_symbol"]}

    score_config = ew_config.get("scores", {})
    weight_range = ew_config.get("range", [0.8, 1.2])
    w_min, w_max = weight_range

    numeric_scores = {
        "targetp": "targetp_score",
        "coexpression_gnf_n50": "coexpression_gnf_n50_score",
    }
    categorical_scores = {
        "yeast_homolog": "yeast_mito_homolog_score",
        "rickettsia_homolog": "rickettsia_homolog_score",
        "mito_domain": "mito_domain_score",
        "msms": "msms_score",
    }

    # Yeast homolog score weights
    yeast_weight_map = {
        "OrthologMitoHighConf": 1.20,
        "OrthologMitoLowConf": 1.10,
        "HomologMitoHighConf": 1.15,
        "HomologMitoLowConf": 1.05,
        "Homolog": 1.05,
        "Ortholog": 1.08,
        "NoHomolog": 1.00,
    }

    # Rickettsia homolog score weights
    rick_weight_map = {
        "Ortholog": 1.15,
        "Homolog": 1.08,
        "NoHomolog": 1.00,
    }

    # Mito domain score weights
    mito_domain_weight_map = {
        "MitoDomain": 1.20,
        "SharedDomain": 1.05,
    }

    # MS/MS score weights (purity tiers from MitoCarta)
    msms_weight_map = {
        "75-100pure": 1.20,
        "50-75pure": 1.10,
        "25-50pure": 1.05,
    }

    def _minmax_norm(series: pd.Series) -> pd.Series:
        """Min-max normalize to [0, 1], handling NaN."""
        valid = series.dropna()
        if len(valid) == 0 or valid.max() == valid.min():
            return pd.Series(0.5, index=series.index)
        return (series - valid.min()) / (valid.max() - valid.min())

    weights = pd.Series(1.0, index=gene_metadata.index)

    # Numeric scores: min-max normalize, then map to [w_min, w_max]
    for key, col in numeric_scores.items():
        if score_config.get(key, True) and col in gene_metadata.columns:
            normed = _minmax_norm(gene_metadata[col])
            mapped = w_min + (w_max - w_min) * normed.fillna(0.5)
            weights *= mapped

    # Categorical scores
    for key, col in categorical_scores.items():
        if score_config.get(key, True) and col in gene_metadata.columns:
            if key == "yeast_homolog":
                wmap = yeast_weight_map
            elif key == "rickettsia_homolog":
                wmap = rick_weight_map
            elif key == "mito_domain":
                wmap = mito_domain_weight_map
            elif key == "msms":
                wmap = msms_weight_map
            else:
                wmap = {}
            cat_weights = gene_metadata[col].map(wmap).fillna(1.0)
            weights *= cat_weights

    # Clip to reasonable range
    weights = weights.clip(lower=0.5, upper=2.0)

    return dict(zip(gene_metadata["gene_symbol"], weights))


def load_all_data(config: dict) -> dict[str, Any]:
    """Load all required data files and return a structured data dictionary.

    Returns:
        dict with keys:
            - expression: DataFrame (N_cells x P_genes, z-scored)
            - pathway_scores: DataFrame (N_cells x M_pathways, z-scored)
            - labels: DataFrame (training labels)
            - submission: DataFrame (test submission template)
            - cell_meta: DataFrame (cell line metadata)
            - gene_meta: DataFrame (gene metadata with pathway annotations)
            - pathway_meta: DataFrame (pathway metadata)
            - gene_module_map: dict (gene -> module assignments)
            - evidence_weights: dict (gene -> composite weight)
    """
    data_dir = Path(config["paths"]["data_dir"])
    features_dir = Path(config["paths"]["features_dir"])
    labels_dir = Path(config["paths"]["labels_dir"])
    metadata_dir = Path(config["paths"]["metadata_dir"])
    submission_dir = Path(config["paths"]["submission_dir"])

    # Load expression data (z-scored)
    expression = pd.read_csv(
        features_dir / "cell_expression_zscore.csv", index_col=0
    )

    # Load pathway scores (z-scored)
    pathway_scores = pd.read_csv(
        features_dir / "cell_pathway_scores_zscore.csv", index_col=0
    )

    # Load labels
    labels = pd.read_csv(labels_dir / "gene_dependency.csv")

    # Load submission template
    submission = pd.read_csv(submission_dir / "sample_submission_gene.csv")

    # Load metadata
    cell_meta = pd.read_csv(metadata_dir / "cell_line_metadata.csv")
    gene_meta = pd.read_csv(metadata_dir / "gene_metadata.csv")
    pathway_meta = pd.read_csv(metadata_dir / "pathway_metadata.csv")

    # Build gene-module mapping
    gene_module_map = build_gene_module_map(gene_meta, pathway_meta)

    # Compute evidence weights
    evidence_weights = compute_evidence_weights(gene_meta, config)

    return {
        "expression": expression,
        "pathway_scores": pathway_scores,
        "labels": labels,
        "submission": submission,
        "cell_meta": cell_meta,
        "gene_meta": gene_meta,
        "pathway_meta": pathway_meta,
        "gene_module_map": gene_module_map,
        "evidence_weights": evidence_weights,
    }


def validate_data(data: dict[str, Any]) -> list[str]:
    """Validate loaded data for consistency. Returns list of issues (empty = OK)."""
    issues = []

    expr = data["expression"]
    gene_meta = data["gene_meta"]
    pw_scores = data["pathway_scores"]
    pw_meta = data["pathway_meta"]
    gmm = data["gene_module_map"]

    # Check expression genes match metadata genes
    expr_genes = set(expr.columns)
    meta_genes = set(gene_meta["gene_symbol"])
    if expr_genes != meta_genes:
        diff = expr_genes.symmetric_difference(meta_genes)
        issues.append(f"Expression/metadata gene mismatch: {len(diff)} genes differ")

    # Check pathway score columns match metadata
    pw_cols = set(pw_scores.columns)
    pw_names = set(pw_meta["pathway_name"])
    if pw_cols != pw_names:
        diff = pw_cols.symmetric_difference(pw_names)
        issues.append(f"Pathway score/metadata mismatch: {len(diff)} pathways differ")

    # Check all expression genes have module assignments
    unassigned = [g for g in expr_genes if g not in gmm]
    if unassigned:
        issues.append(f"Genes without module assignment: {len(unassigned)}")

    # Check NaN in expression
    if expr.isna().any().any():
        nan_count = int(expr.isna().sum().sum())
        issues.append(f"Expression matrix contains {nan_count} NaN values")

    # Summary stats
    n_mods = len(MODULE_DEFINITIONS)
    genes_per_module = {i: 0 for i in range(n_mods)}
    for g, info in gmm.items():
        for m in info["modules"]:
            genes_per_module[m] += 1
    small_modules = [
        MODULE_DEFINITIONS[i]["name"]
        for i, cnt in genes_per_module.items()
        if cnt < 10
    ]
    if small_modules:
        issues.append(f"Modules with <10 genes: {small_modules}")

    return issues
