"""Tests for preprocess module — data loading and gene-module mapping."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np

from src.preprocess import (
    MODULE_DEFINITIONS,
    _leaf_to_pw_key,
    _build_pathway_to_module_map,
    build_gene_module_map,
    compute_evidence_weights,
    load_all_data,
    validate_data,
)
from src.utils import load_config


class TestPathwayNameConversion:
    """Test leaf pathway name -> pathway_metadata key conversion."""

    def test_simple_space_to_underscore(self):
        assert _leaf_to_pw_key("CI subunits") == "CI_subunits"
        assert _leaf_to_pw_key("TCA cycle") == "TCA_cycle"

    def test_comma_removal(self):
        assert _leaf_to_pw_key("Cholesterol, bile acid, steroid synthesis") == \
            "Cholesterol_bile_acid_steroid_synthesis"
        assert _leaf_to_pw_key("Q-linked reactions, other") == "Q-linked_reactions_other"

    def test_hyphen_preserved(self):
        assert _leaf_to_pw_key("Fe-S-containing proteins") == "Fe-S-containing_proteins"
        assert _leaf_to_pw_key("Branched-chain amino acid metabolism") == \
            "Branched-chain_amino_acid_metabolism"

    def test_single_word(self):
        assert _leaf_to_pw_key("Fusion") == "Fusion"
        assert _leaf_to_pw_key("Fission") == "Fission"


class TestModuleDefinitions:
    """Test module definition consistency."""

    def test_fourteen_modules(self):
        assert len(MODULE_DEFINITIONS) == 14

    def test_all_modules_have_required_fields(self):
        for mod in MODULE_DEFINITIONS:
            assert "name" in mod
            assert "description" in mod
            assert "pathways" in mod
            assert len(mod["pathways"]) > 0

    def test_no_duplicate_pathways(self):
        all_pathways = []
        for mod in MODULE_DEFINITIONS:
            all_pathways.extend(mod["pathways"])
        assert len(all_pathways) == len(set(all_pathways)), \
            f"Duplicate pathways found: {set([p for p in all_pathways if all_pathways.count(p) > 1])}"

    def test_pathway_to_module_map_bijection(self):
        pw_to_mod = _build_pathway_to_module_map()
        assert len(pw_to_mod) > 0
        # All values should be valid module indices
        for pw, mod_idx in pw_to_mod.items():
            assert 0 <= mod_idx < 14, f"Pathway '{pw}' maps to invalid module {mod_idx}"


class TestGeneModuleMapping:
    """Test gene-to-module mapping with real data."""

    @pytest.fixture
    def gene_meta(self):
        config = load_config()
        return pd.read_csv(
            f"{config['paths']['metadata_dir']}/gene_metadata.csv"
        )

    @pytest.fixture
    def pathway_meta(self):
        config = load_config()
        return pd.read_csv(
            f"{config['paths']['metadata_dir']}/pathway_metadata.csv"
        )

    def test_all_genes_mapped(self, gene_meta, pathway_meta):
        gmm = build_gene_module_map(gene_meta, pathway_meta)
        assert len(gmm) == len(gene_meta)
        for gene in gene_meta["gene_symbol"]:
            assert gene in gmm

    def test_all_genes_have_at_least_one_module(self, gene_meta, pathway_meta):
        gmm = build_gene_module_map(gene_meta, pathway_meta)
        for gene, info in gmm.items():
            assert len(info["modules"]) >= 1, \
                f"Gene {gene} has no modules assigned"
            assert info["n_modules"] >= 1

    def test_known_genes_map_correctly(self, gene_meta, pathway_meta):
        """Test specific well-known gene-to-module mappings."""
        gmm = build_gene_module_map(gene_meta, pathway_meta)

        # CYC1: Complex III subunit → should be in OXPHOS_CII_CIII (module 1)
        cyc1 = gmm.get("CYC1")
        assert cyc1 is not None
        assert 1 in cyc1["modules"], \
            f"CYC1 should be in OXPHOS_CII_CIII, got {cyc1['modules']}"

        # SDHA: Complex II subunit + TCA cycle → modules 1 and 3
        sdha = gmm.get("SDHA")
        assert sdha is not None
        assert 1 in sdha["modules"], "SDHA should be in OXPHOS_CII_CIII"

        # MRPL12: mitochondrial ribosomal protein → module 6
        mrpl12 = gmm.get("MRPL12")
        assert mrpl12 is not None
        assert 6 in mrpl12["modules"], \
            f"MRPL12 should be in MITO_RIBOSOME, got {mrpl12['modules']}"

    def test_genes_may_belong_to_multiple_modules(self, gene_meta, pathway_meta):
        gmm = build_gene_module_map(gene_meta, pathway_meta)
        multi_module = sum(1 for info in gmm.values() if info["n_modules"] > 1)
        assert multi_module > 0, "Some genes should belong to multiple modules"

    def test_module_sizes_reasonable(self, gene_meta, pathway_meta):
        gmm = build_gene_module_map(gene_meta, pathway_meta)
        mod_counts = {i: 0 for i in range(len(MODULE_DEFINITIONS))}
        for info in gmm.values():
            for m in info["modules"]:
                mod_counts[m] += 1
        for i, count in mod_counts.items():
            assert count >= 10, \
                f"Module {i} ({MODULE_DEFINITIONS[i]['name']}) has only {count} genes"


class TestEvidenceWeights:
    """Test evidence weight computation."""

    @pytest.fixture
    def gene_meta(self):
        config = load_config()
        return pd.read_csv(
            f"{config['paths']['metadata_dir']}/gene_metadata.csv"
        )

    def test_weights_in_reasonable_range(self, gene_meta):
        config = load_config()
        weights = compute_evidence_weights(gene_meta, config)
        assert len(weights) == len(gene_meta)
        for gene, w in weights.items():
            assert 0.5 <= w <= 2.5, \
                f"Gene {gene} weight {w} outside [0.5, 2.5]"

    def test_weights_vary(self, gene_meta):
        """Genes should have different weights based on evidence scores."""
        config = load_config()
        weights = compute_evidence_weights(gene_meta, config)
        unique_weights = len(set(round(w, 4) for w in weights.values()))
        assert unique_weights > 1, "All genes have the same weight"

    def test_disabled_weights_all_one(self, gene_meta):
        config = {"evidence_weights": {"enabled": False}}
        weights = compute_evidence_weights(gene_meta, config)
        for gene, w in weights.items():
            assert w == 1.0


class TestDataLoading:
    """Test full data loading and validation."""

    def test_load_all_data_succeeds(self):
        config = load_config()
        data = load_all_data(config)
        assert "expression" in data
        assert "gene_module_map" in data
        assert "evidence_weights" in data

    def test_expression_shape(self):
        config = load_config()
        data = load_all_data(config)
        expr = data["expression"]
        assert expr.shape == (1140, 1123)

    def test_validate_no_critical_issues(self):
        config = load_config()
        data = load_all_data(config)
        issues = validate_data(data)
        # Allow small-module warnings, but no data corruption issues
        critical = [i for i in issues if "mismatch" in i.lower() or "nan" in i.lower()]
        assert len(critical) == 0, f"Critical issues: {critical}"

    def test_cell_line_ids_consistent(self):
        config = load_config()
        data = load_all_data(config)
        expr_cells = set(data["expression"].index)
        meta_cells = set(data["cell_meta"]["cell_line_id"])
        pw_cells = set(data["pathway_scores"].index)
        assert expr_cells == meta_cells
        assert expr_cells == pw_cells
