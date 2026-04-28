"""Tests for target PDB parsing and motif extraction."""

from __future__ import annotations

import pytest

from binderdiffuser.targets import (
    MotifSegment,
    chain_residue_numbers,
    extract_motif_segments,
    get_chain,
    list_chains,
    load_structure,
    resolve_target_motif,
)


class TestMotifSegment:
    def test_token_format(self):
        seg = MotifSegment(chain="A", start=56, end=65)
        assert seg.as_rfdiff_token() == "A56-65"
        assert seg.length == 10

    def test_invalid_range_rejected(self):
        with pytest.raises(ValueError, match="end .* must be >= start"):
            MotifSegment(chain="A", start=10, end=5)

    def test_single_residue_motif(self):
        seg = MotifSegment(chain="A", start=42, end=42)
        assert seg.length == 1
        assert seg.as_rfdiff_token() == "A42-42"


class TestStructureLoading:
    def test_load_returns_structure(self, tiny_pdb_path):
        structure = load_structure(tiny_pdb_path, "tinytest")
        assert structure.id == "tinytest"

    def test_list_chains(self, tiny_pdb_path):
        structure = load_structure(tiny_pdb_path)
        assert list_chains(structure) == ["A", "B"]

    def test_get_chain_present(self, tiny_pdb_path):
        structure = load_structure(tiny_pdb_path)
        chain = get_chain(structure, "A")
        assert chain.id == "A"

    def test_get_chain_missing_raises(self, tiny_pdb_path):
        structure = load_structure(tiny_pdb_path)
        with pytest.raises(KeyError, match="not found"):
            get_chain(structure, "Z")


class TestChainResidues:
    def test_residue_numbers(self, tiny_pdb_path):
        structure = load_structure(tiny_pdb_path)
        chain_a = get_chain(structure, "A")
        assert chain_residue_numbers(chain_a) == list(range(1, 11))

        chain_b = get_chain(structure, "B")
        assert chain_residue_numbers(chain_b) == [1, 2]


class TestExtractMotifSegments:
    def test_single_contiguous_segment(self, tiny_pdb_path):
        chain = get_chain(load_structure(tiny_pdb_path), "A")
        segments = extract_motif_segments(chain, [3, 4, 5, 6])
        assert len(segments) == 1
        assert segments[0] == MotifSegment(chain="A", start=3, end=6)

    def test_two_disjoint_segments(self, tiny_pdb_path):
        chain = get_chain(load_structure(tiny_pdb_path), "A")
        segments = extract_motif_segments(chain, [2, 3, 7, 8, 9])
        assert len(segments) == 2
        assert segments[0] == MotifSegment(chain="A", start=2, end=3)
        assert segments[1] == MotifSegment(chain="A", start=7, end=9)

    def test_unsorted_input_handled(self, tiny_pdb_path):
        chain = get_chain(load_structure(tiny_pdb_path), "A")
        segments = extract_motif_segments(chain, [9, 7, 3, 8, 2])
        assert segments[0].start == 2
        assert segments[1].start == 7

    def test_duplicates_collapsed(self, tiny_pdb_path):
        chain = get_chain(load_structure(tiny_pdb_path), "A")
        segments = extract_motif_segments(chain, [3, 3, 4, 4])
        assert segments == (MotifSegment(chain="A", start=3, end=4),)

    def test_empty_motif_rejected(self, tiny_pdb_path):
        chain = get_chain(load_structure(tiny_pdb_path), "A")
        with pytest.raises(ValueError, match="non-empty"):
            extract_motif_segments(chain, [])

    def test_missing_residue_rejected(self, tiny_pdb_path):
        chain = get_chain(load_structure(tiny_pdb_path), "A")
        with pytest.raises(ValueError, match="not found in chain"):
            extract_motif_segments(chain, [3, 4, 99])


class TestResolveTargetMotif:
    def test_resolves_segments(self, tiny_pdb_path):
        motif = resolve_target_motif(tiny_pdb_path, "A", [3, 4, 5, 8, 9])
        assert motif.target_chain == "A"
        assert motif.total_residues == 5
        assert len(motif.segments) == 2
