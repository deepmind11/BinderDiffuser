"""Tests for the RFdiffusion contig string builder."""

from __future__ import annotations

import random

import pytest

from binderdiffuser.diffusion.motif_spec import MotifSpec, build_contig_string
from binderdiffuser.targets import MotifSegment, TargetMotif


def _make_motif(segments: tuple[MotifSegment, ...], chain: str = "A") -> TargetMotif:
    return TargetMotif(target_chain=chain, segments=segments)


class TestMotifSpecValidation:
    def test_max_below_min_rejected(self):
        motif = _make_motif((MotifSegment("A", 56, 65),))
        with pytest.raises(ValueError, match="binder_length_max"):
            MotifSpec(target_motif=motif, binder_length_min=80, binder_length_max=60)

    def test_flanking_inverted_rejected(self):
        motif = _make_motif((MotifSegment("A", 56, 65),))
        with pytest.raises(ValueError, match="flanking_max"):
            MotifSpec(
                target_motif=motif,
                binder_length_min=60,
                binder_length_max=100,
                flanking_min=20,
                flanking_max=10,
            )

    def test_binder_min_below_motif_total_rejected(self):
        motif = _make_motif((MotifSegment("A", 1, 50),))  # 50 residues
        with pytest.raises(ValueError, match="motif residues"):
            MotifSpec(target_motif=motif, binder_length_min=40, binder_length_max=80)


class TestBuildContigString:
    def test_single_segment_basic_format(self):
        motif = _make_motif((MotifSegment("A", 56, 65),))
        spec = MotifSpec(
            target_motif=motif,
            binder_length_min=60,
            binder_length_max=80,
            flanking_min=10,
            flanking_max=30,
        )
        contig = build_contig_string(spec, rng=random.Random(0))
        # Expect target block / 0 binder block
        assert "/" in contig
        target_block, binder_block = contig.split("/")
        assert target_block.startswith("A1-")
        assert "A56-65" in binder_block
        assert "0 " in contig  # chain break marker

    def test_two_segments_alternation(self):
        motif = _make_motif((MotifSegment("A", 30, 35), MotifSegment("A", 80, 90)))
        spec = MotifSpec(
            target_motif=motif,
            binder_length_min=60,
            binder_length_max=80,
            flanking_min=5,
            flanking_max=20,
        )
        contig = build_contig_string(spec, rng=random.Random(1))
        binder_block = contig.split("/")[1]
        # Three flanks + two motif tokens = 5 fragments
        fragments = [f.strip() for f in binder_block.split(",") if f.strip()]
        # First fragment also contains "0 " prefix from chain break
        assert "A30-35" in binder_block
        assert "A80-90" in binder_block
        assert len(fragments) == 5

    def test_total_length_within_envelope(self):
        motif = _make_motif((MotifSegment("A", 56, 65),))  # 10-residue motif
        spec = MotifSpec(
            target_motif=motif,
            binder_length_min=70,
            binder_length_max=70,  # force exact length
            flanking_min=5,
            flanking_max=60,
        )
        contig = build_contig_string(spec, rng=random.Random(2))
        binder_block = contig.split("/")[1].lstrip("0 ")
        # Sum sampled flanks + motif length
        total = 0
        for frag in binder_block.split(","):
            frag = frag.strip()
            if frag.startswith("A"):
                # motif segment: 'A56-65'
                start, end = frag[1:].split("-")
                total += int(end) - int(start) + 1
            else:
                # sampled flank: digit or 'min-max' (single value here)
                total += int(frag)
        assert total == 70

    def test_reproducibility_with_seed(self):
        motif = _make_motif((MotifSegment("A", 56, 65),))
        spec = MotifSpec(target_motif=motif)
        a = build_contig_string(spec, rng=random.Random(123))
        b = build_contig_string(spec, rng=random.Random(123))
        assert a == b
