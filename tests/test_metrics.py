"""Tests for structural validation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from binderdiffuser.validation.metrics import (
    compute_iptm,
    compute_pae_interface,
    compute_plddt_summary,
    kabsch_rmsd,
    tm_score,
)


class TestKabschRmsd:
    def test_identical_coords_zero_rmsd(self):
        rng = np.random.default_rng(0)
        p = rng.normal(size=(20, 3))
        rmsd, q_aligned = kabsch_rmsd(p, p.copy())
        assert rmsd == pytest.approx(0.0, abs=1e-9)
        np.testing.assert_allclose(q_aligned, p, atol=1e-9)

    def test_translated_coords_zero_rmsd(self):
        rng = np.random.default_rng(1)
        p = rng.normal(size=(15, 3))
        q = p + np.array([3.0, -2.0, 5.0])
        rmsd, _ = kabsch_rmsd(p, q)
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_rotated_coords_zero_rmsd(self):
        rng = np.random.default_rng(2)
        p = rng.normal(size=(20, 3))
        # 90-deg rotation about z
        rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        q = p @ rot.T
        rmsd, _ = kabsch_rmsd(p, q)
        assert rmsd == pytest.approx(0.0, abs=1e-9)

    def test_known_offset_recovered(self):
        p = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        q = p.copy()
        q[0, 0] += 1.0  # one atom dislocated by 1 Å
        rmsd, _ = kabsch_rmsd(p, q)
        # Optimal alignment will distribute the error; rmsd > 0 but bounded.
        assert rmsd > 0.0
        assert rmsd < 1.0

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="shape mismatch"):
            kabsch_rmsd(np.zeros((10, 3)), np.zeros((11, 3)))


class TestTMScore:
    def test_perfect_alignment_max_score(self):
        rng = np.random.default_rng(3)
        p = rng.normal(size=(80, 3)) * 5.0
        score = tm_score(p, p.copy(), ref_length=80)
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_score_decreases_with_distortion(self):
        rng = np.random.default_rng(4)
        p = rng.normal(size=(60, 3)) * 5.0
        small_noise = p + rng.normal(scale=0.5, size=p.shape)
        big_noise = p + rng.normal(scale=5.0, size=p.shape)
        s_small = tm_score(p, small_noise, ref_length=60)
        s_big = tm_score(p, big_noise, ref_length=60)
        assert s_small > s_big
        assert 0.0 < s_big < s_small <= 1.0

    def test_d0_clamp_for_short_proteins(self):
        # Short protein triggers d0 floor (0.5).
        p = np.zeros((10, 3))
        score = tm_score(p, p.copy(), ref_length=10)
        assert score == pytest.approx(1.0, abs=1e-9)


class TestPlddtSummary:
    def test_overall_mean(self):
        plddt = [70.0, 80.0, 90.0, 60.0]
        overall, binder_mean = compute_plddt_summary(plddt)
        assert overall == pytest.approx(75.0)
        assert binder_mean is None

    def test_binder_only_mean(self):
        plddt = [80.0, 80.0, 60.0, 60.0]
        overall, binder_mean = compute_plddt_summary(plddt, binder_length=2)
        assert overall == pytest.approx(70.0)
        assert binder_mean == pytest.approx(80.0)

    def test_empty_plddt(self):
        assert compute_plddt_summary([]) == (0.0, None)


class TestIptm:
    def test_iptm_field(self):
        assert compute_iptm({"iptm": 0.85}) == pytest.approx(0.85)

    def test_iptm_ptm_combined(self):
        assert compute_iptm({"iptm+ptm": 0.7}) == pytest.approx(0.7)

    def test_missing_returns_none(self):
        assert compute_iptm({"plddt": [50.0]}) is None

    def test_non_numeric_returns_none(self):
        assert compute_iptm({"iptm": "n/a"}) is None


class TestPaeInterface:
    def test_uniform_pae_block_mean(self):
        # binder=3 target=2 -> 5x5 PAE
        pae = np.full((5, 5), 10.0)
        pae[:3, 3:] = 5.0
        pae[3:, :3] = 7.0
        result = compute_pae_interface(pae, binder_length=3, target_length=2)
        # mean of (5.0 and 7.0) = 6.0
        assert result == pytest.approx(6.0)

    def test_shape_mismatch_rejected(self):
        pae = np.zeros((4, 4))
        with pytest.raises(ValueError, match="does not match"):
            compute_pae_interface(pae, binder_length=3, target_length=2)
