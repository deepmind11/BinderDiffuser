"""Tests for design filtering and ranking."""

from __future__ import annotations

from binderdiffuser.config import FilterConfig
from binderdiffuser.validation.filters import (
    DesignRecord,
    composite_score,
    filter_designs,
    rank_designs,
    to_dataframe,
)


def _record(**overrides) -> DesignRecord:
    defaults = dict(
        design_id="d0",
        backbone_id="bb0",
        sequence_index=0,
        sequence="MKVL",
        sc_rmsd=1.0,
        sc_tm=0.85,
        mean_plddt=80.0,
        mean_plddt_binder=82.0,
        iptm=0.75,
        pae_interface=8.0,
        binder_length=70,
        target_length=180,
        mpnn_score=1.2,
    )
    defaults.update(overrides)
    return DesignRecord(**defaults)


class TestFilterDesigns:
    def test_pass_record_kept(self):
        cfg = FilterConfig()
        kept = filter_designs([_record()], cfg)
        assert len(kept) == 1

    def test_high_rmsd_dropped(self):
        cfg = FilterConfig(max_sc_rmsd=2.0)
        kept = filter_designs([_record(sc_rmsd=3.5)], cfg)
        assert kept == []

    def test_low_tm_dropped(self):
        cfg = FilterConfig(min_sc_tm=0.7)
        kept = filter_designs([_record(sc_tm=0.5)], cfg)
        assert kept == []

    def test_low_plddt_dropped(self):
        cfg = FilterConfig(min_plddt=70.0)
        kept = filter_designs([_record(mean_plddt=50.0)], cfg)
        assert kept == []

    def test_missing_iptm_kept(self):
        # ipTM=None must not cause exclusion.
        cfg = FilterConfig(min_iptm=0.6)
        kept = filter_designs([_record(iptm=None)], cfg)
        assert len(kept) == 1

    def test_low_iptm_dropped(self):
        cfg = FilterConfig(min_iptm=0.6)
        kept = filter_designs([_record(iptm=0.3)], cfg)
        assert kept == []

    def test_high_pae_dropped(self):
        cfg = FilterConfig(max_pae_interface=15.0)
        kept = filter_designs([_record(pae_interface=20.0)], cfg)
        assert kept == []


class TestComposite:
    def test_better_metrics_higher_score(self):
        good = _record(sc_rmsd=0.5, sc_tm=0.95, mean_plddt=90.0, iptm=0.9, pae_interface=4.0)
        bad = _record(sc_rmsd=2.5, sc_tm=0.55, mean_plddt=60.0, iptm=0.4, pae_interface=18.0)
        assert composite_score(good) > composite_score(bad)


class TestRank:
    def test_ranks_descending(self):
        a = _record(design_id="a", sc_rmsd=2.0, sc_tm=0.6, mean_plddt=70.0, iptm=0.5)
        b = _record(design_id="b", sc_rmsd=0.5, sc_tm=0.95, mean_plddt=90.0, iptm=0.9)
        ranked = rank_designs([a, b])
        assert [r.design_id for r in ranked] == ["b", "a"]

    def test_top_k(self):
        recs = [_record(design_id=f"d{i}", sc_tm=0.5 + 0.05 * i) for i in range(5)]
        ranked = rank_designs(recs, top_k=2)
        assert len(ranked) == 2
        assert ranked[0].design_id == "d4"


class TestDataFrame:
    def test_dataframe_has_composite_column(self):
        df = to_dataframe([_record()])
        assert "composite_score" in df.columns
        assert df.iloc[0]["design_id"] == "d0"

    def test_empty_list(self):
        df = to_dataframe([])
        assert df.empty
