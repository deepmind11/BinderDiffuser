"""Filter and rank designed binders by structural metrics.

A successful binder design satisfies *all* of:
    1. Self-consistency: AF prediction matches diffuser backbone
       (low scRMSD, high scTM).
    2. Confident fold: high mean pLDDT (binder chain).
    3. Confident interface: high ipTM, low pAE_interface (multimer prediction).

This module provides:
    - :class:`DesignRecord` — flat row tying a sequence to its metrics.
    - :func:`filter_designs` — apply hard thresholds from FilterConfig.
    - :func:`rank_designs` — sort by a composite score.
    - :func:`to_dataframe` — pandas DataFrame for notebooks / CSV export.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from binderdiffuser.config import FilterConfig


@dataclass(frozen=True)
class DesignRecord:
    """Flat record for a single (backbone, sequence) design candidate."""

    design_id: str
    backbone_id: str
    sequence_index: int
    sequence: str
    sc_rmsd: float
    sc_tm: float
    mean_plddt: float
    mean_plddt_binder: float | None
    iptm: float | None
    pae_interface: float | None
    binder_length: int
    target_length: int
    mpnn_score: float


def composite_score(record: DesignRecord) -> float:
    """Single scalar to rank designs by.

    Higher is better. Combines:
        + scTM            (fold agreement)
        + ipTM            (interface confidence)
        + mean_plddt/100  (overall confidence)
        - scRMSD * 0.1    (penalty for backbone disagreement)
        - pae_iface * 0.02 (penalty for sloppy interface PAE)

    Missing optional metrics (ipTM, pAE) contribute 0 / 0 respectively so
    monomer-only folds still get a sensible ordering.
    """
    iptm = record.iptm if record.iptm is not None else 0.0
    pae = record.pae_interface if record.pae_interface is not None else 0.0
    return (
        record.sc_tm
        + iptm
        + record.mean_plddt / 100.0
        - 0.1 * record.sc_rmsd
        - 0.02 * pae
    )


def filter_designs(
    records: list[DesignRecord],
    config: FilterConfig,
) -> list[DesignRecord]:
    """Apply hard thresholds. Records missing optional metrics short-circuit
    to True for those checks (do not penalize monomer-only folds)."""
    kept: list[DesignRecord] = []
    for r in records:
        if r.sc_rmsd > config.max_sc_rmsd:
            continue
        if r.sc_tm < config.min_sc_tm:
            continue
        if r.mean_plddt < config.min_plddt:
            continue
        if r.iptm is not None and r.iptm < config.min_iptm:
            continue
        if r.pae_interface is not None and r.pae_interface > config.max_pae_interface:
            continue
        kept.append(r)
    return kept


def rank_designs(
    records: list[DesignRecord],
    top_k: int | None = None,
) -> list[DesignRecord]:
    """Sort designs by composite score (descending). Keep top_k if given."""
    sorted_records = sorted(records, key=composite_score, reverse=True)
    return sorted_records[:top_k] if top_k is not None else sorted_records


def to_dataframe(records: list[DesignRecord]) -> pd.DataFrame:
    """Render design records as a pandas DataFrame.

    Useful for CSV export, notebook inspection, and figure generation.
    """
    rows = [asdict(r) for r in records]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["composite_score"] = df.apply(
            lambda row: composite_score(DesignRecord(**{k: row[k] for k in DesignRecord.__dataclass_fields__})),
            axis=1,
        )
    return df
