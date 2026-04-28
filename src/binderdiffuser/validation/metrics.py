"""Structural validation metrics.

Glossary
--------
* **scRMSD** (self-consistency RMSD): Cα-RMSD between the RFdiffusion-generated
  backbone and the AlphaFold prediction of the MPNN-designed sequence. Low
  scRMSD = AF agrees the sequence folds back to the diffuser's intended shape.
* **scTM** (self-consistency TM-score): TM-score variant of the same comparison;
  in [0, 1], where >= 0.5 implies the same fold.
* **pLDDT**: Per-residue confidence from AlphaFold (0-100). We surface the mean
  and the binder-only mean.
* **ipTM**: AF2-Multimer interface predicted TM, in [0, 1]. Higher means AF
  is confident in the inter-chain pose. Reported by AF in the scores JSON
  when folding multimer; for monomer-only folds we return None.
* **pAE_interface**: Mean predicted aligned error across binder-target residue
  pairs (Å). Lower = AF more confident in the relative chain placement.

Implementation notes
--------------------
* TM-score uses the canonical Zhang & Skolnick formulation with the
  length-dependent normalization constant d0(L) (TM-align paper, 2005).
* scRMSD is computed after Kabsch superposition over the binder-chain Cα
  atoms (target chain is held fixed; we are not re-aligning the target).
* All metrics operate on Biopython Structure objects so callers can drop in
  any PDB without going through ColabFold.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Structure import Structure


@dataclass(frozen=True)
class StructureMetrics:
    """Bundle of structural validation metrics for one designed sequence."""

    sc_rmsd: float
    sc_tm: float
    mean_plddt: float
    mean_plddt_binder: float | None
    iptm: float | None
    pae_interface: float | None
    binder_length: int
    target_length: int


def _ca_coords(structure: Structure, chain_id: str) -> np.ndarray:
    """Extract Cα coordinates of standard residues from a chain."""
    model = next(structure.get_models())
    if chain_id not in {c.id for c in model}:
        raise KeyError(f"chain {chain_id!r} not in structure")
    chain = model[chain_id]
    coords: list[list[float]] = []
    for residue in chain:
        hetflag, _resnum, _icode = residue.id
        if hetflag.strip():
            continue
        if "CA" in residue:
            coords.append(list(residue["CA"].coord))
    if not coords:
        raise ValueError(f"no CA atoms in chain {chain_id!r}")
    return np.asarray(coords, dtype=np.float64)


def kabsch_rmsd(p: np.ndarray, q: np.ndarray) -> tuple[float, np.ndarray]:
    """Cα RMSD after optimal superposition (Kabsch).

    Args:
        p: (N, 3) reference coordinates.
        q: (N, 3) mobile coordinates.

    Returns:
        (rmsd, q_aligned).
    """
    if p.shape != q.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q.shape}")
    p_c = p - p.mean(axis=0)
    q_c = q - q.mean(axis=0)
    h = q_c.T @ p_c
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    sign = np.diag([1.0, 1.0, d])
    rotation = vt.T @ sign @ u.T
    q_aligned = q_c @ rotation.T + p.mean(axis=0)
    diff = p - q_aligned
    rmsd = float(np.sqrt((diff * diff).sum() / p.shape[0]))
    return rmsd, q_aligned


def tm_score(p: np.ndarray, q_aligned: np.ndarray, ref_length: int | None = None) -> float:
    """Zhang-Skolnick TM-score on already-aligned Cα coordinates.

    .. math::
        TM = \\frac{1}{L_{ref}} \\sum_i \\frac{1}{1 + (d_i/d_0)^2}

    where d0(L) = 1.24 * (L - 15)^(1/3) - 1.8 (clamped at 0.5).
    """
    if p.shape != q_aligned.shape:
        raise ValueError(f"shape mismatch: {p.shape} vs {q_aligned.shape}")
    L = ref_length if ref_length is not None else p.shape[0]
    if L < 16:
        d0 = 0.5
    else:
        d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
        d0 = max(d0, 0.5)
    distances = np.linalg.norm(p - q_aligned, axis=1)
    score = float(np.sum(1.0 / (1.0 + (distances / d0) ** 2)) / L)
    return score


def compute_sc_rmsd(
    backbone_pdb: str,
    folded_pdb: str,
    binder_chain: str = "A",
) -> float:
    """scRMSD between RFdiffusion backbone and AlphaFold prediction (binder Cα)."""
    parser = PDBParser(QUIET=True)
    bb = parser.get_structure("bb", backbone_pdb)
    fold = parser.get_structure("fold", folded_pdb)
    bb_ca = _ca_coords(bb, binder_chain)
    fold_ca = _ca_coords(fold, binder_chain)
    if bb_ca.shape[0] != fold_ca.shape[0]:
        n = min(bb_ca.shape[0], fold_ca.shape[0])
        bb_ca = bb_ca[:n]
        fold_ca = fold_ca[:n]
    rmsd, _ = kabsch_rmsd(bb_ca, fold_ca)
    return rmsd


def compute_sc_tm(
    backbone_pdb: str,
    folded_pdb: str,
    binder_chain: str = "A",
) -> float:
    """Self-consistency TM-score on the binder chain."""
    parser = PDBParser(QUIET=True)
    bb = parser.get_structure("bb", backbone_pdb)
    fold = parser.get_structure("fold", folded_pdb)
    bb_ca = _ca_coords(bb, binder_chain)
    fold_ca = _ca_coords(fold, binder_chain)
    if bb_ca.shape[0] != fold_ca.shape[0]:
        n = min(bb_ca.shape[0], fold_ca.shape[0])
        bb_ca = bb_ca[:n]
        fold_ca = fold_ca[:n]
    _rmsd, fold_aligned = kabsch_rmsd(bb_ca, fold_ca)
    return tm_score(bb_ca, fold_aligned, ref_length=bb_ca.shape[0])


def compute_plddt_summary(
    plddt: list[float],
    binder_length: int | None = None,
) -> tuple[float, float | None]:
    """Return (overall mean pLDDT, binder-only mean pLDDT or None)."""
    if not plddt:
        return 0.0, None
    arr = np.asarray(plddt, dtype=np.float64)
    overall = float(arr.mean())
    binder_mean: float | None = None
    if binder_length is not None and binder_length > 0:
        binder_mean = float(arr[:binder_length].mean())
    return overall, binder_mean


def compute_iptm(scores_json: dict) -> float | None:
    """Pull ipTM from a ColabFold/AF2 scores JSON.

    Returns None if the field is absent (e.g. monomer fold).
    """
    if "iptm" in scores_json:
        try:
            return float(scores_json["iptm"])
        except (TypeError, ValueError):
            return None
    if "iptm+ptm" in scores_json:
        try:
            return float(scores_json["iptm+ptm"])
        except (TypeError, ValueError):
            return None
    return None


def compute_pae_interface(
    pae: list[list[float]] | np.ndarray,
    binder_length: int,
    target_length: int,
) -> float:
    """Mean PAE across the inter-chain (binder x target) submatrices.

    The PAE matrix is (L, L) in residue order; binder residues come first
    (length ``binder_length``), then target. We average the two off-diagonal
    blocks::

        PAE[binder, target] and PAE[target, binder].
    """
    pae_arr = np.asarray(pae, dtype=np.float64)
    L = binder_length + target_length
    if pae_arr.shape != (L, L):
        raise ValueError(
            f"pae shape {pae_arr.shape} does not match L={L} "
            f"(binder={binder_length}, target={target_length})"
        )
    block_bt = pae_arr[:binder_length, binder_length:]
    block_tb = pae_arr[binder_length:, :binder_length]
    return float((block_bt.mean() + block_tb.mean()) / 2.0)
