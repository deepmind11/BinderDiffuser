"""Target-protein parsing and motif extraction.

A *target* is a PDB structure containing the chain we want to design a binder
against. A *motif* is a contiguous (or set of contiguous) stretches of residues
on the target whose geometry must be preserved during diffusion: typically the
hot-spot residues that define the binding interface.

This module:
    1. Loads a PDB file with Biopython.
    2. Selects the target chain.
    3. Extracts motif residues by residue number.
    4. Splits the motif into contiguous segments (the unit RFdiffusion accepts
       in its contig grammar).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.Chain import Chain
from Bio.PDB.Structure import Structure


@dataclass(frozen=True)
class MotifSegment:
    """A contiguous stretch of motif residues on a single chain."""

    chain: str
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def as_rfdiff_token(self) -> str:
        """Render as an RFdiffusion contig fragment, e.g. 'A56-65'."""
        return f"{self.chain}{self.start}-{self.end}"


@dataclass(frozen=True)
class TargetMotif:
    """Resolved motif on a parsed target."""

    target_chain: str
    segments: tuple[MotifSegment, ...]

    @property
    def total_residues(self) -> int:
        return sum(s.length for s in self.segments)


def load_structure(pdb_path: str | Path, structure_id: str = "target") -> Structure:
    """Parse a PDB file. Quiet by default (suppresses Biopython warnings).

    Args:
        pdb_path: Path to a PDB file.
        structure_id: Arbitrary identifier stored on the returned Structure.

    Returns:
        A :class:`Bio.PDB.Structure.Structure` instance.
    """
    parser = PDBParser(QUIET=True)
    return parser.get_structure(structure_id, str(pdb_path))


def get_chain(structure: Structure, chain_id: str) -> Chain:
    """Return the named chain from the first model of ``structure``.

    Raises:
        KeyError if the chain is not present.
    """
    model = next(structure.get_models())
    if chain_id not in {c.id for c in model}:
        available = sorted(c.id for c in model)
        raise KeyError(
            f"chain {chain_id!r} not found in structure; available chains: {available}"
        )
    return model[chain_id]


def list_chains(structure: Structure) -> list[str]:
    """List chain ids present in the first model of ``structure``."""
    model = next(structure.get_models())
    return sorted(c.id for c in model)


def chain_residue_numbers(chain: Chain) -> list[int]:
    """Return author residue numbers (PDB numbering) for standard amino acids only.

    Skips heteroatoms (HETATM with non-standard hetflag) and water.
    """
    nums: list[int] = []
    for residue in chain:
        hetflag, resnum, _icode = residue.id
        if hetflag.strip():
            # heteroatom — skip ligands, ions, water, modified residues
            continue
        nums.append(int(resnum))
    return nums


def extract_motif_segments(
    chain: Chain,
    motif_residues: list[int],
) -> tuple[MotifSegment, ...]:
    """Split a list of motif residue numbers into contiguous segments on ``chain``.

    The grouping is based on residue numbering in the PDB file: residues are
    contiguous if their author residue numbers differ by exactly 1 *and* both
    are present in the chain.

    Args:
        chain: The target chain.
        motif_residues: Residue numbers (PDB numbering) defining the motif.

    Returns:
        Tuple of MotifSegment, sorted by start residue.

    Raises:
        ValueError: If any motif residue is missing from ``chain`` or if the
        motif residue list is empty.
    """
    if not motif_residues:
        raise ValueError("motif_residues must be non-empty")

    chain_resnums = set(chain_residue_numbers(chain))
    missing = [r for r in motif_residues if r not in chain_resnums]
    if missing:
        raise ValueError(
            f"motif residues {missing} not found in chain {chain.id!r}"
        )

    sorted_residues = sorted(set(motif_residues))
    segments: list[MotifSegment] = []
    seg_start = sorted_residues[0]
    prev = sorted_residues[0]

    for r in sorted_residues[1:]:
        if r == prev + 1 and r in chain_resnums:
            prev = r
            continue
        segments.append(MotifSegment(chain=chain.id, start=seg_start, end=prev))
        seg_start = r
        prev = r
    segments.append(MotifSegment(chain=chain.id, start=seg_start, end=prev))

    return tuple(segments)


def resolve_target_motif(
    pdb_path: str | Path,
    target_chain: str,
    motif_residues: list[int],
) -> TargetMotif:
    """High-level helper: load PDB, validate chain, build TargetMotif.

    Returns:
        Resolved TargetMotif ready to feed into the contig-string builder.
    """
    structure = load_structure(pdb_path)
    chain = get_chain(structure, target_chain)
    segments = extract_motif_segments(chain, motif_residues)
    return TargetMotif(target_chain=target_chain, segments=segments)
