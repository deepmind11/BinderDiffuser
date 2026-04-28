"""RFdiffusion *contig string* builder.

RFdiffusion expresses what to scaffold and what to keep fixed via a small DSL
called the contig grammar. A contig string is a comma-separated list of
fragments, where each fragment is either:

    - A *fixed motif* segment, written as ``<chain><start>-<end>`` and copied
      verbatim from the input PDB (e.g. ``A56-65`` keeps PD-L1 residues 56-65
      in chain A).
    - A *sampled* segment, written as a length range ``<min>-<max>`` (e.g.
      ``20-40``), which RFdiffusion fills in with a freely diffused backbone.

Slashes (``/``) separate chains. Inter-chain breaks must be inserted whenever
a new biological chain begins (e.g. between target and binder).

Reference: https://github.com/RosettaCommons/RFdiffusion#motif-scaffolding

This module renders a high-level :class:`MotifSpec` (a target chain + motif
segments + binder length range + flanking lengths) into a syntactically valid
RFdiffusion contig string.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from binderdiffuser.targets import MotifSegment, TargetMotif


@dataclass(frozen=True)
class MotifSpec:
    """High-level specification for a motif-scaffolding run.

    Attributes:
        target_motif: Resolved target chain + motif segments.
        binder_length_min: Minimum total residues in the generated binder chain.
        binder_length_max: Maximum total residues in the generated binder chain.
        flanking_min: Minimum sampled residues on each side of each motif segment
            inside the binder chain. Allows the diffuser room to grow scaffold.
        flanking_max: Maximum sampled residues per flank.
        keep_target_intact: If True, target chain residues outside the motif are
            kept verbatim (typical for binder design — we do not want to redesign
            the target).
    """

    target_motif: TargetMotif
    binder_length_min: int = 60
    binder_length_max: int = 100
    flanking_min: int = 5
    flanking_max: int = 30
    keep_target_intact: bool = True
    extra_target_residues: tuple[MotifSegment, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.binder_length_max < self.binder_length_min:
            raise ValueError("binder_length_max must be >= binder_length_min")
        if self.flanking_max < self.flanking_min:
            raise ValueError("flanking_max must be >= flanking_min")
        if self.binder_length_min < self.target_motif.total_residues:
            raise ValueError(
                "binder_length_min must be >= total motif residues "
                f"({self.target_motif.total_residues})"
            )


def _format_segment(segment: MotifSegment) -> str:
    return segment.as_rfdiff_token()


def _format_sampled(min_len: int, max_len: int) -> str:
    if min_len == max_len:
        return str(min_len)
    return f"{min_len}-{max_len}"


def build_contig_string(
    spec: MotifSpec,
    *,
    rng: random.Random | None = None,
) -> str:
    """Render a :class:`MotifSpec` into an RFdiffusion contig string.

    Layout produced:

        <target_chain_full> / <flank> <motif_seg1> <flank> ... <motif_segN> <flank>

    Concretely, given target chain A spanning residues 1-200 with motif at
    A56-65 and binder length 60-100, the contig string is roughly::

        A1-200/0 5-30,A56-65,5-30

    The leading ``A1-200/0`` keeps the entire target intact and inserts a
    chain break. The trailing comma-separated fragments describe the binder
    chain: a sampled flank, the fixed motif, another sampled flank.

    Note: This builder produces *one* sampling realization per call. The
    caller is expected to call it ``num_designs`` times to generate diverse
    contigs (RFdiffusion will then run inference per contig).

    Args:
        spec: The motif specification.
        rng: Optional random.Random for reproducibility.

    Returns:
        A contig string suitable for ``--contigs '<string>'`` on the
        RFdiffusion CLI.
    """
    rng = rng or random.Random()

    # Target chain block: keep it intact end-to-end, then chain break.
    target_blocks: list[str] = []
    if spec.keep_target_intact:
        target_blocks.append(_target_full_token(spec))
    target_blocks.extend(_format_segment(s) for s in spec.extra_target_residues)
    target_block = ",".join(target_blocks)

    # Binder chain block: alternating <flank> <motif_segment> <flank> ...
    binder_fragments = _build_binder_fragments(spec, rng)
    binder_block = ",".join(binder_fragments)

    return f"{target_block}/0 {binder_block}"


def _target_full_token(spec: MotifSpec) -> str:
    """Token covering the entire target chain.

    For binder design we typically write the *full* target chain (e.g.
    ``A1-200``) so RFdiffusion treats the target as immutable context.
    Because we usually do not know the target chain length here, we use
    the convention ``<chain>1-{N}`` where N is the highest residue number
    seen in the motif rounded up to a sensible value. Callers who need
    exact bounds should pass them via :attr:`MotifSpec.extra_target_residues`.
    """
    chain = spec.target_motif.target_chain
    max_motif_resnum = max(s.end for s in spec.target_motif.segments)
    # We do not know the chain length without parsing again; use motif end as
    # a conservative lower bound. Callers can override via extra_target_residues.
    return f"{chain}1-{max_motif_resnum}"


def _build_binder_fragments(
    spec: MotifSpec,
    rng: random.Random,
) -> list[str]:
    """Build the sampled-flank + motif fragments of the binder chain.

    Honours ``binder_length_min/max``: we draw flank sizes that, together with
    the fixed motif residues, fall inside the requested binder length range.
    """
    motif_total = spec.target_motif.total_residues
    n_segments = len(spec.target_motif.segments)
    n_flanks = n_segments + 1

    target_total = rng.randint(spec.binder_length_min, spec.binder_length_max)
    flank_budget = target_total - motif_total
    if flank_budget < n_flanks * spec.flanking_min:
        # Bump binder total up to honor flank minima.
        flank_budget = n_flanks * spec.flanking_min
    if flank_budget > n_flanks * spec.flanking_max:
        flank_budget = n_flanks * spec.flanking_max

    flank_sizes = _split_budget(flank_budget, n_flanks, spec.flanking_min, spec.flanking_max, rng)

    fragments: list[str] = []
    for i, seg in enumerate(spec.target_motif.segments):
        fragments.append(_format_sampled(flank_sizes[i], flank_sizes[i]))
        fragments.append(_format_segment(seg))
    fragments.append(_format_sampled(flank_sizes[-1], flank_sizes[-1]))
    return fragments


def _split_budget(
    total: int,
    n_parts: int,
    min_each: int,
    max_each: int,
    rng: random.Random,
) -> list[int]:
    """Distribute ``total`` integer units across ``n_parts`` slots.

    Each slot ends up in [min_each, max_each]. Distribution is uniform-ish:
    we start with min_each per slot and randomly grow slots until ``total``
    is consumed.
    """
    if n_parts * min_each > total or n_parts * max_each < total:
        raise ValueError(
            f"cannot split budget {total} into {n_parts} parts in [{min_each},{max_each}]"
        )

    sizes = [min_each] * n_parts
    remaining = total - sum(sizes)
    while remaining > 0:
        i = rng.randrange(n_parts)
        if sizes[i] < max_each:
            sizes[i] += 1
            remaining -= 1
    return sizes
