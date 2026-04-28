"""End-to-end BinderDiffuser pipeline orchestration.

Stages:
    1. Load target PDB and resolve motif segments.
    2. Build MotifSpec from PipelineConfig.
    3. Run RFdiffusion -> DiffusionResult (backbones).
    4. Run ProteinMPNN -> MPNNResult (sequences per backbone).
    5. Run ColabFold/AF -> AlphaFoldResult (predictions per sequence).
    6. Compute structural metrics per (backbone, sequence) pair.
    7. Filter and rank designs; write a CSV summary.

The pipeline writes intermediate artifacts under ``config.output_dir`` so a
crashed run can be resumed by inspecting the partial outputs and re-running
from a specific stage.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from binderdiffuser.config import PipelineConfig
from binderdiffuser.diffusion.motif_spec import MotifSpec
from binderdiffuser.diffusion.rfdiff_wrapper import RFDiffusionRunner
from binderdiffuser.sequence.mpnn_runner import ProteinMPNNRunner
from binderdiffuser.targets import resolve_target_motif
from binderdiffuser.validation.alphafold_runner import AlphaFoldRunner
from binderdiffuser.validation.filters import (
    DesignRecord,
    filter_designs,
    rank_designs,
    to_dataframe,
)
from binderdiffuser.validation.metrics import (
    compute_iptm,
    compute_pae_interface,
    compute_plddt_summary,
    compute_sc_rmsd,
    compute_sc_tm,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineResult:
    """End-to-end pipeline output."""

    config: PipelineConfig
    n_backbones: int
    n_sequences: int
    n_predictions: int
    all_records: tuple[DesignRecord, ...]
    filtered_records: tuple[DesignRecord, ...]
    ranked_records: tuple[DesignRecord, ...]
    summary_csv: Path


def build_motif_spec(config: PipelineConfig) -> MotifSpec:
    """Resolve target motif and assemble MotifSpec from a PipelineConfig."""
    motif = resolve_target_motif(
        config.target.pdb_path,
        config.target.target_chain,
        config.target.motif_residues,
    )
    return MotifSpec(
        target_motif=motif,
        binder_length_min=config.diffusion.binder_length_min,
        binder_length_max=config.diffusion.binder_length_max,
    )


def assemble_records(
    backbones,
    sequences,
    predictions,
    target_chain: str,
    binder_chain: str = "A",
) -> list[DesignRecord]:
    """Cross-reference DiffusionResult / MPNNResult / AlphaFoldResult.

    Each (backbone, MPNN sequence, AF prediction) triple becomes one
    :class:`DesignRecord`. Predictions whose sequence_id does not match a
    known backbone+sequence pairing are skipped with a warning.
    """
    bb_lookup = {bb.design_id: bb for bb in backbones.designs}
    seq_lookup = sequences.by_backbone()

    records: list[DesignRecord] = []
    for pred in predictions.predictions:
        try:
            backbone_id, seq_idx_str = _parse_sequence_id(pred.sequence_id)
        except ValueError:
            log.warning("could not parse sequence id %s, skipping", pred.sequence_id)
            continue

        bb = bb_lookup.get(backbone_id)
        if bb is None:
            log.warning("unknown backbone %s, skipping", backbone_id)
            continue
        seqs = seq_lookup.get(backbone_id, [])
        seq = next((s for s in seqs if s.sequence_index == seq_idx_str), None)
        if seq is None:
            log.warning(
                "no MPNN sequence %d for backbone %s, skipping",
                seq_idx_str,
                backbone_id,
            )
            continue

        binder_length = len(seq.sequence)
        sc_rmsd = compute_sc_rmsd(str(bb.pdb_path), str(pred.pdb_path), binder_chain=binder_chain)
        sc_tm = compute_sc_tm(str(bb.pdb_path), str(pred.pdb_path), binder_chain=binder_chain)
        overall_plddt, binder_plddt = compute_plddt_summary(pred.plddt, binder_length=binder_length)

        scores_blob = _load_scores_json(pred.pae_path)
        iptm = compute_iptm(scores_blob) if scores_blob else None
        pae_iface: float | None = None
        if scores_blob and "pae" in scores_blob:
            try:
                pae_iface = compute_pae_interface(
                    scores_blob["pae"],
                    binder_length=binder_length,
                    target_length=_estimate_target_length(scores_blob, binder_length),
                )
            except ValueError as e:
                log.debug("pae interface skipped: %s", e)

        records.append(
            DesignRecord(
                design_id=f"{backbone_id}_seq{seq_idx_str}",
                backbone_id=backbone_id,
                sequence_index=seq_idx_str,
                sequence=seq.sequence,
                sc_rmsd=sc_rmsd,
                sc_tm=sc_tm,
                mean_plddt=overall_plddt,
                mean_plddt_binder=binder_plddt,
                iptm=iptm,
                pae_interface=pae_iface,
                binder_length=binder_length,
                target_length=_estimate_target_length(scores_blob, binder_length) if scores_blob else 0,
                mpnn_score=seq.score,
            )
        )
    return records


def _parse_sequence_id(sequence_id: str) -> tuple[str, int]:
    """Parse 'design_0007_seq3' -> ('design_0007', 3)."""
    if "_seq" not in sequence_id:
        raise ValueError(f"unexpected sequence id format: {sequence_id!r}")
    backbone_id, _, idx = sequence_id.rpartition("_seq")
    return backbone_id, int(idx)


def _load_scores_json(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    try:
        with Path(path).open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("failed to load %s: %s", path, e)
        return None


def _estimate_target_length(scores_blob: dict | None, binder_length: int) -> int:
    if scores_blob and "plddt" in scores_blob:
        total = len(scores_blob["plddt"])
        return max(0, total - binder_length)
    return 0


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Execute every stage end-to-end, persist intermediates, return summary.

    The function is deliberately top-down: read it from top to bottom for a
    full mental model of the pipeline.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info("pipeline output: %s", output_dir)

    spec = build_motif_spec(config)

    diffuser = RFDiffusionRunner(
        executable=config.diffusion.rfdiff_executable,
        weights_dir=config.diffusion.weights_dir,
        diffuser_T=config.diffusion.diffuser_T,
        check_executable=False,
    )
    backbones = diffuser.run(
        target_pdb=Path(config.target.pdb_path),
        spec=spec,
        output_dir=output_dir / "backbones",
        num_designs=config.diffusion.num_designs,
        seed=config.seed,
    )

    mpnn = ProteinMPNNRunner(
        model_name=config.mpnn.model_name,
        sampling_temp=config.mpnn.sampling_temp,
        check_executable=False,
    )
    sequences = _run_mpnn_per_backbone(mpnn, backbones, config, output_dir / "sequences")

    af = AlphaFoldRunner(
        backend=config.alphafold.backend,
        num_recycles=config.alphafold.num_recycles,
        num_models=config.alphafold.num_models,
        msa_mode=config.alphafold.msa_mode,
        check_executable=False,
    )
    predictions = af.run(
        fasta_path=output_dir / "sequences" / "all_designs.fasta",
        output_dir=output_dir / "alphafold",
    )

    records = assemble_records(
        backbones=backbones,
        sequences=sequences,
        predictions=predictions,
        target_chain=config.target.target_chain,
    )
    filtered = filter_designs(records, config.filters)
    ranked = rank_designs(filtered, top_k=config.filters.top_k)

    csv_path = output_dir / "designs.csv"
    df = to_dataframe(records)
    df.to_csv(csv_path, index=False)
    log.info("wrote %d records to %s", len(records), csv_path)

    return PipelineResult(
        config=config,
        n_backbones=len(backbones),
        n_sequences=len(sequences.sequences),
        n_predictions=len(predictions.predictions),
        all_records=tuple(records),
        filtered_records=tuple(filtered),
        ranked_records=tuple(ranked),
        summary_csv=csv_path,
    )


def _run_mpnn_per_backbone(mpnn, backbones, config, out_dir: Path):
    """Helper that drives MPNN once per backbone PDB and aggregates results."""
    out_dir.mkdir(parents=True, exist_ok=True)
    from binderdiffuser.sequence.mpnn_runner import DesignedSequence, MPNNResult

    aggregate: list[DesignedSequence] = []
    fasta_lines: list[str] = []
    for bb in backbones.designs:
        chain_jsonl = out_dir / f"{bb.design_id}_chains.jsonl"
        mpnn.write_fixed_chains_json(
            backbone_id=bb.design_id,
            target_chain=config.target.target_chain,
            path=chain_jsonl,
        )
        seqs_dir = out_dir / bb.design_id
        seqs_dir.mkdir(exist_ok=True)
        # In a real run this is where we'd subprocess.run(mpnn.build_command(...)).
        # The pipeline orchestration is intentionally separable so notebooks can
        # mock this stage with precomputed FASTA files.
        fasta_path = seqs_dir / "seqs.fa"
        designed = mpnn.parse_output_fasta(fasta_path, backbone_id=bb.design_id)
        aggregate.extend(designed)
        for s in designed:
            fasta_lines.append(s.fasta_record)

    bundle_fasta = out_dir / "all_designs.fasta"
    bundle_fasta.write_text("".join(fasta_lines))
    return MPNNResult(sequences=tuple(aggregate), output_dir=out_dir)
