"""AlphaFold2 / ColabFold inference wrapper.

After ProteinMPNN designs sequences for a generated backbone, we want to
*independently re-fold* those sequences and check that the predicted
structure matches the original RFdiffusion backbone. This is the
**self-consistency** check that filters out sequences that look reasonable
to MPNN but do not actually fold to the intended geometry.

For local M-series Macs, ColabFold (https://github.com/sokrypton/ColabFold)
is the most practical AF2 frontend: ``colabfold_batch`` accepts a FASTA and
emits PDBs + per-residue pLDDT + pAE matrices as JSON.

This module wraps ``colabfold_batch`` and produces a normalized
:class:`AlphaFoldResult` regardless of backend (colabfold today, AF3 hooks
sketched in for swap-in).
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

log = logging.getLogger(__name__)

Backend = Literal["colabfold", "alphafold2", "alphafold3"]


@dataclass(frozen=True)
class FoldedPrediction:
    """An AlphaFold prediction for one designed sequence."""

    sequence_id: str
    pdb_path: Path
    pae_path: Path | None
    plddt: list[float]
    pae: list[list[float]] | None
    rank: int
    backend: Backend

    @property
    def mean_plddt(self) -> float:
        if not self.plddt:
            return 0.0
        return sum(self.plddt) / len(self.plddt)


@dataclass(frozen=True)
class AlphaFoldResult:
    """Result bundle for a folding pass over many sequences."""

    predictions: tuple[FoldedPrediction, ...]
    output_dir: Path
    backend: Backend


class AlphaFoldRunner:
    """Drive ColabFold (default) or AlphaFold2/3 for sequence re-folding.

    Args:
        backend: Which folding backend to invoke. Only ``colabfold`` is
            wired today; ``alphafold3`` slot is reserved for the public
            release and short-circuits with a clear error.
        executable: Path / name of the backend's CLI entry point.
        num_recycles: AF2 recycles. 3 is the default; 6 sometimes helps for
            tricky binders at the cost of latency.
        num_models: How many AF2 models to score (1-5). 1 is fastest.
        msa_mode: ``single_sequence`` is appropriate for de novo binders
            (no MSA exists). Use ``mmseqs2_uniref_env`` only if testing on
            natural proteins.
        check_executable: Validate executable on PATH at construction.
    """

    def __init__(
        self,
        backend: Backend = "colabfold",
        executable: str | None = None,
        num_recycles: int = 3,
        num_models: int = 1,
        msa_mode: str = "single_sequence",
        check_executable: bool = True,
    ) -> None:
        self.backend = backend
        self.executable = executable or _default_executable(backend)
        self.num_recycles = num_recycles
        self.num_models = num_models
        self.msa_mode = msa_mode
        if check_executable and shutil.which(self.executable) is None:
            raise FileNotFoundError(
                f"{backend} executable {self.executable!r} not on PATH. "
                "Install ColabFold (https://github.com/sokrypton/ColabFold) "
                "or pass an explicit path."
            )

    def build_command(
        self,
        fasta_path: Path,
        out_dir: Path,
    ) -> list[str]:
        """Build subprocess argv for ColabFold inference.

        Other backends override this method.
        """
        if self.backend != "colabfold":
            raise NotImplementedError(
                f"backend {self.backend!r} not yet wired; only colabfold is "
                "supported. AF3 will land once the public CLI is released."
            )
        return [
            self.executable,
            str(fasta_path),
            str(out_dir),
            "--num-recycle", str(self.num_recycles),
            "--num-models", str(self.num_models),
            "--msa-mode", self.msa_mode,
            "--model-type", "auto",
            "--rank", "plddt",
        ]

    def run(
        self,
        fasta_path: Path,
        output_dir: Path,
        dry_run: bool = False,
    ) -> AlphaFoldResult:
        """Fold every sequence in ``fasta_path``.

        Args:
            fasta_path: FASTA bundle from :class:`MPNNResult`. The runner
                expects one record per sequence to fold.
            output_dir: Where ColabFold writes per-sequence PDB+JSON.
            dry_run: If True, return a planned result with empty predictions
                without invoking the backend.

        Returns:
            :class:`AlphaFoldResult` aggregating the per-sequence
            :class:`FoldedPrediction` records.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = self.build_command(fasta_path, output_dir)

        if dry_run:
            log.info("[dry-run] %s", " ".join(cmd))
            return AlphaFoldResult(predictions=(), output_dir=output_dir, backend=self.backend)

        log.info("running %s for %s", self.backend, fasta_path)
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            log.error("%s failed:\n%s", self.backend, result.stderr)
            return AlphaFoldResult(predictions=(), output_dir=output_dir, backend=self.backend)

        predictions = self.collect_predictions(output_dir)
        log.info("collected %d predictions", len(predictions))
        return AlphaFoldResult(
            predictions=predictions,
            output_dir=output_dir,
            backend=self.backend,
        )

    def collect_predictions(self, output_dir: Path) -> tuple[FoldedPrediction, ...]:
        """Walk ColabFold's output dir and assemble FoldedPrediction records.

        ColabFold names PDBs as ``<seqid>_unrelaxed_rank_<N>_*.pdb`` and
        scores as ``<seqid>_scores_rank_<N>_*.json`` containing pLDDT and
        pAE arrays. We collect rank-1 only by default (highest confidence).
        """
        output_dir = Path(output_dir)
        predictions: list[FoldedPrediction] = []

        for pdb_path in sorted(output_dir.glob("*_unrelaxed_rank_001_*.pdb")):
            seq_id = pdb_path.name.split("_unrelaxed_rank_")[0]
            score_json = next(
                output_dir.glob(f"{seq_id}_scores_rank_001_*.json"),
                None,
            )
            plddt: list[float] = []
            pae: list[list[float]] | None = None
            if score_json and score_json.exists():
                with score_json.open() as f:
                    data = json.load(f)
                plddt = data.get("plddt", [])
                pae = data.get("pae")

            predictions.append(
                FoldedPrediction(
                    sequence_id=seq_id,
                    pdb_path=pdb_path,
                    pae_path=score_json,
                    plddt=plddt,
                    pae=pae,
                    rank=1,
                    backend=self.backend,
                )
            )
        return tuple(predictions)


def _default_executable(backend: Backend) -> str:
    if backend == "colabfold":
        return "colabfold_batch"
    if backend == "alphafold2":
        return "run_alphafold.sh"
    if backend == "alphafold3":
        return "run_alphafold3.py"
    raise ValueError(f"unknown backend: {backend!r}")
