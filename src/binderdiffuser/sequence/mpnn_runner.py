"""ProteinMPNN sequence-design runner.

After RFdiffusion produces a backbone (Cα/N/C atoms only), we need to assign
amino acids that will fold into that backbone. ProteinMPNN is the standard
tool for this: a graph neural network trained on PDB to score residue identity
given a backbone.

For binder design we want:
    - Target chain residues fixed (we are not redesigning the target).
    - Motif residues on the binder fixed (they are templated on the target
      hot-spot — sequence already comes from there).
    - All other binder positions designable.

This module wraps ProteinMPNN's ``protein_mpnn_run.py`` CLI in the same
subprocess-driven fashion as the diffusion wrapper.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesignedSequence:
    """One ProteinMPNN-designed sequence for a given backbone."""

    backbone_id: str
    sequence_index: int
    sequence: str
    score: float
    sampling_temp: float

    @property
    def fasta_record(self) -> str:
        header = f">{self.backbone_id}_seq{self.sequence_index} score={self.score:.3f}"
        return f"{header}\n{self.sequence}\n"


@dataclass(frozen=True)
class MPNNResult:
    """Result bundle for a sequence-design pass over many backbones."""

    sequences: tuple[DesignedSequence, ...]
    output_dir: Path

    def by_backbone(self) -> dict[str, list[DesignedSequence]]:
        """Group designed sequences by their parent backbone id."""
        grouped: dict[str, list[DesignedSequence]] = {}
        for s in self.sequences:
            grouped.setdefault(s.backbone_id, []).append(s)
        return grouped


class ProteinMPNNRunner:
    """Drive ProteinMPNN via subprocess.

    Args:
        executable: Path to ProteinMPNN's ``protein_mpnn_run.py`` (or shim
            on PATH). Default ``"protein_mpnn_run.py"``.
        model_name: ProteinMPNN model checkpoint (``v_48_020`` is the standard
            recommended weight). Higher numeric suffix = noisier training data
            for robustness.
        sampling_temp: Sampling temperature; lower (~0.1) is more conservative,
            higher (>0.3) more diverse. The upstream paper recommends 0.1-0.2
            for binder design.
        check_executable: Validate executable on PATH at construction.
    """

    def __init__(
        self,
        executable: str = "protein_mpnn_run.py",
        model_name: str = "v_48_020",
        sampling_temp: float = 0.1,
        check_executable: bool = True,
    ) -> None:
        self.executable = executable
        self.model_name = model_name
        self.sampling_temp = sampling_temp
        if check_executable and shutil.which(executable) is None:
            raise FileNotFoundError(
                f"ProteinMPNN executable {executable!r} not on PATH. "
                "Install ProteinMPNN (https://github.com/dauparas/ProteinMPNN) "
                "or pass an absolute path."
            )

    def write_fixed_chains_json(
        self,
        backbone_id: str,
        target_chain: str,
        path: Path,
    ) -> Path:
        """Write the JSON ProteinMPNN consumes to identify fixed chains.

        ProteinMPNN's ``--chain_id_jsonl`` accepts an entry per backbone of
        the form::

            {"<backbone_id>": [["<designable>"], ["<fixed>"]]}

        For binder design we mark the target chain as fixed and the binder
        chain as designable.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # We assume a 2-chain layout: binder = "A", target = "B" (the
        # standard convention emitted by RFdiffusion in binder mode). The
        # caller can override by supplying their own JSONL.
        designable = "A" if target_chain != "A" else "B"
        record = {backbone_id: [[designable], [target_chain]]}
        with path.open("w") as f:
            f.write(json.dumps(record) + "\n")
        return path

    def write_fixed_positions_json(
        self,
        backbone_id: str,
        designable_chain: str,
        fixed_positions_in_binder: list[int],
        path: Path,
    ) -> Path:
        """Write per-position fixing JSON for motif residues on the binder.

        Args:
            backbone_id: Identifier matching the backbone PDB stem.
            designable_chain: Chain id of the binder (the side that should
                still be designable except at motif positions).
            fixed_positions_in_binder: 1-indexed residue numbers within the
                binder chain that should be left untouched.
            path: Where to write the JSONL file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            backbone_id: {designable_chain: sorted(set(fixed_positions_in_binder))}
        }
        with path.open("w") as f:
            f.write(json.dumps(record) + "\n")
        return path

    def build_command(
        self,
        pdb_dir: Path,
        out_dir: Path,
        chain_jsonl: Path,
        fixed_positions_jsonl: Path | None,
        num_seq_per_target: int,
    ) -> list[str]:
        """Build the subprocess argv for one ProteinMPNN run."""
        cmd = [
            self.executable,
            "--pdb_path_chains", str(chain_jsonl),
            "--out_folder", str(out_dir),
            "--pdb_path", str(pdb_dir),
            "--num_seq_per_target", str(num_seq_per_target),
            "--sampling_temp", str(self.sampling_temp),
            "--model_name", self.model_name,
            "--seed", "37",
        ]
        if fixed_positions_jsonl is not None:
            cmd.extend(["--fixed_positions_jsonl", str(fixed_positions_jsonl)])
        return cmd

    def parse_output_fasta(
        self,
        fasta_path: Path,
        backbone_id: str,
    ) -> list[DesignedSequence]:
        """Parse a ProteinMPNN output FASTA into DesignedSequence records.

        ProteinMPNN headers look like::

            >T=0.1, sample=1, score=1.234, ...

        We index sequences in the order they appear.
        """
        if not fasta_path.exists():
            return []
        out: list[DesignedSequence] = []
        index = 0
        score = 0.0
        seq_buf: list[str] = []
        header: str | None = None

        def flush() -> None:
            nonlocal index, header
            if header is None:
                return
            seq = "".join(seq_buf).strip().replace("/", "")
            if seq:
                out.append(
                    DesignedSequence(
                        backbone_id=backbone_id,
                        sequence_index=index,
                        sequence=seq,
                        score=score,
                        sampling_temp=self.sampling_temp,
                    )
                )
                index += 1

        with fasta_path.open() as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    flush()
                    seq_buf.clear()
                    header = line
                    score = _extract_score(line)
                else:
                    seq_buf.append(line)
        flush()
        return out


def _extract_score(header: str) -> float:
    """Pull the numeric score out of a ProteinMPNN FASTA header.

    Accepts headers like ``>T=0.1, sample=1, score=1.234, ...``.
    Returns 0.0 if the field is missing.
    """
    for tok in header.lstrip(">").split(","):
        tok = tok.strip()
        if tok.startswith("score="):
            try:
                return float(tok.split("=", 1)[1])
            except ValueError:
                return 0.0
    return 0.0
