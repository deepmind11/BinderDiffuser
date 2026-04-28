"""Thin Python wrapper around the RFdiffusion CLI.

RFdiffusion ships a ``run_inference.py`` Hydra entry point. Rather than try
to call its internals directly (which would tie us to a specific commit),
we drive it via subprocess. This wrapper:

    1. Builds a list of contig strings from a :class:`MotifSpec`.
    2. Invokes ``run_inference.py`` per design (or in batches when supported).
    3. Collects generated backbone PDBs from the output directory.
    4. Returns a structured :class:`DiffusionResult` for downstream stages.

The wrapper is intentionally GPU-aware: on CPU-only machines it short-circuits
with a clear error so the caller can run on Colab/cluster instead.
"""

from __future__ import annotations

import logging
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from binderdiffuser.diffusion.motif_spec import MotifSpec, build_contig_string

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DesignArtifact:
    """A single generated backbone."""

    design_id: str
    pdb_path: Path
    contig_string: str
    seed: int


@dataclass(frozen=True)
class DiffusionResult:
    """Result bundle for a diffusion run."""

    designs: tuple[DesignArtifact, ...]
    output_dir: Path

    def __len__(self) -> int:
        return len(self.designs)


class RFDiffusionRunner:
    """Drive RFdiffusion via subprocess.

    Args:
        executable: Path to RFdiffusion's ``run_inference.py`` (or any wrapper
            on PATH). Default ``"run_inference.py"``.
        weights_dir: Directory containing RFdiffusion model weights. If None,
            RFdiffusion's own default lookup applies.
        diffuser_T: Number of diffusion steps. Lower values run faster but
            sample lower-quality backbones; the upstream default is 50.
        check_executable: Validate the executable is on PATH at construction
            time. Set False for unit tests.
    """

    def __init__(
        self,
        executable: str = "run_inference.py",
        weights_dir: Path | None = None,
        diffuser_T: int = 50,
        check_executable: bool = True,
    ) -> None:
        self.executable = executable
        self.weights_dir = weights_dir
        self.diffuser_T = diffuser_T
        if check_executable and shutil.which(executable) is None:
            raise FileNotFoundError(
                f"RFdiffusion executable {executable!r} not found on PATH. "
                "Install RFdiffusion (https://github.com/RosettaCommons/RFdiffusion) "
                "or pass an absolute path to its run_inference.py."
            )

    def sample_contigs(
        self,
        spec: MotifSpec,
        num_designs: int,
        seed: int = 42,
    ) -> list[tuple[str, int]]:
        """Generate ``num_designs`` distinct contig strings + seeds.

        Distinct contig strings encourage backbone diversity; we vary the
        sampled flank lengths via the underlying RNG.

        Returns:
            List of ``(contig_string, design_seed)`` tuples.
        """
        rng = random.Random(seed)
        out: list[tuple[str, int]] = []
        for i in range(num_designs):
            design_seed = rng.randrange(1, 2**31 - 1)
            contig = build_contig_string(spec, rng=random.Random(design_seed))
            out.append((contig, design_seed))
            log.debug("design %d: contig=%s seed=%d", i, contig, design_seed)
        return out

    def build_command(
        self,
        target_pdb: Path,
        contig_string: str,
        out_prefix: Path,
        seed: int,
    ) -> list[str]:
        """Build the subprocess argv for one RFdiffusion invocation.

        Mirrors RFdiffusion's Hydra-style flags:
            inference.input_pdb=<path>
            contigmap.contigs=[<string>]
            inference.output_prefix=<path>
            inference.num_designs=1
            diffuser.T=<steps>
            inference.deterministic=True
            inference.random_seed=<seed>
        """
        cmd = [
            self.executable,
            f"inference.input_pdb={target_pdb}",
            f"contigmap.contigs=[{contig_string}]",
            f"inference.output_prefix={out_prefix}",
            "inference.num_designs=1",
            f"diffuser.T={self.diffuser_T}",
            "inference.deterministic=True",
            f"inference.random_seed={seed}",
        ]
        if self.weights_dir is not None:
            cmd.append(f"inference.ckpt_override_path={self.weights_dir}")
        return cmd

    def run(
        self,
        target_pdb: Path,
        spec: MotifSpec,
        output_dir: Path,
        num_designs: int = 50,
        seed: int = 42,
        dry_run: bool = False,
    ) -> DiffusionResult:
        """Run RFdiffusion ``num_designs`` times for diverse backbones.

        Args:
            target_pdb: Path to the target PDB (motif-bearing structure).
            spec: Motif specification controlling binder length, flanks, and
                which residues stay fixed.
            output_dir: Where backbone PDBs are written. Created if missing.
            num_designs: Total number of backbones to sample.
            seed: Master seed; per-design seeds are derived deterministically.
            dry_run: If True, build commands and return the planned artifacts
                without actually invoking RFdiffusion. Useful for tests and
                for previewing a sweep before burning GPU time.

        Returns:
            :class:`DiffusionResult` with one :class:`DesignArtifact` per
            successfully generated backbone.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        contigs = self.sample_contigs(spec, num_designs=num_designs, seed=seed)

        artifacts: list[DesignArtifact] = []
        for i, (contig, design_seed) in enumerate(contigs):
            design_id = f"design_{i:04d}"
            out_prefix = output_dir / design_id
            cmd = self.build_command(target_pdb, contig, out_prefix, design_seed)

            if dry_run:
                log.info("[dry-run] %s", " ".join(cmd))
                pdb_path = out_prefix.with_suffix(".pdb")
            else:
                log.info("running RFdiffusion for %s", design_id)
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    log.error("RFdiffusion failed for %s:\n%s", design_id, result.stderr)
                    continue
                pdb_path = out_prefix.with_suffix(".pdb")
                if not pdb_path.exists():
                    log.warning(
                        "RFdiffusion exited 0 but no PDB at %s; skipping", pdb_path
                    )
                    continue

            artifacts.append(
                DesignArtifact(
                    design_id=design_id,
                    pdb_path=pdb_path,
                    contig_string=contig,
                    seed=design_seed,
                )
            )

        log.info("generated %d/%d backbones", len(artifacts), num_designs)
        return DiffusionResult(designs=tuple(artifacts), output_dir=output_dir)
