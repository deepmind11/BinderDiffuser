"""Configuration schemas for BinderDiffuser pipeline runs.

All configs are Pydantic models so they validate on construction. A pipeline
run is fully described by a ``PipelineConfig`` instance, which can be loaded
from a YAML file via :func:`load_config_from_yaml`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class TargetConfig(BaseModel):
    """Target protein and motif specification."""

    pdb_path: Path = Field(..., description="Path to target PDB file.")
    target_chain: str = Field(..., description="Chain ID of the target.")
    motif_chain: str = Field(
        ...,
        description="Chain ID containing motif residues (often same as target_chain).",
    )
    motif_residues: list[int] = Field(
        ...,
        description="Residue numbers (PDB numbering) defining the motif to scaffold.",
        min_length=1,
    )

    @field_validator("target_chain", "motif_chain")
    @classmethod
    def _single_char_chain(cls, v: str) -> str:
        if len(v) != 1:
            raise ValueError(f"chain id must be a single character, got {v!r}")
        return v


class DiffusionConfig(BaseModel):
    """RFdiffusion sampling parameters."""

    num_designs: int = Field(default=50, ge=1, le=10_000)
    binder_length_min: int = Field(default=60, ge=20)
    binder_length_max: int = Field(default=100, ge=20)
    diffuser_T: int = Field(default=50, ge=1, description="Number of diffusion steps.")
    rfdiff_executable: str = Field(
        default="run_inference.py",
        description="Entry-point for RFdiffusion (assumed on PATH or absolute).",
    )
    weights_dir: Path | None = Field(
        default=None, description="Optional path to RFdiffusion model weights."
    )

    @field_validator("binder_length_max")
    @classmethod
    def _range_ok(cls, v: int, info) -> int:
        min_len = info.data.get("binder_length_min")
        if min_len is not None and v < min_len:
            raise ValueError("binder_length_max must be >= binder_length_min")
        return v


class MPNNConfig(BaseModel):
    """ProteinMPNN sequence-design parameters."""

    num_seqs_per_backbone: int = Field(default=8, ge=1, le=128)
    sampling_temp: float = Field(default=0.1, gt=0.0, lt=2.0)
    model_name: Literal["v_48_002", "v_48_010", "v_48_020", "v_48_030"] = "v_48_020"
    fix_target_residues: bool = Field(
        default=True,
        description="If True, target chain residues are fixed during MPNN design.",
    )


class AlphaFoldConfig(BaseModel):
    """AlphaFold2 / ColabFold inference parameters."""

    backend: Literal["colabfold", "alphafold2", "alphafold3"] = "colabfold"
    num_recycles: int = Field(default=3, ge=0, le=20)
    num_models: int = Field(default=1, ge=1, le=5)
    use_templates: bool = Field(default=False)
    msa_mode: Literal["mmseqs2_uniref_env", "single_sequence"] = "single_sequence"


class FilterConfig(BaseModel):
    """Thresholds applied during ranking and filtering."""

    max_sc_rmsd: float = Field(default=2.0, gt=0.0)
    min_sc_tm: float = Field(default=0.7, ge=0.0, le=1.0)
    min_plddt: float = Field(default=70.0, ge=0.0, le=100.0)
    min_iptm: float = Field(default=0.6, ge=0.0, le=1.0)
    max_pae_interface: float = Field(default=15.0, gt=0.0)
    top_k: int = Field(default=10, ge=1)


class PipelineConfig(BaseModel):
    """Complete configuration for a BinderDiffuser run."""

    run_name: str = Field(..., min_length=1)
    output_dir: Path
    target: TargetConfig
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    mpnn: MPNNConfig = Field(default_factory=MPNNConfig)
    alphafold: AlphaFoldConfig = Field(default_factory=AlphaFoldConfig)
    filters: FilterConfig = Field(default_factory=FilterConfig)
    seed: int = Field(default=42)


def load_config_from_yaml(path: str | Path) -> PipelineConfig:
    """Load and validate a pipeline config from a YAML file."""
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f)
    return PipelineConfig(**data)
