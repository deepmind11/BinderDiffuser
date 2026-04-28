"""BinderDiffuser: de novo protein binder design pipeline.

A motif-scaffolded diffusion + sequence-design + structural-validation stack
for generating target-specific protein binders.

Top-level submodules:
    targets       - target PDB parsing and motif extraction
    diffusion     - RFdiffusion wrapper and contig spec building
    sequence      - ProteinMPNN sequence design
    validation    - AlphaFold2 runner, metrics, and ranking
    pipeline      - end-to-end orchestration
    viz           - figure generation
"""

__version__ = "0.1.0"
__author__ = "Harshit Ghosh"

from binderdiffuser.config import PipelineConfig

__all__ = ["PipelineConfig", "__version__"]
