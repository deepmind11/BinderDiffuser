"""Diffusion-based backbone generation (RFdiffusion wrapper).

Re-exports are added lazily as modules land. Currently exposes:
    - MotifSpec, build_contig_string  (motif_spec.py)
"""

from binderdiffuser.diffusion.motif_spec import MotifSpec, build_contig_string

__all__ = ["MotifSpec", "build_contig_string"]
