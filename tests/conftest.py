"""Shared pytest fixtures for BinderDiffuser tests."""

from __future__ import annotations

from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def tiny_pdb_path() -> Path:
    """Path to a tiny synthetic PDB used in motif extraction tests."""
    return FIXTURES / "tiny.pdb"
