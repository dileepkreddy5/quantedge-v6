"""Artifact locations.

Scan artifacts were written to backend/research_data/, which lives inside the
image. Every `docker compose build backend` restored the git copy over whatever
the nightly job had produced, so the boards served a frozen June scan no matter
how often the scans ran.

Artifacts now live under a mounted volume (ARTIFACT_DIR, default /app/data/artifacts
on the edgar_data volume) and survive rebuilds. Readers fall back to the legacy
baked path so a fresh deploy still serves something until the first scan lands.
"""
from __future__ import annotations
import os
from pathlib import Path

ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "/app/data/artifacts"))
LEGACY_DIR = Path(__file__).resolve().parent.parent / "research_data"


def artifact_write_path(name: str) -> Path:
    """Durable location a job should write to."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACT_DIR / name


def artifact_read_path(name: str) -> Path | None:
    """Newest available copy: volume first, then the baked fallback."""
    for p in (ARTIFACT_DIR / name, LEGACY_DIR / name):
        if p.exists():
            return p
    return None
