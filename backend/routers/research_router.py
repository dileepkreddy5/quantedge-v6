"""
QuantEdge v6.0 — Research Router
=====================================
Public, read-only view of research-zone results. NOT promoted signals,
NOT investment advice. Serves a pre-computed artifact generated offline by
the point-in-time harness (research lab), never computed on request.

Endpoints:
  GET /research/cf   — Cohen-Frazzini customer-momentum directional check
"""
from __future__ import annotations

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from loguru import logger

router = APIRouter()

from core.artifact_paths import artifact_read_path
# Resolved per-request: the volume copy wins over the image-baked fallback,
# so a rebuild no longer reverts the board to a stale scan.
_ARTIFACT_NAME = "cf_artifact.json"


@router.get("/research/cf")
async def research_cf():
    """Serve the CF directional-check artifact (status: research)."""
    try:
        _p = artifact_read_path(_ARTIFACT_NAME)
        if _p is None:
            raise FileNotFoundError(_ARTIFACT_NAME)
        with open(_p) as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Research artifact not generated yet")
    except Exception as e:
        logger.warning(f"research/cf error: {e}")
        raise HTTPException(status_code=500, detail="Could not read research artifact")
