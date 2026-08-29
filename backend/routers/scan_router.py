"""
QuantEdge v6.0 — Multibagger Scan Router
=========================================
Public, read-only cap-tier shortlist ranked by quarterly growth + quiet price.
Serves a pre-computed artifact (scored offline; scoring the universe is too
heavy for request time). A FILTER/shortlist, NOT a predictor — not advice.

Endpoints:
  GET /scan/tiers   — small/mid/large ranked lists + disclaimer
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
_ARTIFACT_NAME = "scan_artifact.json"


@router.get("/scan/tiers")
async def scan_tiers():
    try:
        _p = artifact_read_path(_ARTIFACT_NAME)
        if _p is None:
            raise FileNotFoundError(_ARTIFACT_NAME)
        with open(_p) as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Scan artifact not generated yet")
    except Exception as e:
        logger.warning(f"scan/tiers error: {e}")
        raise HTTPException(status_code=500, detail="Could not read scan artifact")
