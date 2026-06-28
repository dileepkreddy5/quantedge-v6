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
_ARTIFACT = Path(__file__).resolve().parent.parent / "research_data" / "scan_artifact.json"


@router.get("/scan/tiers")
async def scan_tiers():
    try:
        with open(_ARTIFACT) as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Scan artifact not generated yet")
    except Exception as e:
        logger.warning(f"scan/tiers error: {e}")
        raise HTTPException(status_code=500, detail="Could not read scan artifact")
