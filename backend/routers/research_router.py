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

_ARTIFACT = Path(__file__).resolve().parent.parent / "research_data" / "cf_artifact.json"


@router.get("/research/cf")
async def research_cf():
    """Serve the CF directional-check artifact (status: research)."""
    try:
        with open(_ARTIFACT) as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Research artifact not generated yet")
    except Exception as e:
        logger.warning(f"research/cf error: {e}")
        raise HTTPException(status_code=500, detail="Could not read research artifact")
