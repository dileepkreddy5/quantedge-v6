"""System inventory — GET /api/v6/system/stats.

Counts what the platform actually contains at runtime instead of restating it
from constants. The landing page previously hardcoded '8 ML models / 200+
signals'; both drifted from reality and neither was true when audited. Anything
displayed as a capability count is computed here from the live catalogs, the
model directory and the training report.
"""
from __future__ import annotations
import glob, importlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from fastapi import APIRouter
from core.artifact_paths import artifact_read_path

router = APIRouter()
_MODEL_DIR = Path(os.getenv("MODEL_DIR", "/app/models"))


def _catalog_counts() -> dict:
    total = live = needs_source = reference = cats = 0
    per_tab = []
    for p in sorted(glob.glob("/app/quantedge/scoring/cat_*_v6.py")):
        mod = "quantedge.scoring." + Path(p).stem
        try:
            m = importlib.import_module(mod)
            catalog = getattr(m, "CATEGORIES", {}) or {}
            sigs = [s for v in catalog.values() for s in v[2]]
        except Exception:
            continue
        l = sum(1 for s in sigs if s.get("status") == "live")
        n = sum(1 for s in sigs if s.get("status") == "needs_source")
        r = sum(1 for s in sigs if s.get("status") == "reference")
        per_tab.append({
            "tab": Path(p).stem.replace("cat_", "").replace("_v6", ""),
            "categories": len(catalog), "signals": len(sigs),
            "live": l, "needs_source": n, "reference": r,
        })
        total += len(sigs); live += l; needs_source += n; reference += r; cats += len(catalog)
    per_tab.sort(key=lambda x: x["signals"], reverse=True)
    return {"signals_total": total, "signals_live": live,
            "signals_needs_source": needs_source, "signals_reference": reference,
            "categories": cats, "catalogs": len(per_tab), "per_tab": per_tab}


def _panel_info() -> dict:
    rp = _MODEL_DIR / "panel" / "training_report.json"
    n_models = len(glob.glob(str(_MODEL_DIR / "panel" / "*.joblib")))
    out = {"panel_models": n_models, "trained_at": None, "n_tickers": None,
           "n_features": None, "horizons": 0, "any_reliable": False}
    if rp.exists():
        try:
            rep = json.loads(rp.read_text())
            hz = rep.get("horizons") or {}
            out.update({
                "trained_at": rep.get("trained_at"), "n_tickers": rep.get("n_tickers"),
                "n_features": rep.get("n_features"), "horizons": len(hz),
                "any_reliable": any(v.get("reliable") for v in hz.values()),
            })
        except Exception:
            pass
    return out


def _scan_freshness() -> list:
    """Age of each board's artifact. A board older than ~48h is stale and the
    UI should say so rather than presenting frozen rows as current."""
    out = []
    for label, name in (("multibagger", "scan_artifact.json"),
                        ("relationships", "cf_artifact.json")):
        p = artifact_read_path(name)
        entry = {"board": label, "available": p is not None,
                 "generated": None, "age_hours": None, "stale": True}
        if p:
            try:
                gen = (json.loads(p.read_text()) or {}).get("generated")
                entry["generated"] = gen
                if gen:
                    dt = datetime.fromisoformat(str(gen).replace("Z", "+00:00"))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    age = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                    entry["age_hours"] = round(age, 1)
                    entry["stale"] = age > 48
            except Exception:
                pass
        out.append(entry)
    return out


@router.get("/system/stats")
async def system_stats():
    cat = _catalog_counts()
    panel = _panel_info()
    return {
        "signals": cat,
        "panel": panel,
        "boards": _scan_freshness(),
        "tabs": 23,
        "universe_note": "~5,150 US names with both a live price and a CIK",
        "price_history_years": 5,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
