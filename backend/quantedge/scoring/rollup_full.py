"""Full rollup over the Financial Statement catalog. Scores every signal against
the feature dict + peer distributions, rolls leaves->categories->intelligence,
carrying confidence (share of weight actually computed) at each level. 'defined'
signals with no computed value score None and lower confidence — never faked.
"""
from __future__ import annotations
from typing import Dict, Any, List, Optional
from quantedge.scoring.compute import score_signal
from quantedge.scoring.catalog_financial import CATEGORIES


def _roll(children):
    scored = [c for c in children if c.get("score") is not None]
    tw = sum(c["weight"] for c in children)
    aw = sum(c["weight"] for c in scored)
    if not scored or tw == 0:
        return None, 0.0
    return round(sum(c["score"] * c["weight"] for c in scored) / aw, 1), round(aw / tw, 3)


def run_financial(features: Dict[str, float],
                  peers: Optional[Dict[str, List[float]]] = None) -> Dict[str, Any]:
    peers = peers or {}
    cats_out = []
    for cid, (label, wt, sigs) in CATEGORIES.items():
        scored = []
        for spec in sigs:
            val = features.get(spec["field"])
            pv = peers.get(spec.get("peer_key")) if spec.get("peer_key") else None
            res = score_signal(val, spec, pv)
            scored.append({"id": spec["id"], "label": spec["label"], "weight": spec["weight"],
                           "status": spec["status"], "evidence": spec["evidence"], **res})
        cscore, cconf = _roll(scored)
        cats_out.append({"id": cid, "label": label, "weight": wt, "score": cscore,
                         "confidence": cconf, "n_signals": len(sigs),
                         "n_live": sum(1 for s in sigs if s["status"] == "live"),
                         "n_scored": sum(1 for s in scored if s["score"] is not None),
                         "signals": scored})
    iscore, iconf = _roll(cats_out)
    return {"label": "Financial Statement Intelligence", "score": iscore,
            "confidence": iconf, "categories": cats_out}
