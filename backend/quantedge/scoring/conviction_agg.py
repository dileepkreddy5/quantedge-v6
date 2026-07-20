"""Conviction Aggregator — unifies all intelligence modules into ONE consolidated
conviction score. Extensible registry: flip status to 'live' to auto-join the blend.
Normalized over live weight. Deterministic — no LLM in the path.
"""
from __future__ import annotations
import asyncio
from typing import Optional, Dict, List, Any, Callable

MODULE_REGISTRY = [
    {"id":"financial",     "label":"Financial Intelligence",     "weight":18, "status":"live"},
    {"id":"business",      "label":"Business Intelligence",      "weight":12, "status":"live"},
    {"id":"valuation",     "label":"Valuation Intelligence",     "weight":10, "status":"live"},
    {"id":"news",          "label":"News Intelligence",          "weight":10, "status":"live"},
    {"id":"competitive",   "label":"Competitive Intelligence",   "weight":8,  "status":"live"},
    {"id":"management",    "label":"Management Intelligence",    "weight":6,  "status":"live"},
    {"id":"industry",      "label":"Industry Intelligence",      "weight":6,  "status":"live"},
    {"id":"risk",          "label":"Risk Intelligence",          "weight":6,  "status":"live"},
    {"id":"market",        "label":"Market Intelligence",        "weight":5,  "status":"live"},
    {"id":"ownership",     "label":"Ownership Intelligence",     "weight":4,  "status":"pending"},
    {"id":"forecast",      "label":"Forecast Intelligence",      "weight":4,  "status":"pending"},
    {"id":"macro",         "label":"Macroeconomic Intelligence", "weight":3,  "status":"pending"},
    {"id":"alt_data",      "label":"Alternative Data Intelligence","weight":3,"status":"pending"},
    {"id":"institutional", "label":"Institutional Flow Intelligence","weight":2,"status":"pending"},
    {"id":"peers",         "label":"Peers Intelligence",         "weight":2,  "status":"pending"},
    {"id":"ml_models",     "label":"ML Models Intelligence",     "weight":1,  "status":"pending"},
]

def verdict(score: Optional[float]) -> str:
    if score is None: return "NO_DATA"
    if score >= 80: return "STRONG_BUY"
    if score >= 65: return "BUY"
    if score >= 45: return "NEUTRAL"
    if score >= 30: return "SELL"
    return "STRONG_SELL"

async def aggregate_conviction(ticker: str, scorers: Dict[str, Callable]) -> Dict[str, Any]:
    live_mods = [m for m in MODULE_REGISTRY if m["status"]=="live" and m["id"] in scorers]
    async def run(mod):
        try:
            res = await scorers[mod["id"]](ticker)
            return mod, res
        except Exception as e:
            return mod, {"error": str(e)}
    results = await asyncio.gather(*[run(m) for m in live_mods]) if live_mods else []
    modules_out = []; weighted_sum = 0.0; live_weight = 0.0
    for m in MODULE_REGISTRY:
        entry = {"id":m["id"],"label":m["label"],"weight":m["weight"],"status":m["status"],
                 "score":None,"confidence":None}
        for mod, res in results:
            if mod["id"]==m["id"] and res and "score" in res and res["score"] is not None:
                entry["score"]=res["score"]; entry["confidence"]=res.get("confidence")
                entry["coverage"]=res.get("coverage")
                weighted_sum += res["score"]*m["weight"]; live_weight += m["weight"]
        modules_out.append(entry)
    consolidated = round(weighted_sum/live_weight,1) if live_weight>0 else None
    return {"ticker":ticker,"conviction_score":consolidated,"verdict":verdict(consolidated),
            "coverage":{"live_weight":live_weight,"total_weight":100,"pct":round(live_weight/100,3),
                        "modules_live":len([1 for m in modules_out if m['score'] is not None]),
                        "modules_total":len(MODULE_REGISTRY)},
            "modules":modules_out}
