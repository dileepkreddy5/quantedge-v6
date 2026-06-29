"""Full-universe scan job — the real thing (manual 7.2: scan all, display top N).

Pipeline:
  1. one grouped-daily call -> every ticker's close
  2. EDGAR shares-outstanding -> market cap for each (the ranking cost)
  3. bucket into small/mid/large by params.yaml thresholds, take top N each
  4. score each candidate (Piotroski + growth + quiet price + ladder)
  5. keep top 100 per tier by score -> artifact

Designed as a BACKGROUND job (30-45 min for the full ~5000-name universe).
`max_per_tier` caps the scan for testing; set high for the full run.
"""
from __future__ import annotations
import os, sys, json, time
from datetime import datetime
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import yaml
from quantedge.fundamentals.universe_full import all_closes, ticker_cik_map, shares_outstanding
from quantedge.fundamentals.scanner import scan_one

PARAMS = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "params.yaml")))


def tier_of(mc):
    ct = PARAMS["cap_tiers"]
    if mc >= ct["large_cap_min_usd"]: return "large"
    if mc >= ct["mid_cap_usd"][0]:    return "mid"
    if mc >= ct["small_cap_usd"][0]:  return "small"
    return None


def build_universe(rank_limit=None):
    """Rank real companies by market cap into tiers. rank_limit caps shares-fetch."""
    closes = all_closes()
    cikmap = ticker_cik_map()
    common = sorted(set(closes) & set(cikmap))
    if rank_limit:
        common = common[:rank_limit]
    tiers = {"small": [], "mid": [], "large": []}
    for i, t in enumerate(common):
        sh = shares_outstanding(cikmap[t])
        if not sh:
            continue
        mc = closes[t] * sh
        tier = tier_of(mc)
        if tier:
            tiers[tier].append((t, cikmap[t], mc))
        time.sleep(0.03)
    for k in tiers:
        tiers[k].sort(key=lambda x: x[2], reverse=True)  # by market cap desc
    return tiers


def run_scan(top_universe_per_tier=1000, display_per_tier=100, rank_limit=None):
    tiers = build_universe(rank_limit=rank_limit)
    out = {"generated": datetime.utcnow().isoformat() + "Z", "tiers": {},
           "disclaimer": "Top companies by market cap per tier, scored on quarterly "
                         "growth + quality + quiet price. A SHORTLIST FILTER, not a "
                         "predictor; tail inflections are unpredictable. Not advice."}
    for tier, names in tiers.items():
        universe = names[:top_universe_per_tier]
        scored = []
        for t, cik, mc in universe:
            try:
                r = scan_one(t, cik)
                if r:
                    r["market_cap"] = round(mc / 1e9, 2)
                    scored.append(r)
            except Exception:
                pass
        scored.sort(key=lambda x: x["score"], reverse=True)
        out["tiers"][tier] = scored[:display_per_tier]
        print(f"{tier}: universe {len(universe)}, scored {len(scored)}, kept {min(len(scored),display_per_tier)}")
    path = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "research_data", "scan_artifact.json")
    json.dump(out, open(path, "w"), indent=2)
    print("wrote", path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rank-limit", type=int, default=None, help="cap companies ranked (test)")
    ap.add_argument("--top-universe", type=int, default=1000)
    ap.add_argument("--display", type=int, default=100)
    a = ap.parse_args()
    run_scan(a.top_universe, a.display, a.rank_limit)
