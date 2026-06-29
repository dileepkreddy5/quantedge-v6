"""Parallel full-universe scan — 3000 companies tractably via concurrency.

The serial scan was slow because it fetched one company at a time. This fetches
many concurrently with a thread pool, respecting EDGAR's ~10 req/sec limit via
a semaphore. 3000 companies becomes minutes, not hours. Displays top N per tier.
"""
from __future__ import annotations
import os, sys, json, time, threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import yaml
from quantedge.fundamentals.universe_full import all_closes, ticker_cik_map, shares_outstanding
from quantedge.fundamentals.scanner import scan_one

PARAMS = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "params.yaml")))

# rate limiter: cap concurrent EDGAR hits to stay under ~10/sec
_sem = threading.Semaphore(8)

def tier_of(mc):
    ct = PARAMS["cap_tiers"]
    if mc >= ct["large_cap_min_usd"]: return "large"
    if mc >= ct["mid_cap_usd"][0]:    return "mid"
    if mc >= ct["small_cap_usd"][0]:  return "small"
    return None

def _shares_safe(t, cik):
    with _sem:
        try: return t, cik, shares_outstanding(cik)
        except Exception: return t, cik, None

def _scan_safe(t, cik):
    with _sem:
        try: return scan_one(t, cik)
        except Exception: return None

def run_parallel_scan(top_universe=1000, display=100, rank_pool=3500, workers=16):
    t0 = time.time()
    closes = all_closes()
    cikmap = ticker_cik_map()
    common = sorted(set(closes) & set(cikmap))[:rank_pool]
    print(f"ranking {len(common)} companies by market cap (parallel)…", flush=True)

    # PHASE 1: market caps in parallel
    tiers = {"small": [], "mid": [], "large": []}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_shares_safe, t, cikmap[t]) for t in common]
        for f in as_completed(futs):
            t, cik, sh = f.result()
            if not sh: continue
            mc = closes[t] * sh
            tier = tier_of(mc)
            if tier: tiers[tier].append((t, cik, mc))
    for k in tiers: tiers[k].sort(key=lambda x: x[2], reverse=True)
    print(f"ranked in {time.time()-t0:.0f}s — small {len(tiers['small'])}, mid {len(tiers['mid'])}, large {len(tiers['large'])}", flush=True)

    # PHASE 2: score top N per tier in parallel
    out = {"generated": datetime.utcnow().isoformat()+"Z", "tiers": {},
           "disclaimer": "Top companies by market cap per tier, scored on quarterly "
                         "growth + quality + quiet price. A SHORTLIST FILTER, not a "
                         "predictor. Not advice."}
    for tier, names in tiers.items():
        universe = names[:top_universe]
        scored = []
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_scan_safe, t, cik): (t, mc) for t, cik, mc in universe}
            for f in as_completed(futs):
                r = f.result()
                if r:
                    _, mc = futs[f]
                    r["market_cap"] = round(mc/1e9, 2)
                    scored.append(r)
        scored.sort(key=lambda x: x["score"], reverse=True)
        out["tiers"][tier] = scored[:display]
        print(f"{tier}: scored {len(scored)}/{len(universe)}, kept {min(len(scored),display)} ({time.time()-t0:.0f}s elapsed)", flush=True)

    path = os.path.join(os.path.dirname(__file__), "..", "..", "backend", "research_data", "scan_artifact.json")
    json.dump(out, open(path, "w"), indent=2)
    print(f"DONE in {time.time()-t0:.0f}s — wrote {path}", flush=True)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-universe", type=int, default=1000)
    ap.add_argument("--display", type=int, default=100)
    ap.add_argument("--rank-pool", type=int, default=3500)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    run_parallel_scan(a.top_universe, a.display, a.rank_pool, a.workers)
