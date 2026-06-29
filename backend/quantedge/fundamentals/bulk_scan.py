"""Bulk-backed full-universe scan — reads fundamentals from local zip, no 429s.

Market caps still need a live price (grouped-daily, 1 call) and shares (from the
bulk facts — no API!). Fundamentals all come from the local companyfacts.zip.
Volume/price ladder use Polygon aggs (per-ticker, but only for the top N scored,
and rate-limited gently). This scans thousands without EDGAR throttling.
"""
from __future__ import annotations
import os, sys, json, time, threading
from datetime import datetime, date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import yaml

from quantedge.fundamentals.universe_full import all_closes, ticker_cik_map
from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
from quantedge.fundamentals.bulk_adapter import pit_from_bulk, quarterly_revenue_from_bulk
from quantedge.fundamentals.growth_clean import clean_growth_signal
from quantedge.fundamentals.edgar_pit import knowable_as_of
from quantedge.fundamentals.multibagger_score import score
from quantedge.fundamentals.extra_signals import gross_margin_trend, accruals, debt_trend
from quantedge.fundamentals.child_companies import beneficiaries

PARAMS = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "params.yaml")))

def tier_of(mc):
    ct = PARAMS["cap_tiers"]
    if mc >= ct["large_cap_min_usd"]: return "large"
    if mc >= ct["mid_cap_usd"][0]:    return "mid"
    if mc >= ct["small_cap_usd"][0]:  return "small"
    return None

def _latest_shares(pit):
    s = pit.get("shares", [])
    return s[-1][1] if s else None

def score_company_bulk(ticker, cik, as_of):
    """Score one company entirely from bulk facts — NO EDGAR API call."""
    facts = company_facts_from_bulk(cik)
    if not facts: return None, None
    pit = pit_from_bulk(facts)
    known = knowable_as_of(pit, as_of)
    # quarterly growth from bulk
    q = quarterly_revenue_from_bulk(facts)
    growth, base_ok = clean_growth_signal(q, as_of)
    if growth is None: return None, _latest_shares(pit)
    r = score(ticker, known)
    comp = min(max(growth,0),2.0)*50 + (r.piotroski/9)*15
    mt, ac, dt = gross_margin_trend(known), accruals(known), debt_trend(known)
    if mt and mt>0: comp += 8
    if ac and ac<0: comp += 6
    if dt and dt>0.05: comp -= 8
    comp = round(max(comp,0),1)
    return {"ticker":ticker,"score":comp,"qtr_yoy_growth":round(growth,3),
            "piotroski":r.piotroski,"margin_trend":mt,"accruals":ac,"debt_trend":dt,
            "beneficiaries":beneficiaries(ticker)}, _latest_shares(pit)

def run_bulk_scan(top_universe=1000, display=100, rank_pool=4000, workers=12):
    t0=time.time()
    closes=all_closes(); cikmap=ticker_cik_map()
    common=sorted(set(closes)&set(cikmap))[:rank_pool]
    print(f"scoring {len(common)} companies from bulk facts (no EDGAR API)…",flush=True)
    # ONE pass: score + market cap (shares from bulk) — all local, parallel safe
    tiers={"small":[],"mid":[],"large":[]}
    def work(t):
        rec, sh = score_company_bulk(t, cikmap[t], date.today())
        if sh and closes.get(t):
            mc = closes[t]*sh; tier=tier_of(mc)
            if tier and rec:
                rec["market_cap"]=round(mc/1e9,2)
                return tier, rec, mc
        return None
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for f in as_completed([ex.submit(work,t) for t in common]):
            res=f.result()
            if res: tiers[res[0]].append((res[1],res[2]))
    out={"generated":datetime.utcnow().isoformat()+"Z","tiers":{},
         "disclaimer":"Top companies by market cap per tier, scored on quarterly growth + quality. Shortlist filter, not a predictor. Not advice."}
    for tier in tiers:
        ranked=sorted(tiers[tier],key=lambda x:x[1],reverse=True)[:top_universe]  # by mkt cap
        by_score=sorted([r for r,_ in ranked],key=lambda x:x["score"],reverse=True)[:display]
        out["tiers"][tier]=by_score
        print(f"{tier}: {len(tiers[tier])} in tier, kept top {len(by_score)} by score ({time.time()-t0:.0f}s)",flush=True)
    path=os.path.join(os.path.dirname(__file__),"..","..","backend","research_data","scan_artifact.json")
    json.dump(out,open(path,"w"),indent=2)
    print(f"DONE in {time.time()-t0:.0f}s — wrote {path}",flush=True)

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("--top-universe",type=int,default=1000)
    ap.add_argument("--display",type=int,default=100)
    ap.add_argument("--rank-pool",type=int,default=4000)
    ap.add_argument("--workers",type=int,default=12)
    a=ap.parse_args()
    run_bulk_scan(a.top_universe,a.display,a.rank_pool,a.workers)
