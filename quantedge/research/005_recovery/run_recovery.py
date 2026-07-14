"""Deep recovery-to-high backtest (unit 005) — the user's thesis, done rigorously.

Measures, for beaten-down stocks, whether GOOD FINANCIALS predict recovery to
the prior high — the exact event the thesis is about, with an open deadline.

Rigor built in so it cannot flatter itself:
  SURVIVORSHIP  a name that stops trading before recovering is counted a FAILURE
  MAGNITUDE     max fraction of drawdown recovered + return earned vs SPY
  BUCKETS       results split by drawdown depth (35-50, 50-70, 70+%)
  CONTROL       beaten-down + WEAK financials — isolates health as the driver
  PIT           entry uses only data <= as_of; outcome scans only AFTER
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import date, timedelta
from statistics import median
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import yaml
import importlib.util

from quantedge.data.price_store import PriceStore
from quantedge.fundamentals.universe_full import ticker_cik_map
from quantedge.fundamentals.rebound import bulk_extra as bx
from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
from quantedge.fundamentals.bulk_adapter import pit_from_bulk
from quantedge.fundamentals.edgar_pit import knowable_as_of
from quantedge.fundamentals.rebound.health import growth_streak
from quantedge.fundamentals.multibagger_score import score as piotroski_score

_spec = importlib.util.spec_from_file_location(
    "rec", os.path.join(os.path.dirname(__file__), "recovery.py"))
rec = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(rec)

P = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "params_recovery.yaml")))["recovery"]


def _quarter_ends(first: date, last: date) -> List[date]:
    out = []
    for y in range(first.year, last.year + 1):
        for m, d in ((3,31),(6,30),(9,30),(12,31)):
            qd = date(y, m, d)
            if first <= qd <= last:
                out.append(qd)
    return out


def _healthy(fund_db, cikmap, ticker, as_of) -> Optional[bool]:
    cik = cikmap.get(ticker)
    if not cik:
        return None
    try:
        facts = company_facts_from_bulk(cik)
        if not facts:
            return None
        q_rev = bx.quarterly_revenue_complete(facts)
        known = knowable_as_of(pit_from_bulk(facts), as_of)
        gs = growth_streak(q_rev, as_of, 25_000_000)
        pio = piotroski_score(ticker, known).piotroski
        if not gs.get("ok"):
            return None
        return gs["streak"] >= 4 and pio >= 5
    except Exception:
        return None


def _bucket(dd: float) -> str:
    if dd < 0.50: return "35-50"
    if dd < 0.70: return "50-70"
    return "70+"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-per-quarter", type=int, default=400)
    args = ap.parse_args()

    t0 = time.time()
    store = PriceStore(args.price_db)
    cikmap = ticker_cik_map()
    cov = store.coverage()
    first = date.fromisoformat(cov["from"]); last = date.fromisoformat(cov["to"])
    entry_qends = [q for q in _quarter_ends(first + timedelta(days=1100), last)
                   if q <= last - timedelta(days=P["tracking_days"])]
    print(f"entry quarters: {[q.isoformat() for q in entry_qends]}", flush=True)

    events = []
    for q in entry_qends:
        day = q
        for _ in range(7):
            if store.closes_on(day): break
            day -= timedelta(days=1)
        universe = sorted(set(store.closes_on(day)) & set(cikmap))
        scored = 0
        for t in universe:
            if scored >= args.max_per_quarter:
                break
            bars_raw = store.series(t, q - timedelta(days=1200),
                                    q + timedelta(days=P["tracking_days"] + 20))
            hist = [(d, c) for d, c, _ in bars_raw if d <= q]
            if len(hist) < 250:
                continue
            dd = rec.drawdown_at(hist, q, P["lookback_days"])
            if dd is None or dd < P["min_drawdown"]:
                continue
            healthy = _healthy(None, cikmap, t, q)
            if healthy is None:
                continue
            scored += 1
            hi = rec.prior_high(hist, q, P["lookback_days"])
            entry = hist[-1][1]
            target = entry + P["recover_frac"] * (hi - entry)
            allbars = [(d, c) for d, c, _ in bars_raw]
            outcome = rec.recovery_outcome(allbars, q, target, P["tracking_days"])
            if outcome.get("matured") and not outcome.get("recovered"):
                tail_bars = [d for d, c in allbars
                             if q + timedelta(days=P["tracking_days"]-30) < d]
                if not tail_bars:
                    outcome["delisted"] = True
            spy_bars = store.series("SPY", q, q + timedelta(days=P["tracking_days"]+5))
            spy_ret = (spy_bars[-1][1]/spy_bars[0][1]-1) if len(spy_bars) >= 2 else None
            fwd = [c for d, c, _ in bars_raw if d > q]
            stock_ret = (max(fwd)/entry - 1) if fwd else None
            events.append({"as_of": q.isoformat(), "ticker": t, "healthy": healthy,
                           "drawdown": round(dd,4), "bucket": _bucket(dd),
                           "outcome": outcome,
                           "peak_ret_vs_spy": (round(stock_ret - spy_ret,4)
                                               if stock_ret is not None and spy_ret is not None else None)})
        print(f"  {q}: {scored} scored ({time.time()-t0:.0f}s)", flush=True)

    overall = rec.evaluate(events, {"recovery": P["kill_threshold"]})
    buckets = {}
    for b in ("35-50", "50-70", "70+"):
        sub = [e for e in events if e["bucket"] == b]
        buckets[b] = rec.evaluate(sub, {"recovery": P["kill_threshold"]})

    out = {"unit": "005_recovery", "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "params": P, "n_events": len(events),
           "overall": overall, "by_drawdown_bucket": buckets}
    json.dump(out, open(args.out, "w"), indent=1, default=str)

    print(f"\n===== RECOVERY-TO-HIGH ({time.time()-t0:.0f}s) =====")
    h, c = overall.get("healthy"), overall.get("control")
    if h and c:
        print(f"HEALTHY : n={h['n']} recover {h['hit_rate']:.1%} median {h['median_days_to_recover']}d")
        print(f"CONTROL : n={c['n']} recover {c['hit_rate']:.1%} median {c['median_days_to_recover']}d")
        print(f"edge {overall['hit_rate_edge']:+.1%}  z {overall['z_stat']}  -> {overall['verdict']}")
    print("by drawdown bucket:")
    for b, r in buckets.items():
        if r.get("healthy") and r.get("control"):
            print(f"  {b}%: healthy {r['healthy']['hit_rate']:.1%} (n{r['healthy']['n']}) "
                  f"vs control {r['control']['hit_rate']:.1%} (n{r['control']['n']}) "
                  f"edge {r.get('hit_rate_edge',0):+.1%} z {r.get('z_stat',0)}")
    print(f"artifact -> {args.out}")


if __name__ == "__main__":
    main()
