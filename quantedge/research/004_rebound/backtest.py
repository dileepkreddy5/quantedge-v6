"""REBOUND point-in-time backtest (step-21) — the gate decides, not us.

Runs the EXACT nightly-scan code path (price_prefilter + score_one) at
historical quarter-end as_of dates, measures forward 6m/12m returns net of
the committed cost model, and judges the result against the FROZEN
kill-thresholds from params.yaml (step-11).

Honesty notes committed before running:
  CONTROL      prefilter-passers that FAILED the gates — the beaten-down
               base rate. Beating SPY alone is not the claim; beating the
               rest of the beaten-down bin is.
  WINDOW       the price store spans the Polygon plan's 5-year floor
               (~2021-07 onward); with a 3y drawdown lookback the first
               as_of is 2024-09-30. Few dates -> the per-date t-stat will
               be honestly small. If the gate fails on sample size or
               regime coverage, the verdict is INSUFFICIENT DATA — which is
               a data-purchase decision, not a threshold-softening one.
  INSIDERS     the insider component (4/100 pts) runs as None historically
               in v1 (per-candidate Form 4 backfill is a follow-up) — the
               backtest therefore judges a slightly WEAKER score than the
               live scan ships. Conservative direction.
  SURVIVORSHIP the ticker->CIK map is current-day; names delisted before
               today may be under-represented. Passers and control share
               the bias; the SPREAD is the protected statistic. Journaled.
  REGIMES      a date's regime = SPY trailing-6m return: BULL >= +5%,
               BEAR <= -5%, else FLAT. min_distinct_regimes checked on the
               as_of dates actually used.

Run (VPS, background):
  PYTHONPATH=. python3 quantedge/research/004_rebound/backtest.py \
      --price-db /opt/quantedge-research/data/price_store.db \
      --out /opt/quantedge-research/rebound_backtest.json
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Dict, List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import yaml

from quantedge.data.price_store import PriceStore
from quantedge.fundamentals.universe_full import ticker_cik_map
from quantedge.harness.costs import round_trip_cost_bps
from quantedge.fundamentals.rebound.scan import price_prefilter, score_one

_spec = importlib.util.spec_from_file_location(
    "rb_stats", os.path.join(os.path.dirname(__file__), "stats.py"))
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

PARAMS = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "..", "params.yaml")))
KILL = PARAMS["rebound"]["kill_threshold"]


# ── forward returns from the store ────────────────────────────

def fwd_return(store: PriceStore, ticker: str, as_of: date,
               cal_days: int) -> Optional[float]:
    """Entry = first close AFTER as_of; exit = first close on/after
    entry+cal_days. None if either side is missing (not yet matured or
    delisted mid-window — reported, never silently dropped as a win)."""
    bars = store.series(ticker, as_of + timedelta(days=1),
                        as_of + timedelta(days=cal_days + 10))
    if len(bars) < 2:
        return None
    entry_d, entry_p, _ = bars[0]
    target = entry_d + timedelta(days=cal_days)
    exits = [b for b in bars if b[0] >= target]
    if not exits or entry_p <= 0:
        return None
    return exits[0][1] / entry_p - 1.0


def regime_of(store: PriceStore, as_of: date) -> str:
    bars = store.series("SPY", as_of - timedelta(days=190), as_of)
    if len(bars) < 2 or bars[0][1] <= 0:
        return "UNKNOWN"
    r = bars[-1][1] / bars[0][1] - 1.0
    return "BULL" if r >= 0.05 else "BEAR" if r <= -0.05 else "FLAT"


# ── one as_of date ────────────────────────────────────────────

def run_date(store: PriceStore, cikmap: Dict[str, str], as_of: date,
             workers: int, max_control: int) -> List[Dict]:
    tickers = sorted(
        set(store.closes_on(_nearest_trading_day(store, as_of))) & set(cikmap))
    candidates = price_prefilter(store, tickers, as_of)
    print(f"  {as_of}: prefilter {len(candidates)}", flush=True)

    events: List[Dict] = []
    passed_tickers = set()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_one, t, cikmap[t], bars, as_of): t
                for t, bars in candidates}
        for f in as_completed(futs):
            t = futs[f]
            try:
                r = f.result()
            except Exception:
                r = None
            if r:
                passed_tickers.add(t)
                events.append({"as_of": as_of.isoformat(), "ticker": t,
                               "passed": True, "score": r["score"],
                               "stage": r["stage"], "tier": r["tier"]})
    # control group (capped for runtime, deterministic order)
    controls = [t for t, _ in candidates if t not in passed_tickers][:max_control]
    for t in controls:
        events.append({"as_of": as_of.isoformat(), "ticker": t,
                       "passed": False, "score": None,
                       "stage": None, "tier": None})

    spy6 = fwd_return(store, "SPY", as_of, 182)
    spy12 = fwd_return(store, "SPY", as_of, 365)
    for e in events:
        e["ret_6m"] = fwd_return(store, e["ticker"], as_of, 182)
        e["ret_12m"] = fwd_return(store, e["ticker"], as_of, 365)
        e["spy_6m"], e["spy_12m"] = spy6, spy12
    n_pass = sum(1 for e in events if e["passed"])
    print(f"  {as_of}: passers {n_pass}, control {len(controls)}", flush=True)
    return events


def _nearest_trading_day(store: PriceStore, d: date) -> date:
    for back in range(0, 7):
        day = d - timedelta(days=back)
        if store.closes_on(day):
            return day
    return d


# ── gate verdict ──────────────────────────────────────────────

def gate_check(summary6: Dict, summary12: Dict, regimes: List[str]) -> Dict:
    primary = summary12 if (summary12.get("n_dates") or 0) >= 3 else summary6
    checks = {
        "lift_>=_{}".format(KILL["min_oos_lift_over_base_rate"]):
            (primary.get("lift") or 0) >= KILL["min_oos_lift_over_base_rate"],
        "t_stat_>=_{}".format(KILL["min_t_stat"]):
            (primary.get("t_stat") or 0) >= KILL["min_t_stat"],
        "distinct_regimes_>=_{}".format(KILL["min_distinct_regimes"]):
            len(set(r for r in regimes if r != "UNKNOWN")) >= KILL["min_distinct_regimes"],
        "min_3_dates": (primary.get("n_dates") or 0) >= 3,
    }
    verdict = "PASS" if all(checks.values()) else (
        "INSUFFICIENT_DATA" if not checks["min_3_dates"]
        or not checks[f"distinct_regimes_>=_{KILL['min_distinct_regimes']}"]
        else "FAIL")
    return {"primary_horizon": primary.get("horizon"),
            "checks": checks, "verdict": verdict}


def subset(events: List[Dict], key: str, value) -> List[Dict]:
    return [e for e in events if e.get(key) == value or not e["passed"]]


# ── main ──────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-control", type=int, default=400)
    args = ap.parse_args()

    t0 = time.time()
    store = PriceStore(args.price_db)
    cov = store.coverage()
    print("store coverage:", cov, flush=True)
    first_bar = date.fromisoformat(cov["from"])
    last_bar = date.fromisoformat(cov["to"])

    # quarter-ends with full 3y lookback and >=6m of forward data
    as_ofs, d = [], date(first_bar.year + 3, 3, 31)
    while d <= last_bar - timedelta(days=182):
        if d >= first_bar + timedelta(days=3 * 365):
            as_ofs.append(d)
        d = _next_qend(d)
    print("as_of dates:", [x.isoformat() for x in as_ofs], flush=True)

    cikmap = ticker_cik_map()
    all_events: List[Dict] = []
    regimes: List[str] = []
    for a in as_ofs:
        regimes.append(regime_of(store, a))
        all_events += run_date(store, cikmap, a, args.workers, args.max_control)

    rt = round_trip_cost_bps(PARAMS) / 1e4
    s6 = st.summarize(all_events, "6m", rt)
    s12 = st.summarize(all_events, "12m", rt)
    gate = gate_check(s6, s12, regimes)

    per_stage = {stg: st.summarize(subset(all_events, "stage", stg), "6m", rt)
                 for stg in ("FALLING", "BASING", "TURNING", "RECOVERING")}
    per_tier = {tr: st.summarize(subset(all_events, "tier", tr), "6m", rt)
                for tr in ("small", "mid", "large")}

    out = {
        "unit": "004_rebound", "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "store_coverage": cov, "as_of_dates": [x.isoformat() for x in as_ofs],
        "regimes": regimes, "roundtrip_cost": rt,
        "summary_6m": s6, "summary_12m": s12,
        "per_stage_6m": per_stage, "per_tier_6m": per_tier,
        "gate": gate, "kill_thresholds": KILL,
        "caveats": ["insider component scored as None (conservative)",
                    "current-day CIK map: survivorship shared by both groups",
                    f"window limited to plan depth ({cov['from']}..)"],
        "events": all_events,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1, default=str)

    print(f"\n===== REBOUND BACKTEST ({time.time()-t0:.0f}s) =====")
    print("regimes:", regimes)
    for name, s in (("6m", s6), ("12m", s12)):
        print(f"{name}: dates={s['n_dates']} passers={s['n_pass_events']} "
              f"spread={s['mean_spread_per_date']} t={s['t_stat']} "
              f"lift={s['lift']} win={s['passer_win_rate']} vs {s['control_win_rate']}")
    print("per-stage 6m:", {k: (v["mean_spread_per_date"], v["n_pass_events"])
                            for k, v in per_stage.items()})
    print("GATE:", gate["verdict"], gate["checks"])
    print(f"artifact -> {args.out}")


def _next_qend(d: date) -> date:
    y, m = (d.year + 1, 3) if d.month == 12 else (d.year, d.month + 3)
    if m in (3, 12):
        return date(y, m, 31)
    return date(y, m, 30)


if __name__ == "__main__":
    main()
