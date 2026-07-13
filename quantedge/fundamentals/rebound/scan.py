"""REBOUND full-universe scan (step-20) — the assembly line.

Nightly pipeline (also callable with --sample for smoke tests):

  1. UNIVERSE      one Polygon grouped call -> every US common stock + CIK map
  2. PRICE FILTER  price-store pass over ALL tickers (local SQLite, seconds):
                   keep drawdown >= bar, price >= $2, ADV$ >= liquidity floor.
                   ~5,000 -> hundreds without touching the bulk zip.
  3. FUNDAMENTALS  per survivor, from the LOCAL companyfacts.zip (the one the
                   02:00 multibagger job refreshes): discount valuation,
                   health, knife filter, buybacks. Threaded, zero API calls.
  4. INSIDERS      only for stocks that passed ALL gates (typically dozens,
                   not thousands): SEC submissions fetch, SQLite-cached,
                   paced at 8 req/s.
  5. RANK          tier by live market cap (close x latest knowable shares),
                   top N per tier by score, thesis lines assembled.
  6. ARTIFACT      research_data/rebound_artifact.json — the API serves this.

PIT note: run with as_of=today for the nightly scan; the step-21 backtest
calls score_one() with historical as_of dates and a truncated price store —
same code path, zero look-ahead either way.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import yaml

from quantedge.data.price_store import PriceStore
from quantedge.fundamentals.universe_full import all_closes, ticker_cik_map
from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
from quantedge.fundamentals.bulk_adapter import pit_from_bulk, quarterly_revenue_from_bulk
from quantedge.fundamentals.edgar_pit import knowable_as_of
from quantedge.fundamentals.rebound import bulk_extra as bx
from quantedge.fundamentals.rebound.discount import compute_discount, drawdown_structure
from quantedge.fundamentals.rebound.health import compute_health
from quantedge.fundamentals.rebound.disqualify import compute_disqualifiers
from quantedge.fundamentals.rebound.confirm import compute_confirm, volume_signals
from quantedge.fundamentals.rebound.stage import classify_stage
from quantedge.fundamentals.rebound.rebound_score import score_candidate
from quantedge.fundamentals.rebound.insider_form4 import InsiderStore, opportunistic_cluster

PARAMS = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "..", "..", "params.yaml")))
ARTIFACT_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                             "backend", "research_data", "rebound_artifact.json")


def tier_of(mc: float) -> Optional[str]:
    ct = PARAMS["cap_tiers"]
    if mc >= ct["large_cap_min_usd"]:
        return "large"
    if mc >= ct["mid_cap_usd"][0]:
        return "mid"
    if mc >= ct["small_cap_usd"][0]:
        return "small"
    return None


# ── stage 2: price-structure prefilter (local store only) ─────

def price_prefilter(store: PriceStore, tickers: List[str], as_of: date) -> List[Tuple[str, list]]:
    """Returns [(ticker, bars_3y)] for names clearing drawdown/price/liquidity
    bars using ONLY the local price store — no bulk zip, no network."""
    p = PARAMS["rebound"]["universe"]
    start = as_of - timedelta(days=3 * 366)
    out = []
    for t in tickers:
        bars = store.series(t, start, as_of)
        if len(bars) < 200:
            continue
        closes = [(d, c) for d, c, _ in bars]
        dd = drawdown_structure(closes, as_of,
                                PARAMS["rebound"]["discount"]["lookback_high_years"])
        if not dd["ok"]:
            continue
        if dd["drawdown_from_3y_high"] < p["min_drawdown_from_3y_high"]:
            continue
        if dd["price_now"] < p["min_price_usd"]:
            continue
        recent = bars[-60:]
        adv = sum(c * v for _, c, v in recent) / max(len(recent), 1)
        if adv < p["liquidity_floor_adv_usd"]:
            continue
        out.append((t, bars))
    return out


# ── stage 3: full scoring for one candidate ───────────────────

def score_one(ticker: str, cik: str, bars: list, as_of: date,
              insider: Optional[Dict] = None) -> Optional[Dict]:
    facts = company_facts_from_bulk(cik)
    if not facts:
        return None
    closes = [(d, c) for d, c, _ in bars]

    q_rev = bx.quarterly_revenue_complete(facts)   # Q4-synthesized (ADBE bug fix)
    q_gp = bx.quarterly_gross_profit(facts)
    shares = bx.quarterly_shares(facts)
    cash = bx.cash_series(facts)
    debt = bx.debt_series(facts)
    rd = bx.rd_series(facts)
    buybacks = bx.buyback_series(facts)
    known = knowable_as_of(pit_from_bulk(facts), as_of)

    sh_now = bx.latest_knowable(shares, as_of)
    mktcap = closes[-1][1] * sh_now[1] if sh_now and sh_now[1] > 0 else None

    disc = compute_discount(closes, q_rev, shares, cash, debt, as_of, PARAMS)
    heal = compute_health(q_rev, q_gp, rd, known, mktcap, ticker, as_of, PARAMS)
    disq = compute_disqualifiers(q_rev, shares, cash, known, as_of, PARAMS)
    conf = compute_confirm(bars, buybacks, mktcap, as_of, PARAMS)
    stg = classify_stage(disc.get("drawdown", {}), conf.get("volume", {}), PARAMS)

    verdict = score_candidate(disc, heal, disq, conf, insider, stg, PARAMS)
    if not verdict["passes"]:
        return None
    return {
        "ticker": ticker, "cik": cik,
        "price": disc["drawdown"]["price_now"],
        "market_cap": round(mktcap / 1e9, 3) if mktcap else None,
        "tier": tier_of(mktcap) if mktcap else None,
        "score": verdict["score"],
        "stage": verdict["stage"],
        "stage_reason": verdict["stage_reason"],
        "thesis": verdict["thesis"],
        "drawdown": disc["drawdown"]["drawdown_from_3y_high"],
        "days_since_low": disc["drawdown"]["days_since_1y_low"],
        "growth_streak": heal["growth"].get("streak"),
        "piotroski": heal["piotroski"],
        "vol_1w_ratio": conf["volume"].get("vol_1w_ratio"),
        "vol_1m_ratio": conf["volume"].get("vol_1m_ratio"),
        "up_day_share_1m": conf["volume"].get("up_day_share_1m"),
        "insider_cluster": bool(insider and insider.get("cluster")),
        "insider_summary": insider.get("reason") if insider else None,
        "components": verdict["components"],
        "unverified_checks": disq.get("unverified", []),
    }


# ── the scan ──────────────────────────────────────────────────

def run_scan(price_db: str, insider_db: str, as_of: Optional[date] = None,
             sample: Optional[int] = None, skip_insiders: bool = False,
             workers: int = 8, display: Optional[int] = None) -> Dict:
    t0 = time.time()
    display = display or PARAMS["rebound"]["display_n_per_tier"]

    store = PriceStore(price_db)
    if as_of is None:
        # weekend/holiday runs must anchor to the last TRADING day — pricing
        # "today" on a Sunday otherwise fails for every stock in the universe
        as_of = store.last_day() or date.today()
    print(f"as_of: {as_of}", flush=True)

    closes = all_closes()
    cikmap = ticker_cik_map()
    tickers = sorted(set(closes) & set(cikmap))
    if sample:
        tickers = tickers[:sample]
    print(f"universe: {len(tickers)} tickers with CIK", flush=True)
    candidates = price_prefilter(store, tickers, as_of)
    print(f"price prefilter: {len(candidates)} clear drawdown/price/liquidity "
          f"({time.time()-t0:.0f}s)", flush=True)

    passed: List[Dict] = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(score_one, t, cikmap[t], bars, as_of): t
                for t, bars in candidates}
        for f in as_completed(futs):
            try:
                r = f.result()
                if r:
                    passed.append(r)
            except Exception as e:
                print(f"  {futs[f]} error: {e}", flush=True)
    print(f"gates passed: {len(passed)} ({time.time()-t0:.0f}s)", flush=True)

    # insiders — only for gate-passers (the expensive network step stays small)
    if not skip_insiders and passed:
        istore = InsiderStore(insider_db)
        win = PARAMS["rebound"]["confirm"]["insider_cluster_window_days"]
        since = as_of - timedelta(days=win + 10)
        for r in passed:
            try:
                istore.collect_for_issuer(r["cik"], since, as_of)
                buys = istore.buys_for(r["cik"], since, as_of)
                cluster = opportunistic_cluster(buys, as_of, PARAMS)
                if cluster["cluster"]:
                    r["insider_cluster"] = True
                    r["insider_summary"] = cluster["reason"]
                    r["score"] = round(
                        r["score"] + PARAMS["rebound"]["scoring"]["insider"], 1)
                    r["thesis"] += " · " + cluster["reason"]
            except Exception as e:
                print(f"  insider {r['ticker']}: {e}", flush=True)
        istore.close()
        print(f"insider pass done ({time.time()-t0:.0f}s)", flush=True)

    tiers: Dict[str, List[Dict]] = {"small": [], "mid": [], "large": []}
    for r in passed:
        if r["tier"] in tiers:
            tiers[r["tier"]].append(r)
    for k in tiers:
        tiers[k] = sorted(tiers[k], key=lambda x: x["score"], reverse=True)[:display]

    artifact = {
        "generated": datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "as_of": as_of.isoformat(),
        "n_universe": len(tickers),
        "n_prefilter": len(candidates),
        "n_passed_gates": len(passed),
        "tiers": tiers,
        "stage_counts": {s: sum(1 for r in passed if r["stage"] == s)
                         for s in ("FALLING", "BASING", "TURNING", "RECOVERING")},
        "disclaimer": ("Discounted-quality rebound candidates: deep drawdown x "
                       "improving fundamentals x confirmation, from SEC filings "
                       "and market data. A research shortlist, not a prediction "
                       "and not investment advice. Validation status: pending "
                       "step-21 backtest through the promotion gate."),
        "duration_seconds": round(time.time() - t0, 1),
    }
    store.close()
    return artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-db", default=os.environ.get(
        "PRICE_STORE_PATH", "data/price_store.db"))
    ap.add_argument("--insider-db", default=os.environ.get(
        "INSIDER_STORE_PATH", "data/insider_store.db"))
    ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--skip-insiders", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=ARTIFACT_PATH)
    args = ap.parse_args()

    artifact = run_scan(args.price_db, args.insider_db,
                        sample=args.sample, skip_insiders=args.skip_insiders,
                        workers=args.workers)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(artifact, fh, indent=1, default=str)
    print(f"artifact -> {args.out}")
    for tier in ("large", "mid", "small"):
        rows = artifact["tiers"][tier]
        print(f"\n== {tier.upper()} ({len(rows)}) ==")
        for r in rows[:5]:
            print(f"  {r['ticker']:<6} {r['score']:>5}  {r['stage']:<10} {r['thesis'][:110]}")


if __name__ == "__main__":
    main()
