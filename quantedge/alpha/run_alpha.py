"""Cross-sectional alpha — end-to-end driver (Stage F).

Builds the monthly panel from the REAL price store, runs walk-forward
validation, judges rank-IC against the frozen gate, writes an artifact with
the honest verdict AND the feature-importance ranking (what actually drove
cross-sectional returns on your data — valuable regardless of ship/no-ship).

VPS:
  PYTHONPATH=. python3 quantedge/alpha/run_alpha.py \
    --price-db /opt/quantedge-research/data/price_store.db \
    --out /opt/quantedge-research/alpha_xs.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import yaml

from quantedge.data.price_store import PriceStore
from quantedge.fundamentals.universe_full import ticker_cik_map
from quantedge.alpha.panel import build_panel_for_date, month_ends
from quantedge.alpha.model import walk_forward

CFG = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "params_alpha.yaml")))["alpha_xs"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default=None)
    ap.add_argument("--sample-universe", type=int, default=None,
                    help="cap tickers for a fast smoke run")
    args = ap.parse_args()

    t0 = time.time()
    store = PriceStore(args.price_db)
    cov = store.coverage()
    print("store coverage:", cov, flush=True)
    first = date.fromisoformat(cov["from"])
    last = date.fromisoformat(cov["to"])

    start = date.fromisoformat(args.start) if args.start else first + timedelta(days=400)
    end = last - timedelta(days=int(CFG["fwd_nbars"] * 1.6) + 15)
    dates = month_ends(start, end)
    print(f"as_of months: {len(dates)} ({dates[0]} .. {dates[-1]})", flush=True)

    cikmap = ticker_cik_map()
    if args.sample_universe:
        cikmap = dict(list(cikmap.items())[:args.sample_universe])

    panel = {}
    for i, d in enumerate(dates):
        rows = build_panel_for_date(store, cikmap, d,
                                    CFG["fwd_nbars"], CFG["min_dollar_adv"])
        panel[d.isoformat()] = rows
        if i % 3 == 0:
            print(f"  {d}: {len(rows)} rows ({time.time()-t0:.0f}s)", flush=True)

    res = walk_forward(panel, CFG["label"], CFG["min_train_months"])

    K = CFG["kill_threshold"]
    checks = {
        f"mean_ic>={K['min_mean_rank_ic']}":
            (res["mean_rank_ic"] or 0) >= K["min_mean_rank_ic"],
        f"ic_t>={K['min_ic_t_stat']}":
            (res["ic_t_stat"] or 0) >= K["min_ic_t_stat"],
        f"pos_months>={K['min_positive_month_frac']}":
            (res["positive_month_frac"] or 0) >= K["min_positive_month_frac"],
        "min_6_test_months": res["n_test_months"] >= 6,
    }
    verdict = "PASS" if all(checks.values()) else (
        "INSUFFICIENT_DATA" if not checks["min_6_test_months"] else "FAIL")

    out = {
        "model": "alpha_xs", "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "store_coverage": cov, "config": CFG,
        "result": res, "checks": checks, "verdict": verdict,
        "duration_seconds": round(time.time()-t0, 1),
    }
    json.dump(out, open(args.out, "w"), indent=1, default=str)

    print(f"\n===== CROSS-SECTIONAL ALPHA ({time.time()-t0:.0f}s) =====")
    print("backend:", res["backend"], "| test months:", res["n_test_months"])
    print("mean rank-IC:", res["mean_rank_ic"], "| IC t-stat:", res["ic_t_stat"],
          "| positive months:", res["positive_month_frac"])
    print("top features:", list(res["feature_importance"].items())[:8])
    print("checks:", checks)
    print("VERDICT:", verdict)
    print(f"artifact -> {args.out}")


if __name__ == "__main__":
    main()
