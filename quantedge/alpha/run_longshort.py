"""Long-short driver — same walk-forward as run_portfolio, long-short spread + gate."""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import date, timedelta
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import yaml
from quantedge.data.price_store import PriceStore
from quantedge.fundamentals.universe_full import ticker_cik_map
from quantedge.alpha.panel import build_panel_for_date, month_ends, forward_return
from quantedge.alpha.model import CrossSectionalModel
from quantedge.alpha.longshort import backtest_long_short
from quantedge.alpha.run_portfolio import _costs_params
from quantedge.harness.costs import round_trip_cost_bps

A = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "params_alpha.yaml")))
CFG = A["alpha_xs"]; LS = A["alpha_longshort"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    store = PriceStore(args.price_db)
    cov = store.coverage()
    first = date.fromisoformat(cov["from"]); last = date.fromisoformat(cov["to"])
    dates = month_ends(first + timedelta(days=400),
                       last - timedelta(days=int(CFG["fwd_nbars"] * 1.6) + 15))
    cikmap = ticker_cik_map()
    print(f"{len(dates)} months", flush=True)
    panel = {}
    for i, d in enumerate(dates):
        panel[d.isoformat()] = build_panel_for_date(
            store, cikmap, d, CFG["fwd_nbars"], CFG["min_dollar_adv"])
        if i % 6 == 0:
            print(f"  panel {d}: {len(panel[d.isoformat()])} ({time.time()-t0:.0f}s)", flush=True)
    dkeys = sorted(panel)
    rt = round_trip_cost_bps(_costs_params()) / 1e4
    monthly = []
    for i in range(CFG["min_train_months"], len(dkeys)):
        td = dkeys[i]
        train = [r for d in dkeys[:i] for r in panel[d]]
        test = panel[td]
        try:
            m = CrossSectionalModel().fit(train, CFG["label"])
        except ValueError:
            continue
        preds = list(zip([r["ticker"] for r in test], m.predict(test)))
        realized = {r["ticker"]: r["_fwd_3m_raw"] for r in test}
        monthly.append({"as_of": td, "preds": preds, "realized": realized})
    res = backtest_long_short(monthly, LS["top_frac"], LS["bottom_frac"], rt)
    K = LS["kill_threshold"]
    checks = {
        "ann_spread": res.get("ann_spread", -1) >= K["min_ann_spread"],
        "sharpe": (res.get("sharpe") or -1) >= K["min_sharpe"],
        "t_stat": (res.get("t_stat") or -1) >= K["min_t_stat"],
        "positive_frac": (res.get("positive_frac") or -1) >= K["min_positive_frac"],
        "min_periods": res.get("n_periods", 0) >= 6,
    }
    verdict = "PASS" if res.get("ok") and all(checks.values()) else (
        "INSUFFICIENT_DATA" if not res.get("ok") else "FAIL")
    out = {"model": "alpha_longshort", "roundtrip_cost": rt, "config": LS,
           "result": res, "checks": checks, "verdict": verdict}
    json.dump(out, open(args.out, "w"), indent=1, default=str)
    print(f"\n===== LONG-SHORT ({time.time()-t0:.0f}s) =====")
    if res.get("ok"):
        print(f"periods {res['n_periods']} | ann spread {res['ann_spread']} | "
              f"Sharpe {res['sharpe']} | t {res['t_stat']} | pos {res['positive_frac']} | "
              f"MDD {res['max_drawdown']}")
        print(f"gross->net period: {res['gross_mean_period_spread']} -> {res['mean_period_spread']}")
    print("checks:", checks)
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
