"""Portfolio backtest driver (Stage H) — turns the validated model into a book.

Rebuilds the SAME walk-forward as run_alpha (train on past months, predict the
test month) but instead of only scoring IC, it constructs the top-decile long
book each test month, fetches REAL forward returns for the held names and SPY,
applies committed costs to turnover, and judges net performance against the
frozen alpha_portfolio gate.

VPS:
  PYTHONPATH=. python3 quantedge/alpha/run_portfolio.py \
    --price-db /opt/quantedge-research/data/price_store.db \
    --out /opt/quantedge-research/alpha_portfolio.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import yaml

from quantedge.data.price_store import PriceStore
from quantedge.fundamentals.universe_full import ticker_cik_map
from quantedge.alpha.panel import build_panel_for_date, month_ends, forward_return
from quantedge.alpha.model import CrossSectionalModel
from quantedge.alpha.portfolio import backtest_long_book
from quantedge.harness.costs import round_trip_cost_bps

A = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "params_alpha.yaml")))
CFG = A["alpha_xs"]
PCFG = A["alpha_portfolio"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-db", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-universe", type=int, default=None)
    args = ap.parse_args()

    t0 = time.time()
    store = PriceStore(args.price_db)
    cov = store.coverage()
    first = date.fromisoformat(cov["from"]); last = date.fromisoformat(cov["to"])
    start = first + timedelta(days=400)
    end = last - timedelta(days=int(CFG["fwd_nbars"] * 1.6) + 15)
    dates = month_ends(start, end)
    cikmap = ticker_cik_map()
    if args.sample_universe:
        cikmap = dict(list(cikmap.items())[:args.sample_universe])
    print(f"coverage {cov['from']}..{cov['to']} | {len(dates)} months", flush=True)

    panel = {}
    for i, d in enumerate(dates):
        panel[d.isoformat()] = build_panel_for_date(
            store, cikmap, d, CFG["fwd_nbars"], CFG["min_dollar_adv"])
        if i % 6 == 0:
            print(f"  panel {d}: {len(panel[d.isoformat()])} ({time.time()-t0:.0f}s)", flush=True)

    dkeys = sorted(panel)
    label = CFG["label"]
    rt = round_trip_cost_bps(_costs_params()) / 1e4

    monthly = []
    for i in range(CFG["min_train_months"], len(dkeys)):
        test_d = dkeys[i]
        train_rows = [r for d in dkeys[:i] for r in panel[d]]
        test_rows = panel[test_d]
        try:
            m = CrossSectionalModel().fit(train_rows, label)
        except ValueError:
            continue
        preds = list(zip([r["ticker"] for r in test_rows], m.predict(test_rows)))
        realized = {r["ticker"]: r["_fwd_3m_raw"] for r in test_rows}
        spy_bars = store.series("SPY", date.fromisoformat(test_d) - timedelta(days=5),
                                date.fromisoformat(test_d) + timedelta(days=160))
        spy = forward_return(spy_bars, date.fromisoformat(test_d), CFG["fwd_nbars"])
        monthly.append({"as_of": test_d, "preds": preds,
                        "realized": realized, "spy": spy})

    res = backtest_long_book(monthly, PCFG["top_frac"], rt)

    K = PCFG["kill_threshold"]
    checks = {
        "ann_excess": res.get("ann_excess_return", -1) >= K["min_ann_excess_return"],
        "sharpe": (res.get("sharpe_excess") or -1) >= K["min_sharpe_excess"],
        "t_stat": (res.get("t_stat") or -1) >= K["min_t_stat"],
        "drawdown": (res.get("max_drawdown_excess") or -1) >= K["max_drawdown_excess"],
        "min_periods": res.get("n_periods", 0) >= 6,
    }
    verdict = "PASS" if res.get("ok") and all(checks.values()) else (
        "INSUFFICIENT_DATA" if not res.get("ok") else "FAIL")

    out = {"model": "alpha_portfolio", "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
           "roundtrip_cost": rt, "config": PCFG, "result": res,
           "checks": checks, "verdict": verdict}
    json.dump(out, open(args.out, "w"), indent=1, default=str)

    print(f"\n===== PORTFOLIO BACKTEST ({time.time()-t0:.0f}s) =====")
    if res.get("ok"):
        print(f"periods: {res['n_periods']} | ann excess: {res['ann_excess_return']} "
              f"| Sharpe: {res['sharpe_excess']} | hit: {res['hit_rate']}")
        print(f"turnover: {res['avg_turnover']} | MDD excess: {res['max_drawdown_excess']} "
              f"| t: {res['t_stat']}")
        print(f"gross->net period excess: {res['gross_mean_period_excess']} -> {res['mean_period_excess']}")
    print("checks:", checks)
    print("VERDICT:", verdict)
    print(f"artifact -> {args.out}")


def _costs_params():
    import yaml as _y
    for p in ("quantedge/params.yaml", os.path.join(os.path.dirname(__file__), "..", "params.yaml")):
        if os.path.exists(p):
            d = _y.safe_load(open(p))
            if "costs" in d:
                return d
    return {"costs": {"commission_bps": 1.0, "slippage_bps": 5.0, "spread_bps": 2.0}}


if __name__ == "__main__":
    main()
