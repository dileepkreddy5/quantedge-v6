"""Fundamentals-enhanced full evaluation — IC + long-only + long-short in one
pass, using the panel cache and the fundamentals hook. Same frozen gates.

Builds the panel ONCE (with fundamentals features populated), caches it, then
runs all three pre-committed tests against their existing frozen thresholds.
New information (fundamentals), identical bars — a fair, single evaluation.

VPS:
  PYTHONPATH=. python3 quantedge/alpha/run_full.py \
    --price-db /opt/quantedge-research/data/price_store.db \
    --panel-cache /opt/quantedge-research/panels_fund.json \
    --out /opt/quantedge-research/alpha_full_fund.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
from datetime import date, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import yaml

from quantedge.data.price_store import PriceStore
from quantedge.fundamentals.universe_full import ticker_cik_map
from quantedge.alpha.panel import build_panel_for_date, month_ends, forward_return
from quantedge.alpha.panel_cache import _cache_key
from quantedge.alpha.model import CrossSectionalModel, walk_forward
from quantedge.alpha.portfolio import backtest_long_book
from quantedge.alpha.longshort import backtest_long_short
from quantedge.alpha.fundamentals_precompute import FundTable
from quantedge.alpha.run_portfolio import _costs_params
from quantedge.harness.costs import round_trip_cost_bps

A = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), "params_alpha.yaml")))
CFG, PCFG, LS = A["alpha_xs"], A["alpha_portfolio"], A["alpha_longshort"]


def _build_with_fundamentals(store, cikmap, cfg, cache_path, fund_db):
    cov = store.coverage()
    fund_cfg = dict(cfg); fund_cfg["_fund"] = True
    key = _cache_key(cov, fund_cfg, len(cikmap))
    if os.path.exists(cache_path):
        blob = json.load(open(cache_path))
        if blob.get("key") == key:
            print(f"fund panel cache HIT — {len(blob['panel'])} months", flush=True)
            return blob["panel"]

    prov = FundTable(fund_db, cikmap)     # reads scalar rows, memory-safe

    first = date.fromisoformat(cov["from"]); last = date.fromisoformat(cov["to"])
    dates = month_ends(first + timedelta(days=400),
                       last - timedelta(days=int(cfg["fwd_nbars"]*1.6)+15))
    panel, t0 = {}, time.time()
    for i, d in enumerate(dates):
        panel[d.isoformat()] = build_panel_for_date(
            store, cikmap, d, cfg["fwd_nbars"], cfg["min_dollar_adv"],
            fundamentals_fn=prov)
        if i % 6 == 0:
            print(f"  fund panel {d}: {len(panel[d.isoformat()])} ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"key": key, "coverage": cov, "panel": panel},
              open(cache_path, "w"), default=str)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-db", required=True)
    ap.add_argument("--panel-cache", required=True)
    ap.add_argument("--fund-db", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    store = PriceStore(args.price_db)
    cikmap = ticker_cik_map()
    panel = _build_with_fundamentals(store, cikmap, CFG, args.panel_cache, args.fund_db)

    ic = walk_forward(panel, CFG["label"], CFG["min_train_months"])

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
        spy_bars = store.series("SPY", date.fromisoformat(td)-timedelta(days=5),
                                date.fromisoformat(td)+timedelta(days=160))
        spy = forward_return(spy_bars, date.fromisoformat(td), CFG["fwd_nbars"])
        monthly.append({"as_of": td, "preds": preds, "realized": realized, "spy": spy})

    lo = backtest_long_book(monthly, PCFG["top_frac"], rt)
    lsr = backtest_long_short(monthly, LS["top_frac"], LS["bottom_frac"], rt)

    ic_checks = {
        "ic": (ic["mean_rank_ic"] or 0) >= CFG["kill_threshold"]["min_mean_rank_ic"],
        "t": (ic["ic_t_stat"] or 0) >= CFG["kill_threshold"]["min_ic_t_stat"],
        "pos": (ic["positive_month_frac"] or 0) >= CFG["kill_threshold"]["min_positive_month_frac"],
    }
    ls_checks = {
        "spread": (lsr.get("ann_spread") or -1) >= LS["kill_threshold"]["min_ann_spread"],
        "sharpe": (lsr.get("sharpe") or -1) >= LS["kill_threshold"]["min_sharpe"],
        "t": (lsr.get("t_stat") or -1) >= LS["kill_threshold"]["min_t_stat"],
    }
    out = {"model": "alpha_xs_fundamentals", "with_fundamentals": True,
           "ic": ic, "ic_checks": ic_checks,
           "long_only": lo, "long_short": lsr, "ls_checks": ls_checks,
           "duration_seconds": round(time.time()-t0, 1)}
    json.dump(out, open(args.out, "w"), indent=1, default=str)

    print(f"\n===== FUNDAMENTALS-ENHANCED ({time.time()-t0:.0f}s) =====")
    print(f"IC: {ic['mean_rank_ic']} t={ic['ic_t_stat']} pos={ic['positive_month_frac']} "
          f"| checks {ic_checks}")
    print("top features:", list(ic["feature_importance"].items())[:8])
    if lo.get("ok"):
        print(f"long-only: ann excess {lo['ann_excess_return']} Sharpe {lo['sharpe_excess']} t {lo['t_stat']}")
    if lsr.get("ok"):
        print(f"long-short: ann spread {lsr['ann_spread']} Sharpe {lsr['sharpe']} t {lsr['t_stat']} "
              f"| checks {ls_checks}")
    print(f"artifact -> {args.out}")


if __name__ == "__main__":
    main()
