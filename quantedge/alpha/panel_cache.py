"""Panel cache — build the monthly panels once, reuse across every test.

Every alpha backtest (IC, long-only, long-short, risk-adjusted, fundamentals)
rebuilds the SAME monthly feature panels — ~8 minutes each. This caches them
to a JSON file keyed by the store's coverage + config, so subsequent runs load
in seconds. Cache invalidates automatically if the store range or the feature
config changes (so stale panels can never silently be reused)."""
from __future__ import annotations
import hashlib, json, os
from datetime import date, timedelta
from typing import Dict, List

from quantedge.alpha.panel import build_panel_for_date, month_ends


def _cache_key(cov: dict, cfg: dict, n_universe: int) -> str:
    h = hashlib.sha256(json.dumps({
        "from": cov["from"], "to": cov["to"], "rows": cov["rows"],
        "fwd_nbars": cfg["fwd_nbars"], "min_dollar_adv": cfg["min_dollar_adv"],
        "n_universe": n_universe,
    }, sort_keys=True).encode()).hexdigest()[:16]
    return h


def load_or_build(store, cikmap: Dict[str, str], cfg: dict,
                  cache_path: str, verbose: bool = True) -> Dict[str, List[Dict]]:
    cov = store.coverage()
    key = _cache_key(cov, cfg, len(cikmap))
    if os.path.exists(cache_path):
        try:
            blob = json.load(open(cache_path))
            if blob.get("key") == key:
                if verbose:
                    print(f"panel cache HIT ({key}) — {len(blob['panel'])} months loaded", flush=True)
                return blob["panel"]
            elif verbose:
                print("panel cache STALE (config/store changed) — rebuilding", flush=True)
        except Exception:
            pass

    first = date.fromisoformat(cov["from"]); last = date.fromisoformat(cov["to"])
    dates = month_ends(first + timedelta(days=400),
                       last - timedelta(days=int(cfg["fwd_nbars"] * 1.6) + 15))
    panel = {}
    import time
    t0 = time.time()
    for i, d in enumerate(dates):
        panel[d.isoformat()] = build_panel_for_date(
            store, cikmap, d, cfg["fwd_nbars"], cfg["min_dollar_adv"])
        if verbose and i % 6 == 0:
            print(f"  build {d}: {len(panel[d.isoformat()])} ({time.time()-t0:.0f}s)", flush=True)
    json.dump({"key": key, "coverage": cov, "panel": panel},
              open(cache_path, "w"), default=str)
    if verbose:
        print(f"panel cache WRITTEN ({key}) — {len(panel)} months", flush=True)
    return panel
