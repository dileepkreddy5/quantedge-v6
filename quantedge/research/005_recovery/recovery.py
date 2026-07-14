"""Recovery-to-high test (unit 005) — the user's actual thesis, measured directly.

Thesis: "buy a stock when it's low but has good financials, wait for it to come
back to its high." No prior test measured THIS — they measured fixed-3-month
excess return. This measures the EVENT the thesis is about: does a beaten-down
stock with good financials RECOVER to a prior high, and how long does it take?

  ENTRY universe : down >= min_drawdown from 3y high (beaten-down)
  HEALTHY group  : entry universe AND passes health gate (growth streak +
                   F-score) — reuses the tested REBOUND health logic
  CONTROL group  : entry universe AND FAILS health gate (beaten-down but weak)
  SUCCESS        : within tracking_days, the close reaches recover_frac of the
                   way back from entry to the prior 3y high
  MEASURED       : recovery hit-rate (healthy vs control) and median days-to-
                   recover among winners. Frozen gate: healthy hit-rate must
                   exceed control by min_edge with enough events.

No lookahead: entry uses only data <= as_of; outcome scans only bars AFTER as_of.
"""
from __future__ import annotations
from datetime import date, timedelta
from statistics import median
from typing import Dict, List, Optional, Tuple

Bars = List[Tuple[date, float]]


def prior_high(bars: Bars, as_of: date, lookback_days: int) -> Optional[float]:
    win = [c for d, c in bars if as_of - timedelta(days=lookback_days) <= d <= as_of]
    return max(win) if win else None


def drawdown_at(bars: Bars, as_of: date, lookback_days: int) -> Optional[float]:
    hi = prior_high(bars, as_of, lookback_days)
    now = [c for d, c in bars if d <= as_of]
    if not hi or not now or hi <= 0:
        return None
    return 1.0 - now[-1] / hi


def recovery_outcome(bars: Bars, as_of: date, target_price: float,
                     tracking_days: int) -> Dict:
    fut = [(d, c) for d, c in bars if as_of < d <= as_of + timedelta(days=tracking_days)]
    if not fut:
        return {"matured": False}
    for d, c in fut:
        if c >= target_price:
            return {"matured": True, "recovered": True, "days": (d - as_of).days}
    return {"matured": True, "recovered": False,
            "days": None, "max_frac": max(c for _, c in fut)}


def evaluate(events: List[Dict], params: dict) -> Dict:
    P = params["recovery"]
    def summarize(group):
        mat = [e for e in group if e["outcome"].get("matured")]
        if not mat:
            return None
        rec = [e for e in mat if e["outcome"].get("recovered")]
        days = [e["outcome"]["days"] for e in rec if e["outcome"].get("days")]
        return {"n": len(mat), "hit_rate": round(len(rec) / len(mat), 4),
                "median_days_to_recover": median(days) if days else None,
                "n_recovered": len(rec)}
    healthy = summarize([e for e in events if e["healthy"]])
    control = summarize([e for e in events if not e["healthy"]])
    out = {"healthy": healthy, "control": control}
    if not healthy or not control:
        out["verdict"] = "INSUFFICIENT_DATA"
        return out
    edge = healthy["hit_rate"] - control["hit_rate"]
    out["hit_rate_edge"] = round(edge, 4)
    import math
    n1, n2 = healthy["n"], control["n"]
    pool = (healthy["n_recovered"] + control["n_recovered"]) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1/n1 + 1/n2)) if 0 < pool < 1 else 0
    z = edge / se if se > 0 else 0
    out["z_stat"] = round(z, 2)
    checks = {
        "edge>=min": edge >= P["min_hit_rate_edge"],
        "z>=min": z >= P["min_z_stat"],
        "healthy_n>=min": healthy["n"] >= P["min_events"],
        "control_n>=min": control["n"] >= P["min_events"],
    }
    out["checks"] = checks
    out["verdict"] = "PASS" if all(checks.values()) else "FAIL"
    return out
