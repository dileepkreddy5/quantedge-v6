"""Backtest statistics for the REBOUND unit (step-21) — pure, unit-tested.

Design choices (committed before results):
  CONTROL GROUP  the honest benchmark is not SPY alone — any deep-drawdown
                 basket beats SPY in a broad recovery. The control is the
                 BEATEN-DOWN BASE RATE: stocks that cleared the price
                 prefilter (deep drawdown, liquid, >$2) but FAILED the
                 gates. Lift = passers' win rate / control's win rate.
  WINNER         forward excess return vs SPY (same window) > 0, NET of the
                 committed round-trip cost.
  T-STATISTIC    computed over PER-DATE spreads (mean passer excess minus
                 mean control excess per as_of date), not over pooled
                 stock-events — stocks within a date are correlated, and
                 pooling would fake the sample size. Few dates -> honest
                 small t. That is the point.
"""
from __future__ import annotations
from statistics import mean, stdev
from typing import Dict, List, Optional


def excess_net(stock_ret: Optional[float], spy_ret: Optional[float],
               roundtrip_cost: float) -> Optional[float]:
    if stock_ret is None or spy_ret is None:
        return None
    return stock_ret - spy_ret - roundtrip_cost


def per_date_spread(passers: List[float], control: List[float]) -> Optional[float]:
    """Mean net-excess of passers minus mean net-excess of control, one date."""
    if not passers or not control:
        return None
    return mean(passers) - mean(control)


def t_stat_over_dates(spreads: List[float]) -> Optional[float]:
    xs = [s for s in spreads if s is not None]
    if len(xs) < 3:
        return None
    m = mean(xs)
    sd = stdev(xs)
    if sd == 0:
        return None
    return m / (sd / len(xs) ** 0.5)


def win_rate(net_excess: List[float]) -> Optional[float]:
    xs = [x for x in net_excess if x is not None]
    if not xs:
        return None
    return sum(1 for x in xs if x > 0) / len(xs)


def lift(passer_win: Optional[float], control_win: Optional[float]) -> Optional[float]:
    if passer_win is None or not control_win:
        return None
    return passer_win / control_win


def summarize(events: List[Dict], horizon_key: str, roundtrip_cost: float) -> Dict:
    """events: [{as_of, passed(bool), stage, tier, ret_<h>, spy_<h>}].
    Returns the full verdict block for one horizon."""
    by_date: Dict[str, Dict[str, List[float]]] = {}
    for e in events:
        x = excess_net(e.get(f"ret_{horizon_key}"), e.get(f"spy_{horizon_key}"),
                       roundtrip_cost)
        if x is None:
            continue
        d = by_date.setdefault(e["as_of"], {"pass": [], "ctrl": []})
        d["pass" if e["passed"] else "ctrl"].append(x)

    spreads = [per_date_spread(v["pass"], v["ctrl"]) for v in by_date.values()]
    spreads = [s for s in spreads if s is not None]
    all_pass = [x for v in by_date.values() for x in v["pass"]]
    all_ctrl = [x for v in by_date.values() for x in v["ctrl"]]

    pw, cw = win_rate(all_pass), win_rate(all_ctrl)
    return {
        "horizon": horizon_key,
        "n_dates": len(by_date),
        "n_pass_events": len(all_pass),
        "n_ctrl_events": len(all_ctrl),
        "mean_spread_per_date": round(mean(spreads), 4) if spreads else None,
        "t_stat": (round(t, 2) if (t := t_stat_over_dates(spreads)) is not None else None),
        "passer_win_rate": round(pw, 3) if pw is not None else None,
        "control_win_rate": round(cw, 3) if cw is not None else None,
        "lift": (round(l, 2) if (l := lift(pw, cw)) is not None else None),
        "mean_passer_excess": round(mean(all_pass), 4) if all_pass else None,
        "mean_control_excess": round(mean(all_ctrl), 4) if all_ctrl else None,
    }
