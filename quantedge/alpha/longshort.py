"""Long-short spread test (pre-registered) — top-decile minus bottom-decile."""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple


def _mean_ret(names: List[str], realized: Dict[str, Optional[float]]) -> Optional[float]:
    rs = [realized[t] for t in names if realized.get(t) is not None]
    return sum(rs) / len(rs) if rs else None


def backtest_long_short(monthly: List[Dict], top_frac: float, bottom_frac: float,
                        roundtrip_cost: float) -> Dict:
    prev_l, prev_s = set(), set()
    series: List[Tuple[str, float]] = []
    gross: List[float] = []
    for m in monthly:
        preds = m["preds"]
        if not preds:
            continue
        ranked = sorted(preds, key=lambda x: x[1], reverse=True)
        nl = max(1, int(len(ranked) * top_frac))
        ns = max(1, int(len(ranked) * bottom_frac))
        longs = [t for t, _ in ranked[:nl]]
        shorts = [t for t, _ in ranked[-ns:]]
        lr = _mean_ret(longs, m["realized"])
        sr = _mean_ret(shorts, m["realized"])
        if lr is None or sr is None:
            continue
        cl = set(longs); cs = set(shorts)
        turn = (1.0 if not prev_l else len(cl - prev_l) / max(len(cl), 1)) + \
               (1.0 if not prev_s else len(cs - prev_s) / max(len(cs), 1))
        prev_l, prev_s = cl, cs
        spread = (lr - sr) - roundtrip_cost * turn
        series.append((m["as_of"], round(spread, 5)))
        gross.append(lr - sr)
    xs = [v for _, v in series]
    if len(xs) < 6:
        return {"ok": False, "reason": "too_few", "n": len(xs)}
    ppy = 4.0
    mean_x = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mean_x) ** 2 for x in xs) / (len(xs) - 1))
    ann = mean_x * ppy
    annvol = sd * math.sqrt(ppy)
    sharpe = ann / annvol if annvol > 0 else None
    cum, peak, mdd = 1.0, 1.0, 0.0
    for x in xs:
        cum *= (1 + x); peak = max(peak, cum); mdd = min(mdd, cum / peak - 1)
    t = mean_x / (sd / len(xs) ** 0.5) if sd > 0 else None
    return {
        "ok": True, "n_periods": len(xs),
        "ann_spread": round(ann, 4), "ann_vol": round(annvol, 4),
        "sharpe": round(sharpe, 2) if sharpe is not None else None,
        "positive_frac": round(sum(1 for x in xs if x > 0) / len(xs), 3),
        "max_drawdown": round(mdd, 4), "t_stat": round(t, 2) if t is not None else None,
        "mean_period_spread": round(mean_x, 5),
        "gross_mean_period_spread": round(sum(gross) / len(gross), 5),
        "series": series,
    }
