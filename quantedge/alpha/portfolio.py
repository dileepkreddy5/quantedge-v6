"""Portfolio construction + net-of-cost backtest (Stage H) — does IC become money?

An information coefficient is not a return. This unit takes the walk-forward
predictions and builds an ACTUAL tradeable book each month, then measures
realized net-of-cost performance. Honest construction:

  LONG BOOK      each month, long the top-decile predicted stocks (the model's
                 highest-ranked names). Equal-weight within the decile — the
                 robust default; fancier weighting is over-fitting until proven.
  BENCHMARK      SPY over the identical holding window. We report EXCESS return.
  COSTS          committed round-trip cost (harness/costs) applied to TURNOVER:
                 stocks entering/leaving the book each rebalance pay the toll.
  METRICS        annualized excess return, annualized vol, Sharpe (excess/vol),
                 hit rate of period excess, max drawdown of the excess curve.

Long-only first: it's what a single owner can actually trade, needs no borrow,
and if the signal can't even beat SPY long-only after costs, a long-short book
is a fantasy. Gate is frozen in params before results.
"""
from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple


def _decile_longs(preds_rows: List[Tuple[str, float]], top_frac: float) -> List[str]:
    ranked = sorted(preds_rows, key=lambda x: x[1], reverse=True)
    n = max(1, int(len(ranked) * top_frac))
    return [t for t, _ in ranked[:n]]


def _book_return(longs: List[str], realized: Dict[str, Optional[float]]) -> Optional[float]:
    rs = [realized[t] for t in longs if realized.get(t) is not None]
    if not rs:
        return None
    return sum(rs) / len(rs)


def backtest_long_book(monthly: List[Dict], top_frac: float,
                       roundtrip_cost: float) -> Dict:
    prev_longs: set = set()
    excess_series: List[Tuple[str, float]] = []
    gross_series: List[float] = []
    turnovers: List[float] = []

    for m in monthly:
        preds = m["preds"]
        if not preds:
            continue
        longs = _decile_longs(preds, top_frac)
        book = _book_return(longs, m["realized"])
        spy = m.get("spy")
        if book is None or spy is None:
            continue
        cur = set(longs)
        turnover = 1.0 if not prev_longs else len(cur - prev_longs) / max(len(cur), 1)
        prev_longs = cur
        cost = roundtrip_cost * turnover
        net_excess = (book - spy) - cost
        excess_series.append((m["as_of"], round(net_excess, 5)))
        gross_series.append(book - spy)
        turnovers.append(turnover)

    xs = [v for _, v in excess_series]
    if len(xs) < 6:
        return {"ok": False, "reason": "too_few_months", "n": len(xs)}

    periods_per_year = 4.0
    mean_x = sum(xs) / len(xs)
    var_x = sum((x - mean_x) ** 2 for x in xs) / (len(xs) - 1)
    sd_x = math.sqrt(var_x)
    ann_excess = mean_x * periods_per_year
    ann_vol = sd_x * math.sqrt(periods_per_year)
    sharpe = ann_excess / ann_vol if ann_vol > 0 else None

    cum, peak, mdd = 1.0, 1.0, 0.0
    for x in xs:
        cum *= (1 + x)
        peak = max(peak, cum)
        mdd = min(mdd, cum / peak - 1)

    t = mean_x / (sd_x / len(xs) ** 0.5) if sd_x > 0 else None

    return {
        "ok": True,
        "n_periods": len(xs),
        "ann_excess_return": round(ann_excess, 4),
        "ann_vol": round(ann_vol, 4),
        "sharpe_excess": round(sharpe, 2) if sharpe is not None else None,
        "hit_rate": round(sum(1 for x in xs if x > 0) / len(xs), 3),
        "max_drawdown_excess": round(mdd, 4),
        "avg_turnover": round(sum(turnovers) / len(turnovers), 3),
        "mean_period_excess": round(mean_x, 5),
        "t_stat": round(t, 2) if t is not None else None,
        "gross_mean_period_excess": round(sum(gross_series) / len(gross_series), 5),
        "excess_series": excess_series,
    }
