"""Cross-sectional training panel builder (Stage F) — real data only.

For each monthly as_of date and each liquid ticker, emits:
  { as_of, ticker, features..., fwd_ret_3m, fwd_ret_1m }
using ONLY the local price store (features and forward returns) plus optional
pre-computed PIT fundamentals. Forward returns are REAL future closes from the
store; a row with no matured forward window is emitted with label=None and
dropped at training time — never fabricated.

Cross-sectional target: within each as_of date, forward returns are RANKED to
[0,1] (cross-sectional percentile). Gu-Kelly-Xiu predict the cross-section
(which stocks beat OTHERS this month), which is regime-robust — unlike an
absolute-return target that just tracks the market. This is why the approach
can have signal even without a bear market in the window.
"""
from __future__ import annotations
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

from quantedge.alpha.features import compute_features, FEATURE_NAMES

Bars = List[Tuple[date, float, float]]


def forward_return(bars: Bars, as_of: date, nbars: int) -> Optional[float]:
    """Return from first close AFTER as_of to nbars trading-bars later.
    None if the window hasn't matured (no fabrication, no forward-fill)."""
    fut = [(d, c) for d, c, _ in bars if d > as_of]
    if len(fut) < nbars + 1:
        return None
    entry = fut[0][1]
    exit_ = fut[nbars][1]
    if entry <= 0:
        return None
    return exit_ / entry - 1.0


def month_ends(start: date, end: date) -> List[date]:
    out, y, m = [], start.year, start.month
    while (y, m) <= (end.year, end.month):
        nm_y, nm_m = (y + 1, 1) if m == 12 else (y, m + 1)
        last = date(nm_y, nm_m, 1) - timedelta(days=1)
        if start <= last <= end:
            out.append(last)
        y, m = nm_y, nm_m
    return out


def cross_sectional_rank(values: List[Optional[float]]) -> List[Optional[float]]:
    """Rank non-None values to [0,1]; None stays None."""
    idx = [i for i, v in enumerate(values) if v is not None]
    if len(idx) < 2:
        return [None] * len(values)
    order = sorted(idx, key=lambda i: values[i])
    out: List[Optional[float]] = [None] * len(values)
    for rank, i in enumerate(order):
        out[i] = rank / (len(idx) - 1)
    return out


def build_panel_for_date(store, cikmap: Dict[str, str], as_of: date,
                         fwd_nbars: int, min_dollar_adv: float,
                         fundamentals_fn=None) -> List[Dict]:
    """One as_of slice: features + raw forward returns for every liquid ticker.
    Cross-sectional ranking of labels is applied within the slice."""
    rows: List[Dict] = []
    day = as_of
    for _ in range(7):
        if store.closes_on(day):
            break
        day -= timedelta(days=1)
    universe = sorted(set(store.closes_on(day)) & set(cikmap))

    for t in universe:
        bars = store.series(t, as_of - timedelta(days=3 * 366),
                            as_of + timedelta(days=int(fwd_nbars * 1.6) + 20))
        hist = [b for b in bars if b[0] <= as_of]
        if len(hist) < 260:
            continue
        recent = hist[-21:]
        adv = sum(c * v for _, c, v in recent) / len(recent)
        if adv < min_dollar_adv:
            continue
        fund = fundamentals_fn(t, as_of) if fundamentals_fn else None
        feats = compute_features(hist, as_of, fund)
        row = {"as_of": as_of.isoformat(), "ticker": t}
        row.update({k: feats.get(k) for k in FEATURE_NAMES})
        row["_fwd_3m_raw"] = forward_return(bars, as_of, fwd_nbars)
        row["_fwd_1m_raw"] = forward_return(bars, as_of, 21)
        rows.append(row)

    for label_key, raw_key in (("fwd_3m_rank", "_fwd_3m_raw"),
                               ("fwd_1m_rank", "_fwd_1m_raw")):
        ranked = cross_sectional_rank([r[raw_key] for r in rows])
        for r, rk in zip(rows, ranked):
            r[label_key] = rk
    return rows
