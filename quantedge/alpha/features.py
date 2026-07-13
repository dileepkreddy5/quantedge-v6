"""Cross-sectional feature engine (Stage F) — point-in-time, no lookahead.

Implements the feature families that Gu, Kelly & Xiu (2020), "Empirical Asset
Pricing via Machine Learning" found to carry cross-sectional predictive power
on the full US market. Every feature for a (ticker, as_of) pair is computed
ONLY from bars/filings dated <= as_of. The label is a FORWARD return, never
mixed into features.

Feature families (all from the local price store + bulk fundamentals we own):
  MOMENTUM     r_1m, r_3m, r_6m, r_12m, r_12m_ex_1m (classic 12-1 momentum)
  REVERSAL     r_1w, r_1m (short-horizon reversal, opposite sign expected)
  VOLATILITY   realized vol 21d / 63d, downside deviation
  DRAWDOWN     drawdown from 3y/1y high, days underwater, pct off 1y low
  VOLUME       dollar-vol trend (recent/baseline), up-day volume share
  LIQUIDITY    log dollar-ADV (size/tradeability proxy)
  VALUE        P/S vs own 5y median percentile (reuses rebound discount math)
  QUALITY      Piotroski, gross-margin trend, accruals (reuses extractors)
  GROWTH       revenue growth streak, latest YoY

Design: pure functions over a bar list + a facts blob. Missing inputs yield
None for that feature (the model handles missingness), never a fabricated 0.
"""
from __future__ import annotations
import math
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

Bars = List[Tuple[date, float, float]]   # (date, close, volume) ascending


# lookbacks are in TRADING days (bars), not calendar days — a calendar
# window misaligns with the number of bars and corrupts momentum (a 21
# calendar-day window is only ~15 bars). Approx map: 1w=5, 1m=21, 3m=63,
# 6m=126, 12m=252 trading days.
def _ret_bars(bars: Bars, nbars: int, as_of: date) -> Optional[float]:
    px = [(d, c) for d, c, _ in bars if d <= as_of]
    if len(px) < nbars + 1:
        return None
    end = px[-1][1]
    start = px[-(nbars + 1)][1]
    if start <= 0:
        return None
    return end / start - 1.0


_CAL_TO_BARS = {7: 5, 21: 21, 63: 63, 126: 126, 365: 252}


def _ret(bars: Bars, lookback_days: int, as_of: date) -> Optional[float]:
    nbars = _CAL_TO_BARS.get(lookback_days, max(1, round(lookback_days * 252 / 365)))
    return _ret_bars(bars, nbars, as_of)


def _vol(bars: Bars, ndays: int, as_of: date) -> Optional[float]:
    px = [c for d, c, _ in bars if d <= as_of][-(ndays + 1):]
    if len(px) < ndays // 2 or len(px) < 3:
        return None
    rets = [px[i] / px[i - 1] - 1.0 for i in range(1, len(px)) if px[i - 1] > 0]
    if len(rets) < 3:
        return None
    m = sum(rets) / len(rets)
    var = sum((r - m) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var) * math.sqrt(252)


def _downside_dev(bars: Bars, ndays: int, as_of: date) -> Optional[float]:
    px = [c for d, c, _ in bars if d <= as_of][-(ndays + 1):]
    if len(px) < 3:
        return None
    downs = [px[i] / px[i - 1] - 1.0 for i in range(1, len(px))
             if px[i - 1] > 0 and px[i] < px[i - 1]]
    if len(downs) < 2:
        return 0.0
    return math.sqrt(sum(r * r for r in downs) / len(downs)) * math.sqrt(252)


def _drawdown(bars: Bars, lookback_days: int, as_of: date) -> Optional[Dict]:
    px = [(d, c) for d, c, _ in bars if d <= as_of
          and d >= as_of - timedelta(days=lookback_days)]
    if len(px) < 30:
        return None
    now = px[-1][1]
    hi_d, hi = max(px, key=lambda x: x[1])
    lo_d, lo = min(px, key=lambda x: x[1])
    return {
        "dd": 1.0 - now / hi if hi > 0 else None,
        "days_underwater": (as_of - hi_d).days,
        "off_low": now / lo - 1.0 if lo > 0 else None,
        "days_since_low": (as_of - lo_d).days,
    }


def _volume(bars: Bars, as_of: date) -> Dict[str, Optional[float]]:
    b = [(d, c, v) for d, c, v in bars if d <= as_of]
    if len(b) < 65:
        return {"dollar_vol_trend": None, "up_day_share": None, "log_adv": None}
    recent = b[-21:]
    baseline = b[-84:-21]
    rec_dv = sum(c * v for _, c, v in recent) / len(recent)
    base_dv = sum(c * v for _, c, v in baseline) / len(baseline) if baseline else 0
    up_vol = sum(v for (_, c, v), (_, pc, _) in zip(recent[1:], recent[:-1]) if c > pc)
    tot_vol = sum(v for _, _, v in recent[1:])
    return {
        "dollar_vol_trend": rec_dv / base_dv if base_dv > 0 else None,
        "up_day_share": up_vol / tot_vol if tot_vol > 0 else None,
        "log_adv": math.log10(rec_dv) if rec_dv > 0 else None,
    }


def compute_features(bars: Bars, as_of: date,
                     fundamentals: Optional[Dict] = None) -> Dict[str, Optional[float]]:
    """Full point-in-time feature vector for one (ticker, as_of).
    `fundamentals` is an optional pre-computed dict of the slow-moving factors
    (value/quality/growth) so the panel builder can compute them once per
    quarter instead of per month. Price features are always fresh."""
    f: Dict[str, Optional[float]] = {}

    # momentum
    f["r_1w"] = _ret(bars, 7, as_of)
    f["r_1m"] = _ret(bars, 21, as_of)
    f["r_3m"] = _ret(bars, 63, as_of)
    f["r_6m"] = _ret(bars, 126, as_of)
    f["r_12m"] = _ret(bars, 365, as_of)
    if f["r_12m"] is not None and f["r_1m"] is not None:
        r12 = _ret(bars, 365, as_of)
        r1 = _ret(bars, 21, as_of)
        f["r_12m_ex_1m"] = (1 + r12) / (1 + r1) - 1 if r1 is not None and r1 > -1 else None
    else:
        f["r_12m_ex_1m"] = None

    # volatility
    f["vol_21d"] = _vol(bars, 21, as_of)
    f["vol_63d"] = _vol(bars, 63, as_of)
    f["downside_dev"] = _downside_dev(bars, 63, as_of)

    # drawdown
    dd3 = _drawdown(bars, 3 * 365, as_of)
    dd1 = _drawdown(bars, 365, as_of)
    f["dd_3y"] = dd3["dd"] if dd3 else None
    f["days_underwater"] = float(dd3["days_underwater"]) if dd3 else None
    f["off_1y_low"] = dd1["off_low"] if dd1 else None
    f["days_since_1y_low"] = float(dd1["days_since_low"]) if dd1 else None

    # volume / liquidity
    f.update(_volume(bars, as_of))

    # slow-moving fundamentals (passed in pre-computed, PIT)
    keys = ("ps_percentile_own", "piotroski", "gross_margin_slope",
            "accruals", "growth_streak", "latest_yoy", "rd_to_mktcap")
    for k in keys:
        f[k] = fundamentals.get(k) if fundamentals else None

    return f


FEATURE_NAMES = [
    "r_1w", "r_1m", "r_3m", "r_6m", "r_12m", "r_12m_ex_1m",
    "vol_21d", "vol_63d", "downside_dev",
    "dd_3y", "days_underwater", "off_1y_low", "days_since_1y_low",
    "dollar_vol_trend", "up_day_share", "log_adv",
    "ps_percentile_own", "piotroski", "gross_margin_slope",
    "accruals", "growth_streak", "latest_yoy", "rd_to_mktcap",
]
