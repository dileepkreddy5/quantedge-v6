"""
QuantEdge v6.0 — Ascent Radar Scoring
======================================
Surfaces US companies *climbing* from mid-cap toward large-cap on sustained,
real buying — the "don't miss the next SanDisk" board.

ASCENT SCORE = sustained relative strength (35%) + persistent volume expansion
(30%) + cap-tier climb potential (20%) + proximity to 52-week high (15%).

HONESTY NOTE: this is a DISCOVERY tool. A high ascent score means "this is
climbing — investigate," never "this will go up." No prediction is implied.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
import numpy as np
from loguru import logger


CAP_TIERS = [
    ("nano",   0,            50_000_000),
    ("micro",  50_000_000,   300_000_000),
    ("small",  300_000_000,  2_000_000_000),
    ("mid",    2_000_000_000, 10_000_000_000),
    ("large",  10_000_000_000, 200_000_000_000),
    ("mega",   200_000_000_000, float("inf")),
]


def cap_tier(market_cap: Optional[float]) -> str:
    if market_cap is None or market_cap <= 0:
        return "unknown"
    for name, lo, hi in CAP_TIERS:
        if lo <= market_cap < hi:
            return name
    return "unknown"


def _clip(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(max(lo, min(hi, x)))


@dataclass
class AscentScore:
    ticker: str
    name: str = ""
    sector: str = ""
    ascent_score: float = 0.0
    market_cap: Optional[float] = None
    tier: str = "unknown"
    strength_score: float = 0.0
    volume_score: float = 0.0
    tier_score: float = 0.0
    high_score: float = 0.0
    flags: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


def _sustained_strength(m: Dict[str, float]) -> tuple[float, List[str]]:
    flags = []
    mom_12_1 = m.get("mom_12_1", 0.0)
    mom_6m   = m.get("mom_6m", 0.0)
    mom_3m   = m.get("mom_3m", 0.0)
    mom_1m   = m.get("mom_1m", 0.0)
    sharpe_3m = m.get("sharpe_3m", 0.0)

    # NOTE: momentum metrics arrive as PERCENT (e.g. 24.7 = +24.7%), not fractions.
    def ret_to_score(r_pct, scale_pct):
        return _clip(50 + (r_pct / scale_pct) * 50)

    s_12 = ret_to_score(mom_12_1, 100.0)
    s_6  = ret_to_score(mom_6m, 60.0)
    s_3  = ret_to_score(mom_3m, 40.0)
    s_1  = ret_to_score(mom_1m, 20.0)

    all_positive = all(x > 0 for x in (mom_12_1, mom_6m, mom_3m, mom_1m))
    consistency = 1.0
    if all_positive:
        consistency = 1.15
        flags.append("Strength across all horizons (12m->1m)")

    sharpe_bonus = _clip(50 + sharpe_3m * 20)
    base = (0.35 * s_12 + 0.25 * s_6 + 0.20 * s_3 + 0.10 * s_1 + 0.10 * sharpe_bonus)
    score = _clip(base * consistency)

    if mom_3m > 30.0:
        flags.append(f"+{mom_3m*100:.0f}% over 3 months")
    if sharpe_3m > 1.5:
        flags.append("High risk-adjusted momentum")
    return score, flags


def _persistent_volume(m: Dict[str, float]) -> tuple[float, List[str]]:
    flags = []
    vol_surge = m.get("volume_surge", 1.0)
    obv_slope = m.get("obv_slope_norm", 0.0)
    pct_ma200 = m.get("pct_above_ma200", 0.0)

    surge_score = _clip(50 + (vol_surge - 1.0) * 100)
    obv_score = _clip(50 + np.sign(obv_slope) * min(abs(obv_slope) * 200, 45))
    trend_score = _clip(50 + pct_ma200 * 50)
    score = _clip(0.50 * surge_score + 0.30 * obv_score + 0.20 * trend_score)

    if vol_surge > 1.4:
        flags.append(f"Volume {vol_surge:.1f}x its 90-day norm")
    if obv_slope > 0.05:
        flags.append("Accumulation: OBV rising")
    return score, flags


def _tier_potential(market_cap: Optional[float], strength: float) -> tuple[float, str, List[str]]:
    flags = []
    tier = cap_tier(market_cap)
    order = {"nano": 0, "micro": 1, "small": 2, "mid": 3, "large": 4, "mega": 5, "unknown": 3}
    pos = order.get(tier, 3)
    headroom = max(0, 5 - pos)
    headroom_score = _clip((headroom / 5) * 100)
    score = _clip(0.5 * headroom_score + 0.5 * strength)
    if tier in ("small", "mid") and strength > 60:
        flags.append(f"{tier.capitalize()}-cap with room to climb tiers")
    return score, tier, flags


def _near_high(m: Dict[str, float]) -> tuple[float, List[str]]:
    flags = []
    dist = m.get("dist_from_52w_high", None)
    if dist is None:
        return 50.0, flags
    score = _clip(100 - dist * 200)
    if dist < 0.05:
        flags.append("At / near 52-week high")
    return score, flags


def compute_ascent(ticker, name, sector, market_cap, metrics):
    strength, f1 = _sustained_strength(metrics)
    volume, f2 = _persistent_volume(metrics)
    tier_score, tier, f3 = _tier_potential(market_cap, strength)
    high_score, f4 = _near_high(metrics)

    composite = _clip(0.35 * strength + 0.30 * volume + 0.20 * tier_score + 0.15 * high_score)

    return AscentScore(
        ticker=ticker, name=name or "", sector=sector or "",
        ascent_score=round(composite, 1), market_cap=market_cap, tier=tier,
        strength_score=round(strength, 1), volume_score=round(volume, 1),
        tier_score=round(tier_score, 1), high_score=round(high_score, 1),
        flags=(f1 + f2 + f3 + f4)[:5],
        metrics={k: round(v, 4) for k, v in metrics.items() if isinstance(v, (int, float))},
    )


def rank_ascent(scores: List[AscentScore], top_n: int = 25) -> List[Dict]:
    ordered = sorted(scores, key=lambda s: s.ascent_score, reverse=True)
    out = []
    for i, s in enumerate(ordered[:top_n], start=1):
        d = s.to_dict()
        d["rank"] = i
        out.append(d)
    return out
