"""
QuantEdge v6.0 — Peer Comparison Router
========================================
GET /api/v6/peers/{ticker} → where a ticker ranks among its sector-bucket peers
on key factors (momentum, trend, accumulation, liquidity), as percentiles.
"""

from __future__ import annotations

import json
from typing import Optional, List, Dict
from fastapi import APIRouter, HTTPException, Request, Depends, Query
from loguru import logger

from auth.cognito_auth import get_optional_user, CognitoUser

router = APIRouter()

# Factors to compare on, with display labels and whether higher = better.
FACTORS = [
    ("mom_3m",          "3M Momentum",      True),
    ("mom_6m",          "6M Momentum",      True),
    ("mom_12_1",        "12M Momentum",     True),
    ("pct_above_ma200", "Above 200D MA",    True),
    ("obv_slope_norm",  "Accumulation",     True),
    ("dist_from_52w_high","Near 52W High",  False),  # lower distance = better → invert
]

# Fundamental factors (from enriched peer_stats). higher=better except valuation multiples.
FUND_FACTORS = [
    ("fund_roic_approx",   "ROIC",           True),
    ("fund_roe",           "ROE",            True),
    ("fund_net_margin",    "Net Margin",     True),
    ("fund_gross_margin",  "Gross Margin",   True),
    ("fund_revenue_growth","Revenue Growth", True),
    ("fund_pe",            "P/E (cheap)",    False),  # lower P/E = better → invert
    ("fund_ps",            "P/S (cheap)",    False),
]


def _percentile(value: float, population: List[float]) -> float:
    if not population:
        return 50.0
    below = sum(1 for x in population if x < value)
    equal = sum(1 for x in population if x == value)
    pct = (below + 0.5 * equal) / len(population) * 100
    return round(pct, 1)


@router.get("/peers/{ticker}")
async def get_peers(
    ticker: str,
    http_request: Request,
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    ticker = ticker.upper().strip()
    if not ticker.replace("-", "").replace(".", "").isalnum() or len(ticker) > 10:
        raise HTTPException(422, "Invalid ticker")

    store = getattr(http_request.app.state, "peer_store", None)
    if store is None:
        raise HTTPException(503, "Peer data not available yet")

    data = await store.get_peers(ticker)
    if not data.get("available"):
        return {"data": {"available": False, "reason": data.get("reason", "no data")}, "cached": False}

    me = data["me"]
    peers = data["peers"]
    me_factors = json.loads(me["factors"]) if isinstance(me["factors"], str) else (me["factors"] or {})

    # Build percentile per factor
    factor_results = []
    for key, label, higher_better in FACTORS:
        me_val = me_factors.get(key)
        if me_val is None:
            continue
        pop = []
        for p in peers:
            pf = json.loads(p["factors"]) if isinstance(p["factors"], str) else (p["factors"] or {})
            v = pf.get(key)
            if v is not None:
                pop.append(float(v))
        if len(pop) < 3:
            continue
        pct = _percentile(float(me_val), pop)
        if not higher_better:
            pct = round(100 - pct, 1)  # invert so higher percentile is always "better"
        factor_results.append({
            "key": key, "label": label,
            "value": round(float(me_val), 2),
            "percentile": pct,
            "peer_median": round(sorted(pop)[len(pop) // 2], 2),
        })

    # Fundamental factor percentiles (quality / profitability / growth / valuation vs peers)
    fund_factor_results = []
    fund_rank = {}
    for key, label, higher_better in FUND_FACTORS:
        me_val = me_factors.get(key)
        if me_val is None:
            continue
        pop = []
        for p in peers:
            pf = json.loads(p["factors"]) if isinstance(p["factors"], str) else (p["factors"] or {})
            v = pf.get(key)
            if v is not None:
                pop.append(float(v))
        if len(pop) < 3:
            continue
        pct = _percentile(float(me_val), pop)
        if not higher_better:
            pct = round(100 - pct, 1)
        # explicit rank: #N of M (1 = best)
        if higher_better:
            better = sum(1 for v in pop if v > float(me_val))
        else:
            better = sum(1 for v in pop if v < float(me_val))
        rank = better + 1
        fund_rank[key] = {"rank": rank, "of": len(pop) + 1}
        fund_factor_results.append({
            "key": key, "label": label,
            "value": round(float(me_val), 3),
            "percentile": pct,
            "peer_median": round(sorted(pop)[len(pop) // 2], 3),
            "rank": rank, "of": len(pop) + 1,
        })

    # Peer table: same-bucket names with momentum + FUNDAMENTALS + cap
    peer_rows = []
    for p in peers:
        pf = json.loads(p["factors"]) if isinstance(p["factors"], str) else (p["factors"] or {})
        peer_rows.append({
            "ticker": p["ticker"],
            "name": (p["name"] or "")[:32],
            "market_cap": p["market_cap"],
            "mom_3m": pf.get("mom_3m"),
            "mom_6m": pf.get("mom_6m"),
            "pct_above_ma200": pf.get("pct_above_ma200"),
            "roic": pf.get("fund_roic_approx"),
            "net_margin": pf.get("fund_net_margin"),
            "gross_margin": pf.get("fund_gross_margin"),
            "revenue_growth": pf.get("fund_revenue_growth"),
            "pe": pf.get("fund_pe"),
            "is_me": p["ticker"] == ticker,
        })
    peer_rows.sort(key=lambda r: (r["mom_3m"] is None, -(r["mom_3m"] or 0)))

    payload = {
        "available": True,
        "ticker": ticker,
        "name": me["name"],
        "bucket": data["bucket"],
        "peer_count": len(peers),
        "scan_time": data["scan_time"],
        "factors": factor_results,
        "fund_factors": fund_factor_results,
        "me_fundamentals": {
            "roic": me_factors.get("fund_roic_approx"),
            "net_margin": me_factors.get("fund_net_margin"),
            "gross_margin": me_factors.get("fund_gross_margin"),
            "revenue_growth": me_factors.get("fund_revenue_growth"),
            "pe": me_factors.get("fund_pe"),
        },
        "peers": peer_rows,
    }
    return {"data": payload, "cached": False}
