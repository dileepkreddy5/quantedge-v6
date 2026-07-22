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


@router.get("/peers/{ticker}/relative")
async def get_peer_relative(
    ticker: str,
    http_request: Request,
    current_user: Optional[CognitoUser] = Depends(get_optional_user),
):
    """
    Cumulative return of the ticker against its peer group over time.

    Every other figure on the Peers tab is a snapshot: today's margin, today's
    percentile. None of them say whether the position is improving or eroding.
    A stock up 40% while its industry is up 45% is losing ground, and no
    single-name chart shows that.

    Returns the target's rebased cumulative return alongside the peer median and
    the 25th/75th percentile band, plus per-window relative figures.
    """
    ticker = ticker.upper().strip()
    if not ticker.replace("-", "").replace(".", "").isalnum() or len(ticker) > 10:
        raise HTTPException(422, "Invalid ticker")

    store = getattr(http_request.app.state, "peer_store", None)
    if store is None:
        raise HTTPException(503, "peer store unavailable")
    meta = await store.get_peers(ticker)
    if not meta.get("available", True) or not meta.get("peers"):
        return {"data": {"available": False, "reason": meta.get("reason", "no peer group")}}

    tickers = sorted({p["ticker"] for p in meta["peers"]} | {ticker})
    pool = http_request.app.state.db
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT ticker, d, c FROM daily_bars WHERE ticker = ANY($1::text[]) ORDER BY d",
            list(tickers))
    if not rows:
        return {"data": {"available": False, "reason": "no price history"}}

    import pandas as _pd
    df = _pd.DataFrame([(r["ticker"], r["d"], float(r["c"])) for r in rows],
                       columns=["ticker", "d", "c"])
    wide = df.pivot(index="d", columns="ticker", values="c").sort_index().ffill()
    if ticker not in wide.columns or len(wide) < 30:
        return {"data": {"available": False, "reason": "insufficient overlapping history"}}

    WINDOWS = [("1M", 21), ("3M", 63), ("6M", 126), ("1Y", 252), ("2Y", 504)]
    windows = []
    for label, n in WINDOWS:
        # daily_bars holds ~2 years, so the 2Y window would be dropped for
        # being four sessions short. Use whatever history exists and let the
        # caller see n_sessions rather than silently omitting the window.
        n = min(n, len(wide) - 1)
        if n < 15:
            continue
        seg = wide.iloc[-(n + 1):]
        # A peer that listed part-way through the window would rebase off a
        # forward-filled value and drag the median toward zero. Require a real
        # price at the window start.
        valid = [c for c in seg.columns if _pd.notna(seg[c].iloc[0]) and _pd.notna(seg[c].iloc[-1])]
        if ticker not in valid or len(valid) < 3:
            continue
        rets = {c: float(seg[c].iloc[-1] / seg[c].iloc[0] - 1) for c in valid}
        peer_rets = sorted(v for k, v in rets.items() if k != ticker)
        med = peer_rets[len(peer_rets) // 2] if peer_rets else None
        windows.append({
            "window": label,
            "target_pct": round(rets[ticker] * 100, 2),
            "peer_median_pct": round(med * 100, 2) if med is not None else None,
            "relative_pts": round((rets[ticker] - med) * 100, 2) if med is not None else None,
            "n_peers": len(peer_rets),
        })

    # Series for the chart: longest window that has data.
    span = min(len(wide) - 1, 504)
    seg = wide.iloc[-(span + 1):]
    valid = [c for c in seg.columns if _pd.notna(seg[c].iloc[0])]
    base = seg[valid].iloc[0]
    rebased = (seg[valid] / base - 1) * 100
    peer_cols = [c for c in valid if c != ticker]
    series = []
    for d, row in rebased.iterrows():
        pv = sorted(float(row[c]) for c in peer_cols if _pd.notna(row[c]))
        # Quartiles off fewer than four names describe nothing.
        if len(pv) < 4:
            continue
        series.append({
            "d": str(d),
            "t": round(float(row[ticker]), 2) if ticker in row and _pd.notna(row[ticker]) else None,
            "med": round(pv[len(pv) // 2], 2),
            "p25": round(pv[int(len(pv) * 0.25)], 2),
            "p75": round(pv[min(len(pv) - 1, int(len(pv) * 0.75))], 2),
        })

    # Per-ticker series so the client can let the user pick which rivals to
    # plot without another round trip. Small payload: ~8 tickers x 501 points.
    by_ticker = {}
    px_ticker = {}
    for c in valid:
        by_ticker[c] = [None if _pd.isna(v) else round(float(v), 2) for v in rebased[c].tolist()]
        px_ticker[c] = [None if _pd.isna(v) else round(float(v), 2) for v in seg[c].tolist()]
    caps = {p["ticker"]: p.get("market_cap") for p in meta["peers"]}
    roster = sorted(
        [{"ticker": c, "market_cap": caps.get(c),
          "total_pct": by_ticker[c][-1] if by_ticker[c] else None} for c in valid if c != ticker],
        key=lambda r: -(r["market_cap"] or 0))

    return {"data": {
        "available": True,
        "ticker": ticker,
        "group_label": meta.get("bucket"),
        "group_kind": meta.get("group_kind"),
        "dates": [str(d) for d in rebased.index],
        "by_ticker": by_ticker,
        "px_ticker": px_ticker,
        "roster": roster,
        "n_peers": len(peer_cols),
        "n_sessions": len(series),
        "windows": windows,
        "note": ("Cumulative return rebased to zero at the window start. The band "
                 "spans the 25th to 75th percentile of the peer group. With a small "
                 "group the median moves in steps and the band is wide — read the "
                 "direction of the gap, not its precise width."),
    }}


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
        # Exclude the target from its own population — `peers` includes it.
        pop = []
        for p in peers:
            if (p["ticker"] or "").upper() == ticker:
                continue
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
        # `peers` already contains the target — that is what is_me marks — so the
        # target must be excluded from its own comparison population, and the
        # denominator must not add it back. Previously pop included the target
        # AND of was len(pop)+1, which reported "1 of 9" for a group of eight
        # and biased every percentile toward the middle by counting the target
        # as one of the names it was being ranked against.
        pop = []
        for p in peers:
            if (p["ticker"] or "").upper() == ticker:
                continue
            pf = json.loads(p["factors"]) if isinstance(p["factors"], str) else (p["factors"] or {})
            v = pf.get(key)
            if v is not None:
                pop.append(float(v))
        if len(pop) < 3:
            continue
        pct = _percentile(float(me_val), pop)
        if not higher_better:
            pct = round(100 - pct, 1)
        # explicit rank: #N of M (1 = best), M = peers with data + the target
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
        "group_kind": data.get("group_kind"),
        "broad_sector": data.get("broad_sector"),
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
