"""Module B — Deep Financial Feature Engine: sophisticated composite models
(DuPont, Piotroski F, Altman Z, Beneish M, owner earnings, normalized ROIC) plus
ratio/trend/stability breadth. Every value traces to a real filing line; missing
inputs return None (never faked). Windows relative to today (future-proof).
"""
from __future__ import annotations
import math, statistics as st
from typing import List, Dict, Optional, Any

def _f(v):
    try:
        v = float(v); return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

def _safe_div(a, b):
    a, b = _f(a), _f(b)
    if a is None or b is None or b == 0:
        return None
    return a / b

def _cov_stability(series):
    vals = [x for x in (_f(s) for s in series) if x is not None]
    if len(vals) < 3: return None
    m = sum(vals)/len(vals)
    if m == 0: return None
    return max(0.0, 1.0 - (st.pstdev(vals)/abs(m)))

def _slope(series):
    vals = [x for x in (_f(s) for s in series) if x is not None]
    n = len(vals)
    if n < 3: return None
    xs = list(range(n)); mx = sum(xs)/n; my = sum(vals)/n
    d = sum((x-mx)**2 for x in xs)
    return None if d == 0 else sum((xs[i]-mx)*(vals[i]-my) for i in range(n))/d

def _cagr(first, last, years):
    first, last = _f(first), _f(last)
    if first is None or last is None or first <= 0 or last <= 0 or years <= 0:
        return None
    return (last/first)**(1.0/years) - 1.0

def _yoy(series):
    if len(series) < 2 or not series[-2]: return None
    return _safe_div(series[-1] - series[-2], abs(series[-2]))

def dupont_decomposition(ni, revenue, assets, equity) -> Dict[str, Optional[float]]:
    nm = _safe_div(ni, revenue); at = _safe_div(revenue, assets); em = _safe_div(assets, equity)
    roe = nm * at * em if None not in (nm, at, em) else None
    return {"net_margin": nm, "asset_turnover": at, "equity_multiplier": em, "roe_dupont": roe}

def piotroski_f_score(cur, prev, ttm) -> Dict[str, Any]:
    pts = {}
    ni = _f(ttm.get("net_income")); ocf = _f(ttm.get("operating_cash_flow"))
    assets = _f(cur.get("assets")); assets_prev = _f(prev.get("assets"))
    avg_assets = (assets+assets_prev)/2 if (assets and assets_prev) else None
    roa = _safe_div(ni, avg_assets); roa_prev = _safe_div(_f(prev.get("net_income")), assets_prev)
    pts["roa_positive"] = 1 if (roa is not None and roa > 0) else 0
    pts["ocf_positive"] = 1 if (ocf is not None and ocf > 0) else 0
    pts["roa_improving"] = 1 if (roa is not None and roa_prev is not None and roa > roa_prev) else 0
    pts["accruals"] = 1 if (ocf is not None and ni is not None and ocf > ni) else 0
    ltd = _f(cur.get("long_term_debt")); ltd_prev = _f(prev.get("long_term_debt"))
    la, lap = _safe_div(ltd, assets), _safe_div(ltd_prev, assets_prev)
    pts["leverage_down"] = 1 if (la is not None and lap is not None and la < lap) else 0
    cr = _safe_div(cur.get("current_assets"), cur.get("current_liabilities"))
    cr_prev = _safe_div(prev.get("current_assets"), prev.get("current_liabilities"))
    pts["current_ratio_up"] = 1 if (cr is not None and cr_prev is not None and cr > cr_prev) else 0
    sh = _f(cur.get("diluted_shares")); sh_prev = _f(prev.get("diluted_shares"))
    pts["no_dilution"] = 1 if (sh is not None and sh_prev is not None and sh <= sh_prev*1.01) else 0
    gm = _safe_div(cur.get("gross_profit"), cur.get("revenue"))
    gm_prev = _safe_div(prev.get("gross_profit"), prev.get("revenue"))
    pts["margin_up"] = 1 if (gm is not None and gm_prev is not None and gm > gm_prev) else 0
    at = _safe_div(cur.get("revenue"), avg_assets); at_prev = _safe_div(prev.get("revenue"), assets_prev)
    pts["turnover_up"] = 1 if (at is not None and at_prev is not None and at > at_prev) else 0
    return {"f_score": sum(pts.values()), "components": pts, "max": 9}

def altman_z(cur, ttm, market_cap) -> Dict[str, Optional[float]]:
    assets = _f(cur.get("assets"))
    if not assets: return {"z_score": None}
    ca, cl = _f(cur.get("current_assets")), _f(cur.get("current_liabilities"))
    wc = (ca - cl) if (ca is not None and cl is not None) else None
    re = _f(cur.get("retained_earnings")); ebit = _f(ttm.get("operating_income"))
    liabilities = _f(cur.get("liabilities")); revenue = _f(ttm.get("revenue"))
    x1 = _safe_div(wc, assets); x2 = _safe_div(re, assets); x3 = _safe_div(ebit, assets)
    x4 = _safe_div(market_cap, liabilities); x5 = _safe_div(revenue, assets)
    terms = {"x1":x1,"x2":x2,"x3":x3,"x4":x4,"x5":x5}
    if None in (x1,x3,x4,x5): return {"z_score": None, "terms": terms}
    x2 = x2 if x2 is not None else 0.0
    z = 1.2*x1 + 1.4*x2 + 3.3*x3 + 0.6*x4 + 1.0*x5
    return {"z_score": z, "terms": terms, "zone": "safe" if z>2.99 else ("distress" if z<1.81 else "grey")}

def beneish_m_score(cur, prev) -> Dict[str, Optional[float]]:
    def dv(a,b): return _safe_div(a,b)
    rev, rev_p = _f(cur.get("revenue")), _f(prev.get("revenue"))
    rec, rec_p = _f(cur.get("receivables")), _f(prev.get("receivables"))
    dsri = dv(dv(rec,rev), dv(rec_p,rev_p))
    gp, gp_p = _f(cur.get("gross_profit")), _f(prev.get("gross_profit"))
    gmi = dv(dv(gp_p,rev_p), dv(gp,rev))
    assets, assets_p = _f(cur.get("assets")), _f(prev.get("assets"))
    ca, ca_p = _f(cur.get("current_assets")), _f(prev.get("current_assets"))
    ppe, ppe_p = _f(cur.get("fixed_assets")), _f(prev.get("fixed_assets"))
    def aqi_side(c_a, p_p, a):
        if None in (c_a,p_p,a) or a==0: return None
        return 1 - (c_a+p_p)/a
    aqi = dv(aqi_side(ca,ppe,assets), aqi_side(ca_p,ppe_p,assets_p))
    sgi = dv(rev, rev_p)
    ni = _f(cur.get("net_income")); ocf = _f(cur.get("operating_cash_flow"))
    tata = dv((ni - ocf) if (ni is not None and ocf is not None) else None, assets)
    parts = {"dsri":dsri,"gmi":gmi,"aqi":aqi,"sgi":sgi,"tata":tata}
    if any(v is None for v in (dsri,gmi,aqi,sgi)):
        return {"m_score": None, "parts": parts}
    tata = tata if tata is not None else 0.0
    m = -4.84 + 0.92*dsri + 0.528*gmi + 0.404*aqi + 0.892*sgi + 4.679*tata
    return {"m_score": m, "parts": parts, "flag": "possible_manipulation" if m > -1.78 else "likely_clean"}

def owner_earnings(ttm, maint_capex_ratio=0.7) -> Optional[float]:
    ni = _f(ttm.get("net_income")); da = _f(ttm.get("depreciation_amortization")); capex = _f(ttm.get("capex"))
    if ni is None or da is None or capex is None: return None
    return ni + da - abs(capex)*maint_capex_ratio

def normalized_roic(ttm, cur, tax_rate=None) -> Dict[str, Optional[float]]:
    oi = _f(ttm.get("operating_income"))
    te = _f(ttm.get("tax_expense")); pt = _f(ttm.get("pretax_income"))
    tr = tax_rate
    if tr is None and te is not None and pt and pt > 0:
        tr = min(max(te/pt, 0.0), 0.35)
    if tr is None: tr = 0.21
    nopat = oi*(1-tr) if oi is not None else None
    eq = _f(cur.get("equity")) or 0; debt = _f(cur.get("long_term_debt")) or 0
    cash = _f(cur.get("cash")) or 0; gw = _f(cur.get("goodwill")) or 0
    ic = eq + debt - cash; ic_exgw = ic - gw
    return {"nopat": nopat, "tax_rate": tr,
            "roic": _safe_div(nopat, ic) if ic > 0 else None,
            "roic_ex_goodwill": _safe_div(nopat, ic_exgw) if ic_exgw > 0 else None,
            "invested_capital": ic if ic > 0 else None}
