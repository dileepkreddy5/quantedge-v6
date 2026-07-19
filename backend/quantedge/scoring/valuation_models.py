"""Valuation Intelligence — model engine. Intrinsic value via DCF (multi-case),
EPV, Graham, residual income; relative multiples; reverse DCF; margin of safety.
All from real financial data + WACC. Missing -> None. Rolling window vs today.
"""
from __future__ import annotations
import math
from typing import Optional, Dict, List, Any

def _f(v):
    try: v=float(v); return v if math.isfinite(v) else None
    except: return None
def _sd(a,b):
    a,b=_f(a),_f(b); return a/b if (a is not None and b not in (None,0)) else None

def dcf_valuation(fcf_base, growth_rate, wacc, terminal_growth, shares,
                  net_debt=0.0, years=10) -> Dict[str, Any]:
    fcf_base, growth_rate, wacc, tg, shares = (_f(fcf_base),_f(growth_rate),_f(wacc),
                                                _f(terminal_growth),_f(shares))
    if None in (fcf_base, wacc, shares) or shares<=0: return {"intrinsic_per_share": None}
    if wacc is None or (tg is not None and wacc<=tg): return {"intrinsic_per_share": None}
    g = growth_rate if growth_rate is not None else 0.05
    g = min(g, 0.25)  # cap explicit growth — no firm compounds >25%/yr for a decade
    tg = tg if tg is not None else 0.025
    tg = min(tg, wacc - 0.03)  # terminal growth floored >=3% below WACC (prevents explosion)
    pv_sum = 0.0; fcf = fcf_base; projection=[]
    for yr in range(1, years+1):
        yr_growth = g + (tg - g) * (yr-1)/(years-1) if years>1 else tg
        fcf = fcf * (1 + yr_growth)
        pv = fcf / ((1+wacc)**yr); pv_sum += pv
        projection.append({"year":yr,"fcf":round(fcf,0),"growth":round(yr_growth,4),"pv":round(pv,0)})
    tv = fcf * (1+tg) / (wacc - tg)
    pv_tv = tv / ((1+wacc)**years)
    enterprise = pv_sum + pv_tv
    equity = enterprise - (net_debt or 0)
    return {"intrinsic_per_share": round(equity/shares, 2),
            "enterprise_value": round(enterprise,0), "equity_value": round(equity,0),
            "pv_explicit": round(pv_sum,0), "pv_terminal": round(pv_tv,0),
            "terminal_pct": round(pv_tv/enterprise,3) if enterprise else None,
            "wacc_used": wacc, "terminal_growth": tg, "projection": projection}

def epv_valuation(normalized_earnings, wacc, shares, net_debt=0.0) -> Optional[float]:
    ne, wacc, shares = _f(normalized_earnings), _f(wacc), _f(shares)
    if None in (ne, wacc, shares) or wacc<=0 or shares<=0: return None
    return round((ne/wacc - (net_debt or 0))/shares, 2)

def graham_number(eps, bvps) -> Optional[float]:
    eps, bvps = _f(eps), _f(bvps)
    if eps is None or bvps is None or eps<=0 or bvps<=0: return None
    return round(math.sqrt(22.5 * eps * bvps), 2)

def residual_income_value(bvps, eps, wacc, growth=0.03, years=10) -> Optional[float]:
    bvps, eps, wacc = _f(bvps), _f(eps), _f(wacc)
    if None in (bvps, eps, wacc) or bvps<=0 or wacc<=0: return None
    roe = eps/bvps; bv = bvps; total_ri=0.0
    for yr in range(1, years+1):
        ri = (roe - wacc) * bv
        total_ri += ri / ((1+wacc)**yr)
        bv = bv * (1 + roe*(1-0.4))
    return round(bvps + total_ri, 2)

def reverse_dcf(price, fcf_base, wacc, shares, terminal_growth=0.025, years=10) -> Optional[float]:
    price, fcf_base, wacc, shares = _f(price),_f(fcf_base),_f(wacc),_f(shares)
    if None in (price,fcf_base,wacc,shares) or shares<=0 or fcf_base<=0: return None
    target_equity = price * shares
    def implied_equity(g):
        pv=0.0; fcf=fcf_base
        for yr in range(1,years+1):
            yg = g + (terminal_growth-g)*(yr-1)/(years-1) if years>1 else terminal_growth
            fcf*= (1+yg); pv += fcf/((1+wacc)**yr)
        tv = fcf*(1+terminal_growth)/(wacc-terminal_growth)
        return pv + tv/((1+wacc)**years)
    lo, hi = -0.20, 0.60
    for _ in range(60):
        mid=(lo+hi)/2
        if implied_equity(mid) < target_equity: lo=mid
        else: hi=mid
    return round((lo+hi)/2, 4)

def multiples_suite(price, eps, revenue_ps, bvps, ev, ebitda, fcf_ps, growth) -> Dict[str,Optional[float]]:
    pe = _sd(price, eps)
    return {"pe": round(pe,2) if pe else None,
            "peg": round(pe/(growth*100),2) if (pe and growth and growth>0) else None,
            "ps": round(_sd(price, revenue_ps),2) if _sd(price,revenue_ps) else None,
            "pb": round(_sd(price, bvps),2) if _sd(price,bvps) else None,
            "ev_ebitda": round(_sd(ev, ebitda),2) if _sd(ev,ebitda) else None,
            "p_fcf": round(_sd(price, fcf_ps),2) if _sd(price,fcf_ps) else None}
