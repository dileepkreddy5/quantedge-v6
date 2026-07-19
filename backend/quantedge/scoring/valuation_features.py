"""Valuation feature assembler — real financials + price -> CAPM beta -> all
valuation models (bull/base/bear DCF, EPV, Graham, residual income, reverse DCF,
multiples) -> margin of safety + targets. Reuses hybrid financial data (one source).
"""
from __future__ import annotations
import math
from typing import Optional, Dict, List, Any
from quantedge.scoring.valuation_models import (dcf_valuation, epv_valuation,
    graham_number, residual_income_value, reverse_dcf, multiples_suite, _f, _sd)

def compute_capm_beta(stock_returns: List[float], market_returns: List[float]) -> Optional[float]:
    n = min(len(stock_returns), len(market_returns))
    if n < 30: return None
    s = stock_returns[-n:]; m = market_returns[-n:]
    ms = sum(s)/n; mm = sum(m)/n
    cov = sum((s[i]-ms)*(m[i]-mm) for i in range(n))/n
    var = sum((x-mm)**2 for x in m)/n
    if var == 0: return None
    beta = cov/var
    if beta < 0.1: return None  # degenerate; caller falls back to 1.0
    return round(max(0.1, min(4.0, beta)), 3)

def _ttm(m, k):
    vals=[q[k] for q in m[-4:] if q.get(k) is not None]
    return sum(vals) if len(vals)==4 else None
def _latest_ttm(m,k):
    for i in range(len(m)-1,2,-1):
        vals=[m[j].get(k) for j in range(i-3,i+1)]
        if all(v is not None for v in vals): return sum(vals)
    return None

def compute_valuation_features(merged, fin_features, price, market_cap, wacc_dict, beta=None):
    f={}
    if not merged or not price: return f
    cur = merged[-1]
    shares = _f(cur.get("diluted_shares"))
    if not shares or shares<=0: return f
    rev_ttm = _ttm(merged,"revenue") or _latest_ttm(merged,"revenue")
    fcf_ttm = (fin_features.get("fcf_margin") * rev_ttm) if (fin_features.get("fcf_margin") and rev_ttm) else None
    ni_ttm = _ttm(merged,"net_income") or _latest_ttm(merged,"net_income")
    ebitda = fin_features.get("ebitda")
    eps = _sd(ni_ttm, shares)
    bvps = fin_features.get("book_value_per_share")
    net_debt = fin_features.get("net_debt") or 0.0
    ev = fin_features.get("enterprise_value") or (market_cap + net_debt)
    rev_growth = fin_features.get("revenue_cagr_5y") or fin_features.get("revenue_growth") or 0.05
    wacc = wacc_dict.get("mid", 0.09)

    if fcf_ttm and fcf_ttm>0:
        base_g = max(0.02, min(0.20, rev_growth))
        # Bull/base/bear vary growth + WACC, but bull doesn't stack all-optimistic
        # (that produces absurd valuations). Terminal-growth spread guarded in dcf_valuation.
        bull = dcf_valuation(fcf_ttm, min(base_g*1.3,0.25), wacc_dict.get("low",wacc-0.01), 0.028, shares, net_debt)
        base = dcf_valuation(fcf_ttm, base_g, wacc, 0.025, shares, net_debt)
        bear = dcf_valuation(fcf_ttm, base_g*0.6, wacc_dict.get("high",wacc+0.01), 0.02, shares, net_debt)
        f["dcf_bull"]=bull.get("intrinsic_per_share")
        f["dcf_base"]=base.get("intrinsic_per_share")
        f["dcf_bear"]=bear.get("intrinsic_per_share")
        vals=[v for v in [f["dcf_bull"],f["dcf_base"],f["dcf_bear"]] if v is not None]
        if len(vals)==3:
            f["dcf_weighted"]=round(0.25*f["dcf_bull"]+0.5*f["dcf_base"]+0.25*f["dcf_bear"],2)
        f["dcf_terminal_pct"]=base.get("terminal_pct")
        f["reverse_dcf_implied_growth"]=reverse_dcf(price, fcf_ttm, wacc, shares)

    f["epv_per_share"]=epv_valuation(ni_ttm, wacc, shares, net_debt)
    f["graham_number"]=graham_number(eps, bvps)
    f["residual_income_value"]=residual_income_value(bvps, eps, wacc)

    rev_ps = _sd(rev_ttm, shares); fcf_ps = _sd(fcf_ttm, shares)
    mult = multiples_suite(price, eps, rev_ps, bvps, ev, ebitda, fcf_ps, rev_growth)
    for k,v in mult.items(): f[f"mult_{k}"]=v

    fair = f.get("dcf_weighted") or f.get("dcf_base")
    if fair and price:
        f["fair_value"]=fair
        f["margin_of_safety"]=round((fair-price)/fair,4)
        f["upside_to_fair"]=round((fair-price)/price,4)
        f["buy_zone"]=round(fair*0.7,2); f["sell_zone"]=round(fair*1.15,2)
        f["price_to_fair"]=round(price/fair,3)
    intrinsics=[v for v in [f.get("dcf_weighted"),f.get("epv_per_share"),
                f.get("graham_number"),f.get("residual_income_value")] if v and v>0]
    if intrinsics:
        f["intrinsic_consensus"]=round(sum(intrinsics)/len(intrinsics),2)
        f["consensus_upside"]=round((f["intrinsic_consensus"]-price)/price,4)

    # --- ASSUMPTIONS PANEL (expose what drives the DCF) ---
    ni_ttm2 = _ttm(merged,"net_income") or _latest_ttm(merged,"net_income")
    f["assumption_revenue_cagr"]=round(rev_growth,4)
    f["assumption_wacc"]=round(wacc,4)
    f["assumption_terminal_growth"]=0.025
    f["assumption_operating_margin"]=fin_features.get("operating_margin")
    f["assumption_tax_rate"]=fin_features.get("effective_tax_rate")
    f["assumption_fcf_margin"]=fin_features.get("fcf_margin")
    f["assumption_forecast_years"]=7
    f["assumption_beta"]=beta if beta is not None else wacc_dict.get("beta_used")

    # --- BULL/BASE/BEAR PROBABILITIES + CONFIDENCE RANGE ---
    if f.get("dcf_bull") and f.get("dcf_base") and f.get("dcf_bear"):
        f["case_probabilities"]={"bear":0.25,"base":0.50,"bull":0.25}
        lo=min(f["dcf_bear"],f["dcf_base"]); hi=max(f["dcf_bull"],f["dcf_base"])
        f["confidence_range_low"]=round(f["dcf_bear"]+(f["dcf_base"]-f["dcf_bear"])*0.3,2)
        f["confidence_range_high"]=round(f["dcf_base"]+(f["dcf_bull"]-f["dcf_base"])*0.7,2)

    # --- MODEL AGREEMENT (dispersion + consensus across all methods) ---
    method_vals=[v for v in [f.get("dcf_weighted"),f.get("epv_per_share"),f.get("graham_number"),
                 f.get("residual_income_value")] if v and v>0]
    if len(method_vals)>=3:
        mean_v=sum(method_vals)/len(method_vals)
        var=sum((x-mean_v)**2 for x in method_vals)/len(method_vals)
        cv=(var**0.5)/mean_v if mean_v>0 else None
        f["model_dispersion_cv"]=round(cv,3) if cv is not None else None  # lower = methods agree
        below=sum(1 for v in method_vals if v<price)
        f["model_consensus_overvalued"]=round(below/len(method_vals),3)  # frac saying overvalued
        f["model_agreement_score"]=round(1-cv,3) if (cv is not None and cv<1) else 0.0

    f["beta_used"]=beta if beta is not None else wacc_dict.get("beta_used")
    f["wacc_used"]=wacc; f["current_price"]=price
    return f
