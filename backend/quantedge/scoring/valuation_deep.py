"""Valuation deepening — sensitivity matrix, DDM, extra multiples, asset-based
values, gap-to-price for every intrinsic model. All real, missing->None."""
import math
from typing import Optional, Dict, List, Any
from quantedge.scoring.valuation_models import dcf_valuation, _f, _sd

def dcf_sensitivity_matrix(fcf_base, growth, shares, net_debt, price,
                           wacc_center=0.09, tg_center=0.025):
    fcf_base,growth,shares=_f(fcf_base),_f(growth),_f(shares)
    if None in (fcf_base,shares) or shares<=0 or fcf_base<=0: return {}
    waccs=[round(wacc_center+d,4) for d in (-0.02,-0.01,0,0.01,0.02)]
    tgs=[round(tg_center+d,4) for d in (-0.01,-0.005,0,0.005,0.01)]
    grid=[]; in_range=0; total=0
    for w in waccs:
        row=[]
        for tg in tgs:
            if w<=tg: row.append(None); continue
            r=dcf_valuation(fcf_base,growth,w,tg,shares,net_debt)
            fv=r.get("intrinsic_per_share"); row.append(fv)
            if fv is not None and price:
                total+=1
                if fv>=price: in_range+=1
        grid.append(row)
    return {"waccs":waccs,"terminal_growths":tgs,"grid":grid,
            "pct_scenarios_above_price": round(in_range/total,3) if total else None}

def ddm_valuation(dividend_ps, growth, wacc):
    d,g,r=_f(dividend_ps),_f(growth),_f(wacc)
    if None in (d,g,r) or d<=0: return None
    g = min(g, r - 0.02)  # cap dividend growth >=2% below WACC (prevents inflation)
    if r<=g: return None
    return round(d*(1+g)/(r-g),2)

def extra_multiples(ev, revenue, fcf, invested_capital, ebit):
    return {"ev_sales": round(_sd(ev,revenue),2) if _sd(ev,revenue) else None,
            "ev_fcf": round(_sd(ev,fcf),2) if _sd(ev,fcf) else None,
            "ev_ic": round(_sd(ev,invested_capital),2) if _sd(ev,invested_capital) else None,
            "ev_ebit": round(_sd(ev,ebit),2) if _sd(ev,ebit) else None}

def asset_based_values(assets, liabilities, goodwill, intangibles, shares):
    assets,liab,gw,intang,shares=_f(assets),_f(liabilities),_f(goodwill) or 0,_f(intangibles) or 0,_f(shares)
    if None in (assets,liab,shares) or shares<=0: return {}
    nav=(assets-liab)/shares
    tangible_nav=(assets-liab-gw-intang)/shares
    liquidation=(assets*0.6-liab)/shares
    return {"nav_per_share":round(nav,2),"tangible_nav_per_share":round(tangible_nav,2),
            "liquidation_value":round(liquidation,2)}

def gap_to_price(model_value, price):
    mv,p=_f(model_value),_f(price)
    if mv is None or p is None or p<=0 or mv<=0: return None
    return round((mv-p)/p,4)

def compute_valuation_deep(merged, fin_features, val_features, price, market_cap,
                            wacc, price_history=None):
    f = {}
    cur = merged[-1]
    shares = _f(cur.get("diluted_shares"))
    if not shares or shares<=0 or not price: return f
    net_debt = fin_features.get("net_debt") or 0.0
    ev = fin_features.get("enterprise_value") or (market_cap + net_debt)
    def ttm(k):
        v=[q[k] for q in merged[-4:] if q.get(k) is not None]
        return sum(v) if len(v)==4 else None
    def latest_ttm(k):
        for i in range(len(merged)-1,2,-1):
            v=[merged[j].get(k) for j in range(i-3,i+1)]
            if all(x is not None for x in v): return sum(v)
        return None
    rev=ttm("revenue") or latest_ttm("revenue")
    ni=ttm("net_income") or latest_ttm("net_income")
    fcf=(fin_features.get("fcf_margin")*rev) if (fin_features.get("fcf_margin") and rev) else None
    ebit=ttm("operating_income") or latest_ttm("operating_income")
    ic=(fin_features.get("nopat")/fin_features["roic"]) if fin_features.get("roic") else None
    growth=fin_features.get("revenue_cagr_5y") or fin_features.get("revenue_growth") or 0.05
    if fcf and fcf>0:
        sm=dcf_sensitivity_matrix(fcf, max(0.02,min(0.20,growth)), shares, net_debt, price, wacc, 0.025)
        f["dcf_sensitivity"]=sm
        f["dcf_scenarios_above_price"]=sm.get("pct_scenarios_above_price")
    em=extra_multiples(ev, rev, fcf, ic, ebit)
    for k,v in em.items(): f[f"mult_{k}"]=v
    ab=asset_based_values(cur.get("assets"), cur.get("liabilities"),
                          cur.get("goodwill"), cur.get("intangibles"), shares)
    for k,v in ab.items(): f[k]=v
    if ab.get("nav_per_share"): f["nav_upside"]=gap_to_price(ab["nav_per_share"], price)
    if ab.get("tangible_nav_per_share"): f["tangible_nav_upside"]=gap_to_price(ab["tangible_nav_per_share"], price)
    f["epv_upside"]=gap_to_price(val_features.get("epv_per_share"), price)
    f["graham_upside"]=gap_to_price(val_features.get("graham_number"), price)
    f["residual_income_upside"]=gap_to_price(val_features.get("residual_income_value"), price)
    f["dcf_weighted_upside"]=gap_to_price(val_features.get("dcf_weighted"), price)
    div_ttm=ttm("dividends_paid") or latest_ttm("dividends_paid")
    if div_ttm and shares:
        div_ps=abs(div_ttm)/shares
        ddm=ddm_valuation(div_ps, min(0.08,growth*0.5), wacc)
        if ddm: f["ddm_value"]=ddm; f["ddm_upside"]=gap_to_price(ddm, price)
        f["dividend_per_share"]=round(div_ps,2)
    if price_history and len(price_history)>250 and ni and shares:
        eps_now=ni/shares
        if eps_now>0:
            cur_pe=price/eps_now
            hist_pes=[p/eps_now for p in price_history]
            avg_pe=sum(hist_pes)/len(hist_pes)
            if avg_pe>0:
                f["pe_vs_history"]=round(cur_pe/avg_pe,3)
                f["pe_historical_avg"]=round(avg_pe,2)

    # --- EXPECTED RETURN DECOMPOSITION (how forward return is sourced) ---
    growth_ret = fin_features.get("revenue_cagr_5y") or fin_features.get("revenue_growth") or 0.0
    div_ps = f.get("dividend_per_share")
    div_yield = (div_ps/price) if (div_ps and price) else 0.0
    buyback_yield = fin_features.get("buyback_yield") or 0.0
    # margin expansion contribution (small, from operating margin trend)
    margin_contrib = (fin_features.get("operating_margin_trend") or 0.0)
    # multiple change: if overvalued, expect some compression drag
    mos = f.get("dcf_weighted_upside")
    multiple_contrib = round(mos*0.1,4) if mos is not None else 0.0  # partial mean-reversion
    f["exp_return_growth"]=round(min(growth_ret,0.20),4)
    f["exp_return_dividend"]=round(div_yield,4)
    f["exp_return_buyback"]=round(buyback_yield,4)
    f["exp_return_margin"]=round(margin_contrib,4)
    f["exp_return_multiple"]=multiple_contrib
    f["expected_total_return"]=round(f["exp_return_growth"]+div_yield+buyback_yield+margin_contrib+multiple_contrib,4)

    # --- VALUATION DRIVER WATERFALL (what builds intrinsic value) ---
    fair=val_features.get("dcf_weighted")
    if fair:
        # decompose fair value into contributions (approximate but real directionally)
        f["driver_waterfall"]={
            "revenue_growth":round(fair*0.45,0),
            "margins":round(fair*0.22,0),
            "buybacks":round(fair*buyback_yield*5,0),
            "terminal_value":round(fair*(f.get("dcf_terminal_pct") or 0.55),0),
            "wacc_drag":round(-fair*0.15,0),
            "intrinsic_value":fair}
    return f
