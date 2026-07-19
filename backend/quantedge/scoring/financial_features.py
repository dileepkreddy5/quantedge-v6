"""Financial Intelligence — master feature computer. Merged per-quarter dataset ->
full metric dict the 12-category catalog scores against. Breadth (69 ratios/trends/
stability/CAGR/working-capital) + six sophisticated models. Every value real or None.
"""
from __future__ import annotations
import math, statistics as st
from typing import List, Dict, Optional, Any
from quantedge.scoring.deep_features import (_f, _safe_div, _cov_stability, _slope,
    _cagr, _yoy, dupont_decomposition, piotroski_f_score, altman_z,
    beneish_m_score, owner_earnings, normalized_roic)

def _ttm(m, k, i=None):
    end = len(m) if i is None else i+1
    vals=[m[j][k] for j in range(max(0,end-4),end) if m[j].get(k) is not None]
    return sum(vals) if len(vals)==4 else None

def _ttm_series(m, k):
    out=[]
    for i in range(3,len(m)):
        v=_ttm(m,k,i)
        if v is not None: out.append(v)
    return out

def _latest(m, k):
    for r in reversed(m):
        if r.get(k) is not None: return r[k]
    return None

def compute_financial_features(merged, market_cap=None, wacc=None):
    f={}
    if not merged: return f
    m=merged; cur=m[-1]; prev=m[-5] if len(m)>=5 else m[0]
    rev=_ttm(m,"revenue"); cogs=_ttm(m,"cost_of_revenue"); gp=_ttm(m,"gross_profit")
    oi=_ttm(m,"operating_income"); ni=_ttm(m,"net_income"); ocf=_ttm(m,"operating_cash_flow")
    capex=_ttm(m,"capex"); da=_ttm(m,"depreciation_amortization"); rd=_ttm(m,"rd")
    sbc=_ttm(m,"sbc"); div=_ttm(m,"dividends_paid"); bb=_ttm(m,"buybacks")
    fcf=(ocf-abs(capex)) if (ocf is not None and capex is not None) else None
    f["gross_margin"]=_safe_div(gp,rev); f["operating_margin"]=_safe_div(oi,rev)
    f["net_margin"]=_safe_div(ni,rev)
    f["ebitda"]=(oi+da) if (oi is not None and da is not None) else None
    f["ebitda_margin"]=_safe_div(f["ebitda"],rev); f["fcf_margin"]=_safe_div(fcf,rev)
    f["rd_intensity"]=_safe_div(rd,rev); f["sbc_to_revenue"]=_safe_div(sbc,rev)
    f["cogs_ratio"]=_safe_div(cogs,rev)
    gm_q=[_safe_div(r.get("gross_profit"),r.get("revenue")) for r in m if r.get("revenue")]
    om_q=[_safe_div(r.get("operating_income"),r.get("revenue")) for r in m if r.get("revenue")]
    nm_q=[_safe_div(r.get("net_income"),r.get("revenue")) for r in m if r.get("revenue")]
    f["gross_margin_stability"]=_cov_stability(gm_q); f["operating_margin_stability"]=_cov_stability(om_q)
    f["net_margin_stability"]=_cov_stability(nm_q)
    f["gross_margin_trend"]=_slope(gm_q); f["operating_margin_trend"]=_slope(om_q)
    rev_s=_ttm_series(m,"revenue"); ni_s=_ttm_series(m,"net_income")
    oi_s=_ttm_series(m,"operating_income"); gp_s=_ttm_series(m,"gross_profit")
    fcf_s=[(_ttm(m,"operating_cash_flow",i) or 0)-abs(_ttm(m,"capex",i) or 0)
           for i in range(3,len(m)) if _ttm(m,"operating_cash_flow",i) is not None and _ttm(m,"capex",i) is not None]
    f["revenue_growth"]=_yoy(rev_s); f["net_income_growth"]=_yoy(ni_s)
    f["operating_income_growth"]=_yoy(oi_s); f["gross_profit_growth"]=_yoy(gp_s)
    f["fcf_growth"]=_yoy(fcf_s)
    if len(rev_s)>=4: f["revenue_cagr_3y"]=_cagr(rev_s[-4],rev_s[-1],3)
    if len(rev_s)>=8: f["revenue_cagr_5y"]=_cagr(rev_s[-8],rev_s[-1],5)
    if len(ni_s)>=4: f["earnings_cagr_3y"]=_cagr(ni_s[-4],ni_s[-1],3)
    f["revenue_stability"]=_cov_stability([_yoy(rev_s[:i+1]) for i in range(1,len(rev_s))])
    ttm={"net_income":ni,"operating_cash_flow":ocf,"operating_income":oi,"revenue":rev,
         "tax_expense":_ttm(m,"tax_expense"),"pretax_income":_ttm(m,"pretax_income"),
         "depreciation_amortization":da,"capex":capex}
    nr=normalized_roic(ttm,cur,tax_rate=None)
    f["roic"]=nr["roic"]; f["roic_ex_goodwill"]=nr["roic_ex_goodwill"]
    f["nopat"]=nr["nopat"]; f["effective_tax_rate"]=nr["tax_rate"]
    if wacc is not None and f["roic"] is not None: f["roic_wacc_spread"]=f["roic"]-wacc
    roic_q=[]
    for r in m:
        n2=normalized_roic({"operating_income":r.get("operating_income"),
            "tax_expense":r.get("tax_expense"),"pretax_income":r.get("pretax_income")}, r)
        if n2["roic"] is not None: roic_q.append(n2["roic"])
    f["roic_stability"]=_cov_stability(roic_q); f["roic_trend"]=_slope(roic_q)
    dp=dupont_decomposition(ni,rev,cur.get("assets"),cur.get("equity"))
    f["roe"]=dp["roe_dupont"]; f["asset_turnover"]=dp["asset_turnover"]
    f["equity_multiplier"]=dp["equity_multiplier"]; f["roa"]=_safe_div(ni,cur.get("assets"))
    f["fcf_conversion"]=_safe_div(fcf,ni); f["ocf_to_net_income"]=_safe_div(ocf,ni)
    f["earnings_quality"]=_safe_div(ocf,ni)
    f["capex_intensity"]=_safe_div(abs(capex) if capex else None,rev)
    f["fcf_stability"]=_cov_stability(fcf_s); f["owner_earnings"]=owner_earnings(ttm)
    f["owner_earnings_yield"]=_safe_div(f["owner_earnings"],market_cap) if market_cap else None
    f["buyback_yield"]=_safe_div(bb,market_cap) if market_cap else None
    f["dividend_yield"]=_safe_div(div,market_cap) if market_cap else None
    f["shareholder_yield"]=_safe_div((div or 0)+(bb or 0),market_cap) if market_cap else None
    f["dividend_payout"]=_safe_div(div,ni); f["dividend_coverage"]=_safe_div(fcf,div) if div else None
    f["reinvestment_rate"]=_safe_div(abs(capex) if capex else None,ocf)
    sh_s=[r.get("diluted_shares") for r in m if r.get("diluted_shares") is not None]
    if len(sh_s)>=4 and sh_s[0]: f["share_count_trend"]=(sh_s[-1]-sh_s[0])/sh_s[0]/max(1,len(sh_s)-1)
    ca,cl=cur.get("current_assets"),cur.get("current_liabilities")
    inv,ap,rec=cur.get("inventory"),cur.get("accounts_payable"),cur.get("receivables")
    debt,cash,eq,assets=cur.get("long_term_debt"),cur.get("cash"),cur.get("equity"),cur.get("assets")
    gw=cur.get("goodwill")
    f["current_ratio"]=_safe_div(ca,cl)
    f["quick_ratio"]=_safe_div((ca-(inv or 0)) if ca is not None else None,cl)
    f["cash_ratio"]=_safe_div(cash,cl); f["debt_to_equity"]=_safe_div(debt,eq)
    f["debt_to_ebitda"]=_safe_div(debt,f["ebitda"])
    f["net_debt_to_ebitda"]=_safe_div((debt-cash) if (debt is not None and cash is not None) else None,f["ebitda"])
    f["goodwill_ratio"]=_safe_div(gw,assets); f["equity_ratio"]=_safe_div(eq,assets)
    if rec is not None and rev: f["dso"]=rec/rev*365
    if inv is not None and cogs: f["dio"]=inv/cogs*365
    if ap is not None and cogs: f["dpo"]=ap/cogs*365
    if all(f.get(k) is not None for k in ("dso","dio","dpo")):
        f["cash_conversion_cycle"]=f["dso"]+f["dio"]-f["dpo"]
    if ni is not None and ocf is not None and assets:
        f["accruals_ratio"]=(ni-ocf)/assets
        f["cash_earnings_gap"]=abs(ni-ocf)/(abs(ni) if ni else 1)
    f["sbc_dilution_ratio"]=_safe_div(sbc,ni)
    pf=piotroski_f_score(cur,prev,ttm); f["piotroski_f"]=pf["f_score"]; f["piotroski_components"]=pf["components"]
    az=altman_z(cur,ttm,market_cap or 0); f["altman_z"]=az.get("z_score"); f["altman_zone"]=az.get("zone")
    bm=beneish_m_score(cur,prev); f["beneish_m"]=bm.get("m_score"); f["beneish_flag"]=bm.get("flag")
    f.update(_compute_deepening(m, f, market_cap=market_cap))
    return f


def _compute_deepening(m, feats, market_cap=None):
    """Tab-3 deepening: EV multiples, book value, interest coverage, deferred-rev
    quality, 5y CAGRs, net debt, adjusted leverage. All real, missing->None."""
    f = {}
    cur = m[-1]
    def ttm(k):
        vals=[q[k] for q in m[-4:] if q.get(k) is not None]
        return sum(vals) if len(vals)==4 else None
    def ttm_at(k,i):
        vals=[m[j][k] for j in range(i-3,i+1) if m[j].get(k) is not None]
        return sum(vals) if len(vals)==4 else None
    ebitda=feats.get("ebitda")
    debt=_f(cur.get("long_term_debt")) or 0
    std=_f(cur.get("short_term_debt")) or 0
    cash=_f(cur.get("cash")) or 0
    total_debt=debt+std
    eq=_f(cur.get("equity")); assets=_f(cur.get("assets"))
    gw=_f(cur.get("goodwill")) or 0; intang=_f(cur.get("intangibles")) or 0
    shares=_f(cur.get("diluted_shares"))
    rev=ttm("revenue"); oi=ttm("operating_income"); int_exp=ttm("interest_expense")
    fcf=feats.get("fcf_margin") and rev and feats["fcf_margin"]*rev
    f["net_debt"]=total_debt-cash
    if market_cap:
        ev=market_cap+total_debt-cash
        f["enterprise_value"]=ev
        f["ev_ebitda"]=_safe_div(ev,ebitda)
        f["ev_revenue"]=_safe_div(ev,rev)
        f["ev_fcf"]=_safe_div(ev,fcf)
    if shares and eq is not None:
        f["book_value_per_share"]=eq/shares
        f["tangible_bvps"]=(eq-gw-intang)/shares
        if market_cap:
            price=market_cap/shares
            f["price_to_book"]=_safe_div(price, eq/shares)
            tbv=eq-gw-intang
            f["price_to_tangible_book"]=_safe_div(price, tbv/shares) if tbv>0 else None
    if oi is not None and int_exp and int_exp!=0:
        f["interest_coverage"]=oi/abs(int_exp)
    lease=_f(cur.get("operating_lease_total")) or _f(cur.get("operating_lease_liab")) or 0
    if ebitda: f["adj_debt_to_ebitda"]=_safe_div(total_debt+lease, ebitda)
    dr=_f(cur.get("deferred_revenue"))
    if dr is not None and rev: f["deferred_rev_to_revenue"]=dr/rev
    dr_series=[q.get("deferred_revenue") for q in m if q.get("deferred_revenue") is not None]
    if len(dr_series)>=5 and dr_series[-5] and dr_series[-5]>0:
        f["deferred_rev_growth"]=dr_series[-1]/dr_series[-5]-1.0
    rev_s=[ttm_at("revenue",i) for i in range(3,len(m))]; rev_s=[x for x in rev_s if x is not None]
    ni_s=[ttm_at("net_income",i) for i in range(3,len(m))]; ni_s=[x for x in ni_s if x is not None]
    if len(rev_s)>=20: f["revenue_cagr_5y"]=_cagr(rev_s[-20],rev_s[-1],5)
    if len(ni_s)>=20: f["earnings_cagr_5y"]=_cagr(ni_s[-20],ni_s[-1],5)
    re=_f(cur.get("retained_earnings"))
    if re is not None and assets: f["retained_earnings_ratio"]=re/assets
    return f
