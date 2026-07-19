"""Valuation Intelligence catalog v6 — 10 categories, weights sum to 10.00.
Valuation signals are mostly lower-is-better (cheap multiples) or higher-is-better
(margin of safety). Reuses financial data; analyst-estimate signals are needs_source.
"""
def _s(id,label,field,weight,good,great,hib=True,status="live",floor=None,floor_score=30.0,
       cap=None,cap_score=85.0,evidence=""):
    d={"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
       "good":good,"great":great,"status":status,"evidence":evidence}
    if floor is not None: d["floor"]=floor;d["floor_score"]=floor_score
    if cap is not None: d["cap"]=cap;d["cap_score"]=cap_score
    return d

CATEGORIES = {
 "dcf": ("DCF Valuation", 2.00, [
   _s("dcf_mos","Margin of safety (DCF)","margin_of_safety",0.60,0.0,0.30,floor=-0.5,floor_score=15,evidence="(fair value - price)/fair value, weighted DCF"),
   _s("dcf_upside","Upside to DCF fair value","upside_to_fair",0.55,0.0,0.30,evidence="(fair - price)/price"),
   _s("dcf_terminal","Terminal value reliance","dcf_terminal_pct",0.35,0.75,0.50,hib=False,evidence="lower = less speculative (terminal % of DCF)"),
   _s("dcf_price_to_fair","Price / fair value","price_to_fair",0.50,1.0,0.7,hib=False,evidence="1.0 = fairly valued, <1 = discount"),
 ]),
 "intrinsic": ("Intrinsic Value", 2.00, [
   _s("consensus_upside","Intrinsic consensus upside","consensus_upside",0.70,0.0,0.30,evidence="upside vs avg of DCF/EPV/Graham/RI"),
   _s("epv_upside","EPV vs price","epv_per_share",0.45,0,0,status="reference",evidence="Earnings Power Value / share (no-growth floor)"),
   _s("graham_ref","Graham number","graham_number",0.45,0,0,status="reference",evidence="sqrt(22.5 x EPS x BVPS)"),
   _s("residual_income_ref","Residual income value","residual_income_value",0.40,0,0,status="reference",evidence="book value + PV of excess returns"),
 ]),
 "relative_multiples": ("Relative Valuation", 1.50, [
   _s("pe","P/E ratio","mult_pe",0.35,25,12,hib=False,floor=60,floor_score=10,evidence="price / earnings"),
   _s("peg","PEG ratio","mult_peg",0.35,2.0,1.0,hib=False,evidence="P/E / growth (growth-adjusted)"),
   _s("ev_ebitda","EV / EBITDA","mult_ev_ebitda",0.35,18,10,hib=False,evidence="enterprise value / EBITDA"),
   _s("p_fcf","P / FCF","mult_p_fcf",0.25,30,15,hib=False,evidence="price / free cash flow"),
   _s("ps","P / S","mult_ps",0.20,8,3,hib=False,evidence="price / sales"),
 ]),
 "earnings_based": ("Earnings-Based Value", 1.20, [
   _s("pb","P / B ratio","mult_pb",0.40,6,2,hib=False,evidence="price / book value"),
   _s("earnings_yield","Earnings yield (E/P)","mult_pe",0.40,25,12,hib=False,evidence="inverse P/E = earnings yield"),
   _s("graham_mos","Graham margin of safety","graham_number",0.40,0,0,status="reference",evidence="Graham fair value cushion"),
 ]),
 "reverse_dcf": ("Market Expectations", 1.00, [
   _s("implied_growth","Reverse-DCF implied growth","reverse_dcf_implied_growth",0.60,0.15,0.05,hib=False,cap=0.35,cap_score=10,evidence="FCF growth the current price assumes; lower = less demanding"),
   _s("expectation_gap","Growth expectation gap","reverse_dcf_implied_growth",0.40,0.15,0.05,hib=False,evidence="market-implied vs achievable growth"),
 ]),
 "margin_of_safety": ("Margin of Safety", 1.00, [
   _s("mos_primary","Primary margin of safety","margin_of_safety",0.60,0.0,0.35,floor=-0.5,floor_score=10,evidence="downside protection vs fair value"),
   _s("mos_consensus","Consensus MoS","consensus_upside",0.40,0.0,0.30,evidence="upside across all intrinsic models"),
 ]),
 "quality_price": ("Quality at a Price", 0.60, [
   _s("price_to_fair2","Price to fair","price_to_fair",0.60,1.0,0.75,hib=False,evidence="how far above/below fair value"),
   _s("fcf_yield","FCF yield","mult_p_fcf",0.40,30,15,hib=False,evidence="inverse P/FCF"),
 ]),
 "asset_based": ("Asset-Based Value", 0.30, [
   _s("pb_asset","Price / book (asset)","mult_pb",0.60,6,2,hib=False,evidence="asset-based value check"),
   _s("graham_asset","Graham defensive value","graham_number",0.40,0,0,status="reference",evidence="conservative asset+earnings value"),
 ]),
 "valuation_context": ("Valuation Context", 0.20, [
   _s("ps_context","P/S in context","mult_ps",0.60,8,3,hib=False,evidence="sales multiple vs history"),
   _s("multiple_reasonable","Multiple reasonableness","mult_ev_ebitda",0.40,18,10,hib=False,evidence="EV/EBITDA reasonableness"),
 ]),
 "analyst_targets": ("Analyst Targets", 0.20, [
   _s("analyst_target","Analyst consensus target","analyst_upside",0.0,0.0,0.20,status="needs_source",evidence="consensus price target upside (feed pending)"),
   _s("forward_pe","Forward P/E","forward_pe",0.0,25,12,hib=False,status="needs_source",evidence="P/E on forward estimates (feed pending)"),
 ]),
}
VALUATION_INTELLIGENCE = {"label":"Valuation Intelligence","weight":10.0,"categories":CATEGORIES}
