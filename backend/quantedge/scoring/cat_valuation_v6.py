"""Valuation Intelligence catalog v6 — 10 categories, weights sum to 10.00, 34 signals.
Every deep valuation signal scored: gap-to-price per model, DCF sensitivity robustness,
full multiples suite, asset-based upsides, DDM, historical P/E context.
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
   _s("dcf_mos","Margin of safety (DCF)","margin_of_safety",0.40,0.0,0.30,floor=-0.5,floor_score=15,evidence="(fair - price)/fair, weighted DCF"),
   _s("dcf_upside","Upside to weighted DCF","dcf_weighted_upside",0.40,0.0,0.30,evidence="weighted DCF fair value vs price"),
   _s("dcf_scenarios","% DCF scenarios above price","dcf_scenarios_above_price",0.40,0.3,0.7,evidence="fraction of WACCxTG grid where fair>price"),
   _s("dcf_terminal","Terminal value reliance","dcf_terminal_pct",0.30,0.75,0.50,hib=False,evidence="lower = less speculative"),
   _s("dcf_p2f","Price / fair value","price_to_fair",0.50,1.0,0.7,hib=False,evidence="1.0 = fair, <1 = discount"),
 ]),
 "intrinsic": ("Intrinsic Value", 2.00, [
   _s("consensus_upside","Intrinsic consensus upside","consensus_upside",0.55,0.0,0.30,evidence="upside vs avg of all intrinsic models"),
   _s("epv_upside","EPV upside","epv_upside",0.40,0.0,0.20,evidence="Earnings Power Value vs price"),
   _s("graham_upside","Graham upside","graham_upside",0.35,0.0,0.20,evidence="Graham number vs price"),
   _s("ri_upside","Residual income upside","residual_income_upside",0.35,0.0,0.20,evidence="residual income value vs price"),
   _s("ddm_upside","DDM upside (payers)","ddm_upside",0.35,0.0,0.15,evidence="dividend discount model vs price"),
 ]),
 "relative_multiples": ("Relative Valuation", 1.50, [
   _s("pe","P/E","mult_pe",0.30,25,12,hib=False,floor=60,floor_score=10,evidence="price / earnings"),
   _s("peg","PEG","mult_peg",0.30,2.0,1.0,hib=False,evidence="P/E / growth"),
   _s("ev_ebitda","EV/EBITDA","mult_ev_ebitda",0.30,18,10,hib=False,evidence="enterprise value / EBITDA"),
   _s("ev_sales","EV/Sales","mult_ev_sales",0.20,8,3,hib=False,evidence="enterprise value / revenue"),
   _s("ev_ebit","EV/EBIT","mult_ev_ebit",0.20,22,12,hib=False,evidence="enterprise value / EBIT"),
   _s("p_fcf","P/FCF","mult_p_fcf",0.20,30,15,hib=False,evidence="price / free cash flow"),
 ]),
 "earnings_based": ("Earnings-Based Value", 1.20, [
   _s("pb","P/B","mult_pb",0.35,6,2,hib=False,evidence="price / book value"),
   _s("earnings_yield","Earnings yield","mult_pe",0.35,25,12,hib=False,evidence="inverse P/E"),
   _s("ev_ic","EV/Invested Capital","mult_ev_ic",0.30,8,3,hib=False,evidence="EV / invested capital"),
   _s("pe_history","P/E vs own history","pe_vs_history",0.20,1.2,0.8,hib=False,evidence="current P/E / 1.5yr avg (<1=cheap)"),
 ]),
 "reverse_dcf": ("Market Expectations", 1.00, [
   _s("implied_growth","Reverse-DCF implied growth","reverse_dcf_implied_growth",0.60,0.15,0.05,hib=False,cap=0.35,cap_score=10,evidence="growth the price assumes; lower=less demanding"),
   _s("expectation_gap","Expectation demand","reverse_dcf_implied_growth",0.40,0.15,0.05,hib=False,evidence="how much growth is priced in"),
 ]),
 "margin_of_safety": ("Margin of Safety", 1.00, [
   _s("mos_primary","Primary MoS","margin_of_safety",0.50,0.0,0.35,floor=-0.5,floor_score=10,evidence="downside protection vs fair value"),
   _s("mos_consensus","Consensus MoS","consensus_upside",0.30,0.0,0.30,evidence="upside across all models"),
   _s("scenario_safety","Scenario robustness","dcf_scenarios_above_price",0.20,0.3,0.7,evidence="% of DCF scenarios with margin"),
 ]),
 "quality_price": ("Quality at a Price", 0.60, [
   _s("price_to_fair2","Price to fair","price_to_fair",0.60,1.0,0.75,hib=False,evidence="how far above/below fair"),
   _s("fcf_yield","FCF yield","mult_p_fcf",0.40,30,15,hib=False,evidence="inverse P/FCF"),
 ]),
 "asset_based": ("Asset-Based Value", 0.30, [
   _s("nav_upside","NAV upside","nav_upside",0.40,-0.5,0.0,evidence="net asset value vs price"),
   _s("tangible_nav","Tangible NAV upside","tangible_nav_upside",0.30,-0.7,-0.2,evidence="tangible NAV vs price"),
   _s("pb_asset","P/B (asset)","mult_pb",0.30,6,2,hib=False,evidence="asset-based value check"),
 ]),
 "valuation_context": ("Valuation Context", 0.20, [
   _s("ps_context","P/S context","mult_ps",0.50,8,3,hib=False,evidence="sales multiple"),
   _s("pe_hist_context","P/E history context","pe_vs_history",0.50,1.2,0.8,hib=False,evidence="P/E vs own history"),
 ]),
 "model_agreement": ("Model Agreement", 0.20, [
   _s("model_agreement","Cross-model agreement","model_agreement_score",0.50,0.5,0.85,evidence="1 - dispersion across DCF/EPV/Graham/RI (high = methods agree)"),
   _s("model_consensus","Overvalued consensus","model_consensus_overvalued",0.50,0.5,0.2,hib=False,evidence="fraction of methods placing fair value below price (low = cheap)"),
 ]),
}
VALUATION_INTELLIGENCE = {"label":"Valuation Intelligence","weight":10.0,"categories":CATEGORIES}

def valuation_rating(score):
    """Categorical verdict parallel to Moat Rating — the spec's Valuation Rating output."""
    if score is None: return "Unrated"
    if score >= 75: return "Undervalued"
    if score >= 58: return "Modestly Undervalued"
    if score >= 42: return "Fairly Valued"
    if score >= 25: return "Modestly Overvalued"
    return "Overvalued"
