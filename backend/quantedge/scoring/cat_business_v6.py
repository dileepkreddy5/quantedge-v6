"""Business Intelligence catalog v6 — 8 categories, weights sum to 12.00, 20 signals.
Real quantitative moat/durability proxies (persistent excess returns, pricing power,
recurring revenue, operating leverage, reinvestment economics). Qualitative signals
(market share, switching costs) honestly needs_source -> AI Analyst layer.
"""
def _s(id,label,field,weight,good,great,hib=True,status="live",floor=None,floor_score=30.0,
       cap=None,cap_score=85.0,evidence=""):
    d={"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
       "good":good,"great":great,"status":status,"evidence":evidence}
    if floor is not None: d["floor"]=floor;d["floor_score"]=floor_score
    if cap is not None: d["cap"]=cap;d["cap_score"]=cap_score
    return d

CATEGORIES = {
 "moat_strength": ("Moat Strength", 2.50, [
   _s("excess_return_spread","ROIC-WACC excess return","excess_return_spread",0.55,0.0,0.20,evidence="persistent excess returns = economic moat"),
   _s("roic_current","Return on invested capital","roic_current",0.45,0.10,0.25,evidence="current ROIC level"),
   _s("roe_stability","ROE stability (5y)","roe_stability",0.50,0.6,0.9,evidence="consistency of returns = durable advantage"),
   _s("moat_trend","Margin/moat trend","gross_margin_trend",0.50,-0.005,0.005,evidence="widening vs eroding advantage"),
 ]),
 "pricing_power": ("Pricing Power", 2.00, [
   _s("gross_margin_level","Gross margin level","gross_margin_level",0.45,0.35,0.60,evidence="ability to price above cost"),
   _s("gross_margin_stability","Gross margin stability","gross_margin_stability",0.35,0.7,0.95,evidence="pricing power durability"),
   _s("gross_margin_trend","Gross margin trend","gross_margin_trend",0.20,-0.005,0.005,evidence="expanding vs compressing pricing"),
 ]),
 "recurring_revenue": ("Revenue Quality & Recurrence", 2.00, [
   _s("recurring_revenue_ratio","Recurring revenue mix","recurring_revenue_ratio",0.50,0.05,0.25,evidence="deferred/contracted revenue base"),
   _s("recurring_revenue_growth","Recurring revenue growth","recurring_revenue_growth",0.30,0.0,0.15,evidence="growth in contracted revenue"),
   _s("revenue_consistency","Revenue consistency","revenue_consistency",0.20,0.6,0.9,evidence="predictability of revenue"),
 ]),
 "scale_advantages": ("Scale & Operating Leverage", 1.50, [
   _s("operating_leverage","Operating leverage","operating_leverage",0.55,1.0,1.5,cap=3.0,cap_score=95,evidence="margin expansion as revenue scales"),
   _s("operating_margin_trend","Operating margin trend","operating_margin_trend",0.45,-0.005,0.005,evidence="improving efficiency at scale"),
 ]),
 "reinvestment": ("Reinvestment Economics", 1.50, [
   _s("reinvestment_quality","Reinvestment quality","reinvestment_quality",0.60,0.05,0.15,evidence="reinvestment rate x ROIC = value-creating growth"),
   _s("revenue_growth_durability","Growth durability","revenue_growth_durability",0.40,0.03,0.12,evidence="sustainable growth rate"),
 ]),
 "capital_efficiency": ("Capital Efficiency", 1.00, [
   _s("capital_intensity","Capital intensity","capital_intensity",0.60,0.15,0.05,hib=False,evidence="capex/revenue (asset-light = durable)"),
   _s("roic_current2","Capital productivity","roic_current",0.40,0.10,0.25,evidence="return per dollar invested"),
 ]),
 "competitive_position": ("Competitive Position", 1.00, [
   _s("market_share","Market share","market_share",0.0,0.1,0.3,status="needs_source",evidence="share of addressable market (industry data pending)"),
   _s("share_trend","Market share trend","share_trend",0.0,0.0,0.05,status="needs_source",evidence="gaining vs losing share (pending)"),
 ]),
 "business_durability": ("Business Model Durability", 0.50, [
   _s("switching_costs","Switching costs","switching_costs",0.0,0.3,0.7,status="needs_source",evidence="customer lock-in (qualitative — AI Analyst layer)"),
   _s("network_effects","Network effects","network_effects",0.0,0.3,0.7,status="needs_source",evidence="demand-side scale (qualitative — AI Analyst layer)"),
 ]),
}
BUSINESS_INTELLIGENCE = {"label":"Business Intelligence","weight":12.0,"categories":CATEGORIES}

def moat_rating(score):
    if score is None: return "Unrated"
    if score>=75: return "Wide Moat"
    if score>=58: return "Narrow Moat"
    if score>=40: return "Emerging Moat"
    return "No Moat"
