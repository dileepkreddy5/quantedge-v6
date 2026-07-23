"""Business Intelligence catalog v6 — FULL DEPTH. 10 categories, weight 12.00, 76 signals.
Real quantitative moat/durability scoring. Qualitative -> needs_source (AI Analyst layer).
"""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "moat_strength": ("Moat Strength", 2.20, [
   _s("roic_level","Return on invested capital","roic_level",0.14,0.10,0.25,evidence="core capital productivity"),
   _s("roic_wacc_spread","ROIC-WACC excess return","roic_wacc_spread",0.16,0.0,0.15,evidence="excess return = economic moat"),
   _s("spread_persistence_yrs","Excess-return persistence","spread_persistence_yrs",0.12,2.0,4.0,evidence="years of positive spread"),
   _s("roic_trend","ROIC trend","roic_trend",0.10,-0.005,0.005,evidence="widening vs eroding moat"),
   _s("roic_5y_avg","5-year avg ROIC","roic_5y_avg",0.10,0.10,0.22,evidence="sustained returns"),
   _s("croic","Cash ROIC","croic",0.10,0.08,0.20,evidence="cash-based capital productivity"),
   _s("incremental_roic","Incremental ROIC (ROIIC)","incremental_roic",0.10,0.10,0.30,evidence="returns on NEW capital"),
   _s("economic_profit","Economic profit ($)","economic_profit",0.06,0,1e10,evidence="dollar value creation above cost of capital"),
   _s("moat_direction","Moat direction","moat_direction",0.06,0.0,1.0,evidence="widening/stable/eroding"),
   _s("excess_return_consistency","Excess-return consistency","excess_return_consistency",0.06,0.6,0.9,evidence="stability of the moat"),
 ]),
 "pricing_power": ("Pricing Power", 1.60, [
   _s("gross_margin_level","Gross margin level","gross_margin_level",0.20,0.35,0.60,evidence="ability to price above cost"),
   _s("gross_margin_stability","Gross margin stability","gross_margin_stability",0.16,0.7,0.95,evidence="pricing durability"),
   _s("gross_margin_trend","Gross margin trend","gross_margin_trend",0.12,-0.003,0.003,evidence="expanding vs compressing"),
   _s("ebitda_margin_level","EBITDA margin level","ebitda_margin_level",0.14,0.20,0.40,evidence="operating profitability"),
   _s("ebitda_margin_trend","EBITDA margin trend","ebitda_margin_trend",0.10,-0.003,0.003,evidence="operating margin direction"),
   _s("margin_resilience","Margin resilience","margin_resilience",0.12,0.85,0.97,evidence="margin floor vs average"),
   _s("price_vs_volume_growth","Pricing contribution","price_vs_volume_growth",0.08,-0.01,0.02,evidence="gross-profit growth above revenue"),
   _s("gross_margin_vs_peers","Gross margin vs peers","gross_margin_vs_peers",0.0,0.5,0.8,status="needs_source",evidence="requires enriched peer fundamentals"),
 ]),
 "revenue_quality": ("Revenue Quality & Recurrence", 1.60, [
   _s("recurring_revenue_ratio","Recurring revenue mix","recurring_revenue_ratio",0.20,0.05,0.25,evidence="deferred/contracted revenue"),
   _s("deferred_rev_growth","Deferred revenue growth","deferred_rev_growth",0.14,0.0,0.15,evidence="growth in contracted base"),
   _s("revenue_predictability","Revenue predictability","revenue_predictability",0.16,0.7,0.95,evidence="R² of revenue trajectory"),
   _s("revenue_consistency","Revenue consistency","revenue_consistency",0.14,0.6,0.9,evidence="low growth volatility"),
   _s("revenue_cagr_5y","Revenue CAGR (5y)","revenue_cagr_5y",0.12,0.05,0.15,evidence="growth track record"),
   _s("organic_growth_est","Organic growth","organic_growth_est",0.10,0.03,0.12,evidence="growth ex-acquisitions (proxy)"),
   _s("revenue_diversification","Revenue diversification","revenue_diversification",0.0,0.3,0.7,status="needs_source",evidence="segment data pending"),
   _s("backlog_rpo","Backlog / RPO","backlog_rpo",0.0,0.5,1.5,status="needs_source",evidence="remaining performance obligations (pending)"),
   _s("customer_concentration","Customer concentration","customer_concentration",0.0,0.3,0.1,hib=False,status="needs_source",evidence="10-K text"),
 ]),
 "unit_economics": ("Unit Economics", 1.30, [
   _s("incremental_margin","Incremental margin","incremental_margin",0.20,0.20,0.45,evidence="ΔOperating income / ΔRevenue"),
   _s("contribution_margin_proxy","Contribution margin","contribution_margin_proxy",0.12,0.35,0.60,evidence="gross margin as contribution proxy"),
   _s("capital_turnover","Capital turnover","capital_turnover",0.12,0.4,1.0,evidence="revenue per dollar of assets"),
   _s("cash_conversion_cycle","Cash conversion cycle","cash_conversion_cycle",0.10,60,10,hib=False,evidence="days to cash (lower better)"),
   _s("cash_conversion_ratio","Cash conversion ratio","cash_conversion_ratio",0.12,0.9,1.2,evidence="OCF / net income"),
   _s("fcf_margin","FCF margin","fcf_margin",0.12,0.10,0.25,evidence="free cash flow / revenue"),
   _s("fcf_margin_trend","FCF margin trend","fcf_margin_trend",0.06,-0.003,0.003,evidence="cash-margin direction"),
 ]),
 "competitive_position": ("Competitive Position", 1.20, [
   _s("relative_reinvestment","Reinvestment intensity","relative_reinvestment",0.30,0.15,0.40,evidence="reinvestment vs typical"),
   _s("relative_growth_vs_industry","Growth vs industry","relative_growth_vs_industry",0.20,0.0,0.1,evidence="revenue growth vs industry median"),
   _s("roic_percentile_vs_peers","ROIC percentile vs peers","roic_percentile_vs_peers",0.25,0.5,0.8,evidence="ROIC rank vs sector peers"),
   _s("margin_percentile_vs_peers","Margin percentile vs peers","margin_percentile_vs_peers",0.25,0.5,0.8,evidence="margin rank vs sector peers"),
   _s("growth_percentile_vs_peers","Growth percentile vs peers","growth_percentile_vs_peers",0.20,0.5,0.8,evidence="growth rank vs sector peers"),
   _s("scale_rank","Scale rank","scale_rank",0.10,0.5,0.85,evidence="size rank vs sector peers"),
   _s("market_share","Market share","market_share",0.0,0.1,0.3,status="needs_source",evidence="industry data"),
   _s("share_trend","Market share trend","share_trend",0.0,0.0,0.05,status="needs_source",evidence="gaining vs losing share"),
 ]),
 "capital_allocation": ("Management & Capital Allocation", 1.30, [
   _s("buyback_yield","Buyback yield","buyback_yield",0.14,0.0,0.03,evidence="share reduction return"),
   _s("buyback_consistency","Buyback consistency","buyback_consistency",0.10,0.5,1.0,evidence="steady share-count reduction"),
   _s("share_count_cagr","Share count CAGR","share_count_cagr",0.12,0.0,-0.02,hib=False,evidence="dilution vs reduction"),
   _s("dividend_growth","Dividend growth","dividend_growth",0.10,0.0,0.10,evidence="growing shareholder returns"),
   _s("total_payout_ratio","Total payout ratio","total_payout_ratio",0.10,0.2,0.6,evidence="capital returned to owners"),
   _s("reinvestment_rate","Reinvestment rate","reinvestment_rate",0.12,0.1,0.4,evidence="capital plowed back"),
   _s("reinvestment_quality","Reinvestment quality","reinvestment_quality",0.14,0.05,0.15,evidence="reinvestment x ROIC = value creation"),
   _s("ma_intensity","M&A intensity","ma_intensity",0.08,0.15,0.02,hib=False,evidence="goodwill growth (lower=organic)"),
   _s("sbc_intensity","Stock comp intensity","sbc_intensity",0.10,0.08,0.02,hib=False,evidence="SBC / revenue (dilution)"),
 ]),
 "scale_leverage": ("Scale & Operating Leverage", 1.00, [
   _s("operating_leverage","Operating leverage","operating_leverage",0.28,1.0,1.5,evidence="margin expansion as revenue scales"),
   _s("operating_margin_trend","Operating margin trend","operating_margin_trend",0.22,-0.003,0.003,evidence="efficiency direction"),
   _s("opex_efficiency","Opex efficiency","opex_efficiency",0.16,-0.003,0.003,evidence="declining opex/revenue"),
   _s("revenue_per_employee","Revenue per employee","revenue_per_employee",0.0,300000,800000,status="needs_source",evidence="headcount data (pending)"),
 ]),
 "growth_quality": ("Growth Quality & Durability", 0.90, [
   _s("revenue_growth_durability","Growth durability","revenue_growth_durability",0.20,0.03,0.12,evidence="sustainable growth rate"),
   _s("growth_consistency","Growth consistency","growth_consistency",0.18,0.6,0.9,evidence="predictable growth"),
   _s("reinvestment_runway","Reinvestment runway","reinvestment_runway",0.14,0.1,0.4,evidence="room to reinvest"),
   _s("growth_efficiency","Growth efficiency","growth_efficiency",0.14,0.3,1.0,evidence="growth per capex dollar"),
   _s("rule_of_40","Rule of 40","rule_of_40",0.16,0.30,0.45,evidence="growth% + FCF margin%"),
   _s("sustainable_growth_rate","Sustainable growth rate","sustainable_growth_rate",0.10,0.05,0.15,evidence="ROIC x reinvestment"),
   _s("growth_deceleration_risk","Growth deceleration","growth_deceleration_risk",0.08,-0.02,0.01,evidence="recent vs prior growth"),
 ]),
 "business_risk": ("Business Risk & Durability", 0.60, [
   _s("earnings_cyclicality","Earnings cyclicality","earnings_cyclicality",0.24,0.5,0.15,hib=False,evidence="earnings-growth volatility (lower=stable)"),
   _s("revenue_volatility","Revenue volatility","revenue_volatility",0.22,0.15,0.03,hib=False,evidence="revenue-growth volatility (lower=stable)"),
   _s("asset_intensity","Asset intensity","asset_intensity",0.16,0.15,0.05,hib=False,evidence="capex/revenue (lower=asset-light)"),
   _s("leverage_stability","Leverage stability","leverage_stability",0.16,0.6,0.9,evidence="stable debt profile"),
   _s("obsolescence_proxy","R&D intensity","rd_intensity",0.22,0.03,0.12,evidence="innovation investment"),
   _s("customer_concentration_risk","Customer concentration risk","customer_concentration_risk",0.0,0.3,0.1,hib=False,status="needs_source",evidence="10-K disclosure (pending)"),
 ]),
 "intangible_moat": ("Intangible Moat", 0.30, [
   _s("rd_intensity_moat","R&D intensity","rd_intensity_moat",0.35,0.03,0.15,evidence="innovation commitment"),
   _s("rd_productivity","R&D productivity","rd_productivity",0.35,1.0,4.0,evidence="revenue growth per R&D dollar"),
   _s("brand_proxy","Brand investment (SG&A)","brand_proxy",0.30,0.05,0.20,evidence="brand/distribution proxy"),
   _s("switching_costs","Switching costs","switching_costs",0.0,0.3,0.7,status="needs_source",evidence="customer lock-in (AI Analyst)"),
   _s("network_effects","Network effects","network_effects",0.0,0.3,0.7,status="needs_source",evidence="demand-side scale (AI Analyst)"),
 ]),
}
BUSINESS_INTELLIGENCE = {"label":"Business Intelligence","weight":12.0,"categories":CATEGORIES}

def moat_rating(score):
    if score is None: return "Unrated"
    if score>=75: return "Wide Moat"
    if score>=58: return "Narrow Moat"
    if score>=40: return "Emerging Moat"
    return "No Moat"
