"""Competitive Intelligence catalog v6 — 11 categories, weight 8.00, 43 signals."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "mkt_pos":("Market Position",1.00,[
   _s("scale_rank","Scale rank vs peers","scale_rank",0.25,0.5,0.85,evidence="market-cap rank in sector"),
   _s("mcap_rank","Market-cap rank","mcap_rank",0.20,0.5,0.85,evidence="size dominance"),
   _s("market_share_proxy","Market share proxy","market_share_proxy",0.25,0.05,0.3,evidence="revenue share of peer set"),
   _s("size_dominance","Size dominance","size_dominance",0.15,0.5,0.85,evidence="scale leadership"),
   _s("revenue_rank","Revenue rank","revenue_rank",0.15,0.5,0.85,evidence="revenue scale vs peers")]),
 "rel_profit":("Relative Profitability",1.10,[
   _s("net_margin_pctile","Net margin percentile","net_margin_pctile",0.20,0.5,0.85,evidence="net margin rank vs peers"),
   _s("gross_margin_pctile","Gross margin percentile","gross_margin_pctile",0.15,0.5,0.85,evidence="gross margin rank"),
   _s("roic_pctile","ROIC percentile","roic_pctile",0.20,0.5,0.85,evidence="ROIC rank vs peers"),
   _s("roe_pctile","ROE percentile","roe_pctile",0.15,0.5,0.85,evidence="ROE rank vs peers"),
   _s("margin_advantage","Margin advantage","margin_advantage",0.15,0.0,0.15,evidence="net margin above peer median"),
   _s("roic_advantage","ROIC advantage","roic_advantage",0.15,0.0,0.1,evidence="ROIC above peer median")]),
 "rel_growth":("Relative Growth",1.00,[
   _s("growth_pctile","Growth percentile","growth_pctile",0.30,0.5,0.85,evidence="revenue growth rank"),
   _s("growth_advantage","Growth advantage","growth_advantage",0.30,0.0,0.1,evidence="growth above peer median"),
   _s("share_gain_proxy","Share gain proxy","share_gain_proxy",0.25,0.0,0.08,evidence="growing faster = gaining share"),
   _s("rev_growth_vs_median","Growth vs median","revenue_growth_vs_median",0.15,0.0,0.1,evidence="revenue growth vs peers")]),
 "pricing":("Pricing Power",0.75,[
   _s("gm_level","Gross margin level","gross_margin_level",0.30,0.3,0.6,evidence="pricing power indicator"),
   _s("gm_pctile","Gross margin percentile","gross_margin_pctile_pp",0.30,0.5,0.85,evidence="margin rank vs peers"),
   _s("margin_adv","Margin advantage","margin_advantage_pp",0.20,0.0,0.15,evidence="gross margin above median"),
   _s("premium","Price premium proxy","price_premium_proxy",0.20,0.5,0.85,evidence="margin-based pricing premium")]),
 "efficiency":("Efficiency Edge",0.75,[
   _s("at_pctile","Asset turnover percentile","asset_turnover_pctile",0.30,0.5,0.85,evidence="asset efficiency rank"),
   _s("ocf_pctile","OCF margin percentile","ocf_margin_pctile",0.30,0.5,0.85,evidence="cash conversion rank"),
   _s("eff_adv","Efficiency advantage","efficiency_advantage",0.20,0.0,0.2,evidence="asset turnover above median"),
   _s("cap_eff","Capital efficiency","capital_efficiency_vs_peers",0.20,0.5,0.85,evidence="ROIC rank")]),
 "moat":("Competitive Moat Strength",0.85,[
   _s("excess_ret","Excess return vs peers","excess_return_vs_peers",0.25,0.5,0.85,evidence="ROIC percentile"),
   _s("moat_spread","Economic moat spread","economic_moat_spread",0.30,0.0,0.15,evidence="ROIC minus WACC"),
   _s("durability","Margin durability","margin_durability",0.25,0.5,0.85,evidence="gross margin stability"),
   _s("persistence","ROIC persistence","roic_persistence",0.20,0.5,0.85,evidence="return stability")]),
 "rel_val":("Relative Valuation Position",0.55,[
   _s("pe_disc","P/E discount vs peers","pe_discount_vs_peers",0.40,0.5,0.85,evidence="cheaper than peers"),
   _s("cheap_rank","Cheapness rank","cheapness_rank",0.35,0.5,0.85,evidence="valuation attractiveness"),
   _s("pe_rel","P/E vs median","pe_relative_to_median",0.25,1.2,0.7,hib=False,evidence="P/E relative to sector median")]),
 "fin_strength":("Financial Strength vs Peers",0.65,[
   _s("roe_rank","ROE strength rank","roe_strength_rank",0.30,0.5,0.85,evidence="ROE rank vs peers"),
   _s("liquidity","Liquidity position","liquidity_position",0.25,1.0,2.5,evidence="current ratio"),
   _s("profit_rank","Profitability rank","profitability_rank",0.25,0.5,0.85,evidence="net margin rank"),
   _s("profit_spread","Profit spread vs median","profit_spread_vs_median",0.20,0.0,0.1,evidence="margin above median")]),
 "momentum":("Competitive Momentum",0.50,[
   _s("growth_mom","Growth momentum rank","growth_momentum_rank",0.40,0.5,0.85,evidence="growth rank vs peers"),
   _s("earn_growth","Earnings growth","earnings_growth",0.35,0.0,0.2,evidence="YoY earnings growth"),
   _s("share_gain2","Share gain proxy","share_gain_proxy",0.25,0.0,0.08,evidence="relative growth")]),
 "threat":("Threat & Disruption",0.40,[
   _s("growth_decel","Growth deceleration risk","growth_decel_risk",0.35,-0.05,0.05,evidence="losing growth ground vs peers"),
   _s("share_loss","Share loss risk","share_loss_risk",0.35,1,0,hib=False,evidence="growing slower than peers"),
   _s("margin_comp","Margin compression risk","margin_compression_risk",0.30,1,0,hib=False,evidence="margins below peers")]),
 "scale":("Scale & Network",0.45,[
   _s("abs_scale","Absolute scale ($B)","absolute_scale",0.35,10,200,evidence="market cap"),
   _s("emp_scale","Employee scale","employee_scale",0.30,5000,100000,evidence="workforce size"),
   _s("scale_adv","Scale advantage","scale_advantage",0.35,0.5,0.85,evidence="scale rank vs peers")]),
}
COMPETITIVE_INTELLIGENCE={"label":"Competitive Intelligence","weight":8.0,"categories":CATEGORIES}

def competitive_rating(score):
    if score is None: return "Unrated"
    if score>=75: return "Dominant"
    if score>=60: return "Strong"
    if score>=45: return "Competitive"
    if score>=30: return "Challenged"
    return "Weak"
