"""Competitive Intelligence catalog v6 — 11 categories, weight 8.00, 43 signals."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 # Eleven categories became seven. Sixteen of the old 39 signals read fields the
 # features layer had aliased to an already-scored value, so the composite
 # counted the same handful of facts repeatedly — every removed signal here has
 # its computation deleted upstream, not merely hidden.
 "mkt_pos":("Scale & Market Position",0.85,[
   _s("mcap_rank","Size rank vs peers","scale_rank",0.40,0.5,0.85,evidence="market-cap rank within the peer set"),
   _s("market_share_proxy","Revenue share of peers","market_share_proxy",0.35,0.05,0.3,evidence="this company's revenue as a share of the peer set"),
   _s("abs_scale","Absolute scale ($B)","absolute_scale",0.25,10,200,evidence="market capitalisation")]),
 "rel_profit":("Profitability vs Peers",1.30,[
   _s("net_margin_pctile","Net margin rank","net_margin_pctile",0.25,0.5,0.85,evidence="net margin percentile in the peer set"),
   _s("gross_margin_pctile","Gross margin rank","gross_margin_pctile",0.20,0.5,0.85,evidence="gross margin percentile"),
   _s("roic_pctile","ROIC rank","roic_pctile",0.25,0.5,0.85,evidence="return on invested capital percentile"),
   _s("roe_pctile","ROE rank","roe_pctile",0.15,0.5,0.85,evidence="return on equity percentile"),
   _s("margin_advantage","Net margin above median","margin_advantage",0.15,0.0,0.15,evidence="percentage points above the peer median")]),
 "rel_growth":("Growth vs Peers",1.00,[
   _s("growth_pctile","Revenue growth rank","growth_pctile",0.40,0.5,0.85,evidence="revenue growth percentile"),
   _s("growth_advantage","Growth above median","growth_advantage",0.35,0.0,0.1,evidence="growth in excess of the peer median"),
   _s("earn_growth","Earnings growth","earnings_growth",0.25,0.0,0.2,evidence="year-on-year earnings growth")]),
 "pricing":("Pricing Power",0.75,[
   _s("gm_level","Gross margin level","gross_margin_level",0.50,0.3,0.6,evidence="absolute gross margin — what the market lets it charge"),
   _s("margin_adv_gm","Gross margin above median","margin_advantage_gm",0.50,0.0,0.15,evidence="gross margin in excess of peers")]),
 "efficiency":("Capital Efficiency",0.75,[
   _s("at_pctile","Asset turnover rank","asset_turnover_pctile",0.40,0.5,0.85,evidence="revenue generated per unit of assets"),
   _s("ocf_pctile","Cash conversion rank","ocf_margin_pctile",0.35,0.5,0.85,evidence="operating cash flow margin percentile"),
   _s("eff_adv","Turnover above median","efficiency_advantage",0.25,0.0,0.2,evidence="asset turnover in excess of peers")]),
 "moat":("Moat Durability",0.85,[
   _s("moat_spread","Economic moat spread","economic_moat_spread",0.40,0.0,0.15,evidence="ROIC minus cost of capital"),
   _s("durability","Margin durability","margin_durability",0.35,0.5,0.85,evidence="how stable gross margin has been"),
   _s("persistence","Return persistence","roic_persistence",0.25,0.5,0.85,evidence="stability of returns over time")]),
 "rel_val":("Valuation vs Peers",0.55,[
   _s("pe_disc","P/E rank (cheaper is better)","pe_discount_vs_peers",0.55,0.5,0.85,evidence="where this P/E sits in the peer set"),
   _s("pe_rel","P/E vs peer median","pe_relative_to_median",0.45,1.2,0.7,hib=False,evidence="multiple of the peer median P/E")]),
 "threat":("Threat & Disruption",0.40,[
   _s("share_loss","Losing share","share_loss_risk",0.50,1,0,hib=False,evidence="growing slower than the peer median"),
   _s("margin_comp","Margin compression","margin_compression_risk",0.50,1,0,hib=False,evidence="margins below the peer median")]),
}
COMPETITIVE_INTELLIGENCE={"label":"Competitive Intelligence","weight":8.0,"categories":CATEGORIES}

def competitive_rating(score):
    if score is None: return "Unrated"
    if score>=78: return "Dominant"
    if score>=62: return "Strong"
    if score>=45: return "Competitive"
    if score>=30: return "Challenged"
    return "Weak"
