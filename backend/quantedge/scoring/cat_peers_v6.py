"""Peers Intelligence catalog v6 — 7 categories, weight 2.00. All percentile ranks vs peer set."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "quality_rank":("Quality Rank",0.40,[
   _s("roic_rank","ROIC rank","roic_rank",0.35,0.5,0.8,evidence="ROIC percentile vs peers"),
   _s("roe_rank","ROE rank","roe_rank",0.30,0.5,0.8,evidence="ROE percentile vs peers"),
   _s("roa_rank","ROA rank","roa_rank",0.20,0.5,0.8,evidence="ROA percentile vs peers"),
   _s("quality_comp","Quality composite","quality_composite",0.15,0.5,0.8,evidence="blended quality rank")]),
 "profitability_rank":("Profitability Rank",0.35,[
   _s("gm_rank","Gross margin rank","gross_margin_rank",0.30,0.5,0.8,evidence="gross margin percentile"),
   _s("nm_rank","Net margin rank","net_margin_rank",0.30,0.5,0.8,evidence="net margin percentile"),
   _s("ocfm_rank","OCF margin rank","ocf_margin_rank",0.20,0.5,0.8,evidence="cash margin percentile"),
   _s("profit_comp","Profitability composite","profitability_composite",0.20,0.5,0.8,evidence="blended margin rank")]),
 "valuation_rank":("Valuation Rank",0.35,[
   _s("pe_rank","P/E rank","pe_rank",0.30,0.5,0.8,evidence="cheaper P/E than peers"),
   _s("ps_rank","P/S rank","ps_rank",0.20,0.5,0.8,evidence="cheaper P/S than peers"),
   _s("pb_rank","P/B rank","pb_rank",0.15,0.5,0.8,evidence="cheaper P/B than peers"),
   _s("ey_rank","Earnings yield rank","earnings_yield_rank",0.20,0.5,0.8,evidence="higher earnings yield"),
   _s("ocfy_rank","OCF yield rank","ocf_yield_rank",0.15,0.5,0.8,evidence="higher cash yield")]),
 "growth_rank":("Growth Rank",0.35,[
   _s("rev_g_rank","Revenue growth rank","revenue_growth_rank",0.55,0.5,0.8,evidence="revenue growth percentile"),
   _s("earn_g_rank","Earnings growth rank","earnings_growth_rank",0.45,0.5,0.8,evidence="earnings growth percentile")]),
 "safety_rank":("Safety & Efficiency Rank",0.25,[
   _s("cr_rank","Current ratio rank","current_ratio_rank",0.50,0.5,0.8,evidence="liquidity vs peers"),
   _s("at_rank","Asset turnover rank","asset_turnover_rank",0.50,0.5,0.8,evidence="efficiency vs peers")]),
 "composite_rank":("Composite Standing",0.20,[
   _s("overall","Overall peer rank","overall_peer_rank",0.45,0.5,0.75,evidence="average rank across dimensions"),
   _s("top_q","Top-quartile breadth","top_quartile_count",0.30,0.3,0.6,evidence="share of top-quartile metrics"),
   _s("bot_q","Bottom-quartile exposure","bottom_quartile_count",0.25,0.3,0.1,hib=False,evidence="share of bottom-quartile metrics")]),
 "peer_context":("Peer Context",0.10,[
   _s("consistency","Rank consistency","peer_rank_consistency",0.60,0.5,0.8,evidence="consistent across metrics"),
   _s("set_size","Peer set size","peer_set_size",0.40,10,40,evidence="comparison robustness")]),
}
PEERS_INTELLIGENCE={"label":"Peers Intelligence","weight":2.0,"categories":CATEGORIES}

def peers_rating(score):
    if score is None: return "Unrated"
    if score>=72: return "Peer Leader"
    if score>=58: return "Above Peers"
    if score>=44: return "In Line"
    if score>=30: return "Below Peers"
    return "Peer Laggard"
