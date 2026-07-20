"""Institutional Flow catalog v6 — 6 categories, weight 2.00."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "money_flow":("Money Flow",0.45,[
   _s("mfi","Money Flow Index","money_flow_index",0.35,0.5,0.65,evidence="volume-weighted buying pressure"),
   _s("cmf","Chaikin Money Flow","chaikin_money_flow",0.35,0.0,0.15,evidence="21-day accumulation pressure"),
   _s("adl","ADL slope","adl_slope",0.30,0.0,0.3,evidence="accumulation/distribution trend")]),
 "block_trade":("Block-Trade Activity",0.35,[
   _s("trade_size","Avg trade size trend","avg_trade_size_trend",0.35,0.0,0.2,evidence="institutional footprint (rising size)"),
   _s("block_freq","Block-trade frequency","block_trade_frequency",0.35,0.1,0.4,evidence="large-trade days"),
   _s("inst_foot","Institutional footprint","institutional_footprint",0.30,0.0,0.25,evidence="recent vs baseline trade size")]),
 "accumulation":("Accumulation/Distribution",0.35,[
   _s("accum20","20-day accumulation","accumulation_20d",0.45,0.5,0.6,evidence="up vs down volume"),
   _s("dollar_flow","Dollar-flow momentum","dollar_flow_momentum",0.55,0.0,0.2,evidence="dollar-volume acceleration")]),
 "inst_holdings_flow":("13G Filing Flow",0.35,[
   _s("recent_13g","Recent 13G filings","recent_13g_filings",0.55,0,3,evidence="new institutional interest (180d)"),
   _s("holder_count","Institutional holders","institutional_holder_count",0.45,1,5,evidence="major-holder breadth")]),
 "insider_flow":("Insider Flow",0.35,[
   _s("insider_vel","Insider flow velocity","insider_flow_velocity",0.35,5,30,evidence="Form 4 filing frequency"),
   _s("insider_net","Insider net flow","insider_net_flow",0.40,0.15,0.5,evidence="insider buy conviction"),
   _s("insider_cluster","Insider cluster flow","insider_cluster_flow",0.25,0,1,evidence="3+ insiders buying")]),
 "smart_footprint":("Smart-Money Footprint",0.15,[
   _s("mfi2","Money-flow strength","money_flow_index",0.40,0.5,0.65,evidence="composite buying pressure"),
   _s("accum2","Net accumulation","accumulation_20d",0.35,0.5,0.6,evidence="volume accumulation"),
   _s("foot2","Trade-size footprint","institutional_footprint",0.25,0.0,0.2,evidence="block-trade presence")]),
}
IFLOW_INTELLIGENCE={"label":"Institutional Flow","weight":2.0,"categories":CATEGORIES}

def iflow_rating(score):
    if score is None: return "Unrated"
    if score>=68: return "Strong Inflow"
    if score>=54: return "Net Inflow"
    if score>=44: return "Balanced"
    if score>=30: return "Net Outflow"
    return "Strong Outflow"
