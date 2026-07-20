"""Ownership Intelligence catalog v6 — 8 categories, weight 4.00."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "structure":("Ownership Structure",0.65,[
   _s("share_trend","Share count trend","share_count_trend",0.30,0.01,-0.02,hib=False,evidence="buyback (shrink) vs dilution"),
   _s("conc_trend","Concentration trend","ownership_concentration_trend",0.30,0.0,0.02,evidence="float shrinking = concentrating"),
   _s("dilution_gap","Options overhang","dilution_gap",0.20,0.05,0.0,hib=False,evidence="diluted vs basic shares"),
   _s("shares_out","Shares outstanding","shares_outstanding_b",0.20,50,1,hib=False,evidence="lower share count = tighter")]),
 "insider_own":("Insider Ownership",0.60,[
   _s("insider_net","Insider net activity","insider_net_activity",0.30,-2,2,evidence="officer+director net buys"),
   _s("insider_buy","Insider buy ratio","insider_buy_ratio",0.25,0.15,0.5,evidence="buy conviction"),
   _s("insider_cluster","Cluster buying","insider_cluster",0.20,0,1,evidence="3+ insiders buying"),
   _s("insider_conv","Insider net conviction","insider_net_conviction",0.25,-0.005,0.002,evidence="net buy value / mcap")]),
 "institutional":("Institutional Holders",0.50,[
   _s("holder_count","Major holder count","major_holder_count",0.30,1,5,evidence="13G filers >5%"),
   _s("inst_conc","Institutional concentration","institutional_concentration",0.25,5,25,evidence="top-3 holder stake"),
   _s("avg_stake","Avg holder stake","avg_holder_stake",0.20,3,10,evidence="average institutional position"),
   _s("inst_interest","Institutional interest","institutional_interest",0.25,0.2,0.8,evidence="breadth of major holders")]),
 "share_stability":("Share Stability",0.55,[
   _s("share_stab","Share count stability","share_count_stability",0.35,0.5,0.9,evidence="consistent share count"),
   _s("dilution_press","Dilution pressure","dilution_pressure",0.35,0.06,0.01,hib=False,evidence="SBC / revenue"),
   _s("conc_dir","Concentration direction","concentration_direction",0.30,0.0,0.02,evidence="ownership tightening")]),
 "smart_money":("Smart Money Signals",0.55,[
   _s("smart_buy","Smart money buying","smart_money_buying",0.30,0,1,evidence="insider open-market buys"),
   _s("officer_conv","Officer conviction","officer_conviction",0.30,0,1,evidence="net officer buying"),
   _s("inst_int2","Institutional interest","institutional_interest",0.20,0.2,0.8,evidence="major holder breadth"),
   _s("buyback_int","Buyback intensity","buyback_intensity",0.20,0.005,0.03,evidence="buybacks / mcap")]),
 "float_liquidity":("Float & Liquidity",0.40,[
   _s("float_liq","Float liquidity","float_liquidity",0.35,0.3,0.8,evidence="volume vs shares"),
   _s("turnover","Turnover ratio","turnover_ratio",0.35,0.002,0.01,evidence="daily volume / shares"),
   _s("mcap_liq","Market cap ($B)","market_cap_b",0.30,10,200,evidence="size/liquidity")]),
 "conviction":("Ownership Conviction",0.40,[
   _s("own_conv","Ownership conviction","ownership_conviction",0.45,0.15,0.5,evidence="insider buy conviction"),
   _s("buyback_int2","Buyback conviction","buyback_intensity",0.30,0.005,0.03,evidence="management buying back"),
   _s("unique_buyers","Unique insider buyers","insider_unique_buyers",0.25,1,4,evidence="breadth of insider buying")]),
 "concentration":("Concentration Risk",0.35,[
   _s("conc_risk","Holder concentration risk","holder_concentration_risk",0.50,25,8,hib=False,evidence="single-holder dominance risk"),
   _s("top_pct","Top holder stake","top_holder_pct",0.50,20,7,hib=False,evidence="largest holder %")]),
}
OWNERSHIP_INTELLIGENCE={"label":"Ownership Intelligence","weight":4.0,"categories":CATEGORIES}

def ownership_rating(score):
    if score is None: return "Unrated"
    if score>=70: return "Strong Hands"
    if score>=56: return "Stable"
    if score>=42: return "Mixed"
    if score>=28: return "Weak Hands"
    return "Distributed"
