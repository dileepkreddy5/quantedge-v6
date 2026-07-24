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
 # Insider Activity removed. Form 4 data needs the SEC submissions index plus
 # up to sixty individual filings per company, and the endpoint throttles within
 # an hour of light use — the category worked for one ticker and was empty for
 # the next, so the composite moved for reasons unrelated to the company. The
 # fetcher in insider_fetch.py is left intact and tested; serving this properly
 # means a nightly job writing to Postgres, the shape relationships.py already
 # uses, rather than fetching at request time.
 "institutional":("Institutional Holders",0.50,[
   _s("holder_count","Institutional holders","major_holder_count",0.30,50,800,evidence="manager families with a reported position"),
   _s("inst_conc","Top-3 holder stake","institutional_concentration",0.25,25,8,hib=False,evidence="combined stake of the three largest holders — lower is less concentrated"),
   _s("avg_stake","Avg holder stake","avg_holder_stake",0.20,0.5,0.02,hib=False,evidence="mean position size across institutional holders"),
   _s("inst_interest","Holders above 5%","institutional_interest",0.25,3,0,hib=False,evidence="managers each holding more than 5% — fewer means less single-holder risk")]),
 "share_stability":("Share Stability",0.55,[
   _s("share_stab","Share count stability","share_count_stability",0.35,0.5,0.9,evidence="consistent share count"),
   _s("dilution_press","Dilution pressure","dilution_pressure",0.35,0.06,0.01,hib=False,evidence="SBC / revenue"),
   ]),
 "smart_money":("Smart Money Signals",0.55,[
   _s("buyback_int","Buyback intensity","buyback_intensity",0.20,0.005,0.03,evidence="buybacks / mcap")]),
 "float_liquidity":("Float & Liquidity",0.40,[
   _s("float_liq","Float liquidity","float_liquidity",0.35,0.3,0.8,evidence="volume vs shares"),
   _s("turnover","Turnover ratio","turnover_ratio",0.35,0.002,0.01,evidence="daily volume / shares"),
   _s("mcap_liq","Market cap ($B)","market_cap_b",0.30,10,200,evidence="size/liquidity")]),
 "conviction":("Ownership Conviction",0.40,[
   ]),
 "concentration":("Concentration Risk",0.35,[
   _s("top_pct","Largest holder","top_holder_pct",0.50,20,7,hib=False,evidence="stake of the single largest institutional holder")]),
}
OWNERSHIP_INTELLIGENCE={"label":"Ownership Intelligence","weight":4.0,"categories":CATEGORIES}

def ownership_rating(score):
    if score is None: return "Unrated"
    if score>=70: return "Strong Hands"
    if score>=56: return "Stable"
    if score>=42: return "Mixed"
    if score>=28: return "Weak Hands"
    return "Distributed"
