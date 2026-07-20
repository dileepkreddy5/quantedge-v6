"""Alt-Data Intelligence catalog v6 — 8 categories, weight 3.00.
Real signals from news/volume/microstructure; genuinely-unavailable feeds marked needs_source."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "volume_microstructure":("Volume Microstructure",0.55,[
   _s("accumulation","Accumulation ratio","accumulation_ratio",0.25,0.5,0.6,evidence="up-day vs down-day volume"),
   _s("obv_slope","OBV trend","obv_slope",0.25,0.0,0.3,evidence="on-balance-volume slope"),
   _s("unusual_vol","Unusual volume","unusual_volume",0.20,0.0,0.5,evidence="recent vs baseline volume"),
   _s("vol_trend","Volume trend","volume_trend",0.15,0.0,0.3,evidence="10d vs 40d volume"),
   _s("trade_size","Avg trade size trend","avg_trade_size_trend",0.15,0.0,0.2,evidence="institutional footprint (rising size)")]),
 "news_flow":("News Flow",0.50,[
   _s("news_velocity","News velocity","news_velocity",0.30,0.0,1.0,evidence="7d vs prior-3wk article rate"),
   _s("news_volume","News volume (7d)","news_volume_7d",0.20,3,15,evidence="recent article count"),
   _s("sent_mean","News sentiment","news_sentiment_mean",0.25,0.0,0.4,evidence="mean article sentiment"),
   _s("sent_disp","Sentiment dispersion","news_sentiment_dispersion",0.10,0.9,0.3,hib=False,evidence="controversy (lower=consensus)"),
   _s("coverage","Coverage breadth","news_coverage_breadth",0.15,5,30,evidence="analyst/media attention")]),
 "price_anomalies":("Price-Action Anomalies",0.45,[
   _s("range_exp","Range expansion","range_expansion",0.25,0.0,0.3,evidence="intraday volatility regime"),
   _s("gap_freq","Gap frequency","gap_frequency",0.20,0.03,0.005,hib=False,evidence="overnight gaps (lower=stable)"),
   _s("vp_diverge","Volume-price divergence","volume_price_divergence",0.30,1,0,hib=False,evidence="bearish divergence flag"),
   _s("vol_attention","Volatility attention","volatility_attention",0.25,0.03,0.01,hib=False,evidence="volatility-as-attention proxy")]),
 "attention_proxy":("Attention & Retail Flow",0.40,[
   _s("txn_surge","Transaction surge","txn_count_surge",0.40,0.0,0.3,evidence="trade-count spike (retail attention)"),
   _s("dollar_vol","Dollar volume trend","dollar_volume_trend",0.35,0.0,0.2,evidence="liquidity/interest trend"),
   _s("vol_attn2","Volatility attention","volatility_attention",0.25,0.03,0.015,hib=False,evidence="attention via volatility")]),
 "smart_flow":("Smart-Money Flow",0.35,[
   _s("insider_vel","Insider filing velocity","insider_filing_velocity",0.30,5,30,evidence="Form 4 filing frequency"),
   _s("insider_buy","Insider buy signal","insider_buy_signal",0.35,0.15,0.5,evidence="insider buy conviction"),
   _s("accum2","Accumulation","accumulation_ratio",0.35,0.5,0.6,evidence="net accumulation")]),
 "liquidity_flow":("Liquidity & Flow",0.30,[
   _s("amihud","Amihud illiquidity","amihud_illiquidity",0.40,0.02,0.001,hib=False,evidence="price impact per $ volume (lower=liquid)"),
   _s("dollar_vol2","Dollar volume trend","dollar_volume_trend",0.30,0.0,0.2,evidence="flow trend"),
   _s("rel_flow","Relative flow vs peers","relative_flow_vs_peers",0.30,0.0,0.05,evidence="momentum vs sector")]),
 "alt_signals_unavailable":("Options & Short Flow",0.30,[
   _s("options_flow","Options flow","options_flow",0.0,0.5,0.8,status="needs_source",evidence="options chain 403 on data tier"),
   _s("put_call","Put/call ratio","put_call_ratio",0.0,1.0,0.7,hib=False,status="needs_source",evidence="requires options data"),
   _s("short_interest","Short interest","short_interest",0.0,0.15,0.03,hib=False,status="needs_source",evidence="short-interest feed unavailable free"),
   _s("days_to_cover","Days to cover","days_to_cover",0.0,5,1,hib=False,status="needs_source",evidence="requires short-interest data"),
   _s("gamma_exposure","Dealer gamma","gamma_exposure",0.0,0,1,status="needs_source",evidence="requires options chain")]),
 "alternative_datasets":("Alternative Datasets",0.15,[
   _s("satellite","Satellite/foot-traffic","satellite_data",0.0,0.5,0.8,status="needs_source",evidence="paid alt-data (Orbital/Placer)"),
   _s("card_data","Credit-card spend","card_spend",0.0,0.5,0.8,status="needs_source",evidence="paid alt-data (Earnest/Facteus)"),
   _s("web_traffic","Web-traffic trend","web_traffic",0.0,0.0,0.2,status="needs_source",evidence="paid alt-data (SimilarWeb)"),
   _s("app_downloads","App-download trend","app_downloads",0.0,0.0,0.2,status="needs_source",evidence="paid alt-data (Sensor Tower)")]),
}
ALTDATA_INTELLIGENCE={"label":"Alt-Data Intelligence","weight":3.0,"categories":CATEGORIES}

def altdata_rating(score):
    if score is None: return "Unrated"
    if score>=68: return "Strong Signals"
    if score>=54: return "Positive Flow"
    if score>=44: return "Neutral Flow"
    if score>=30: return "Weak Flow"
    return "Negative Signals"
