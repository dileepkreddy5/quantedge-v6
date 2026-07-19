"""News Intelligence catalog v6 — 10 categories, weight 10.00, 49 signals.
Sentiment (real Polygon insights), volume/velocity, source quality, topics, events,
sentiment-price divergence, recency, narrative momentum, news risk. All real."""
def _s(id,label,field,weight,good,great,hib=True,status="live",evidence=""):
    return {"id":id,"label":label,"field":field,"weight":weight,"higher_is_better":hib,
            "good":good,"great":great,"status":status,"evidence":evidence}

CATEGORIES = {
 "sentiment_level": ("Sentiment Level", 2.80, [
   _s("net_sentiment","Net sentiment","net_sentiment",0.22,0.10,0.50,evidence="(positive-negative)/total articles"),
   _s("positive_ratio","Positive article share","positive_ratio",0.15,0.4,0.7,evidence="share of bullish coverage"),
   _s("negative_ratio","Negative article share","negative_ratio",0.15,0.2,0.05,hib=False,evidence="share of bearish coverage"),
   _s("sentiment_mean","Mean sentiment score","sentiment_score_mean",0.18,0.10,0.50,evidence="average sentiment (-1 to +1)"),
   _s("sentiment_dispersion","Sentiment dispersion","sentiment_dispersion",0.10,0.9,0.5,hib=False,evidence="disagreement in coverage (lower=clearer)"),
   _s("bull_bear_ratio","Bull/bear ratio","bullish_bearish_ratio",0.12,1.5,4.0,evidence="positive vs negative articles"),
   _s("strong_sentiment","Non-neutral share","strong_sentiment_share",0.08,0.3,0.6,evidence="conviction in coverage"),
 ]),
 "sentiment_trend": ("Sentiment Trend", 1.80, [
   _s("sentiment_7d","7-day sentiment","sentiment_7d",0.22,0.05,0.45,evidence="recent-week sentiment"),
   _s("sentiment_30d","30-day sentiment","sentiment_30d",0.18,0.05,0.45,evidence="month sentiment"),
   _s("sentiment_trend","Sentiment trend","sentiment_trend",0.25,-0.1,0.15,evidence="7d vs prior weeks (improving?)"),
   _s("recent_vs_base","Recent vs baseline","recent_vs_baseline_sentiment",0.18,-0.1,0.1,evidence="7d vs 30d shift"),
   _s("sentiment_accel","Sentiment acceleration","sentiment_acceleration",0.17,-0.1,0.1,evidence="latest vs prior articles"),
 ]),
 "news_volume": ("News Volume & Velocity", 0.50, [
   _s("count_7d","Articles (7d)","article_count_7d",0.20,2,15,evidence="recent coverage volume"),
   _s("count_30d","Articles (30d)","article_count_30d",0.15,10,60,evidence="month coverage volume"),
   _s("velocity","News velocity","news_velocity",0.20,0.5,3,evidence="articles per day"),
   _s("vol_vs_base","Volume vs baseline","volume_vs_baseline",0.20,0.5,2,evidence="recent vs typical volume"),
   _s("intensity","Coverage intensity","coverage_intensity",0.15,10,60,evidence="total articles in window"),
   _s("vol_spike","Volume spike","volume_spike",0.10,0,1,evidence="unusual coverage surge"),
 ]),
 "source_quality": ("Source Quality", 0.50, [
   _s("tier1_share","Tier-1 source share","tier1_source_share",0.35,0.2,0.6,evidence="Reuters/Bloomberg/WSJ etc."),
   _s("unique_pubs","Unique publishers","unique_publishers",0.25,2,10,evidence="breadth of sources"),
   _s("pub_diversity","Publisher diversity","publisher_diversity",0.20,0.1,0.5,evidence="not single-source"),
   _s("source_conc","Source concentration","source_concentration",0.20,0.6,0.2,hib=False,evidence="reliance on one outlet (lower=better)"),
 ]),
 "topic_breakdown": ("Topic Breakdown", 0.70, [
   _s("earnings_topic","Earnings coverage","earnings_mentions",0.16,0,3,evidence="earnings/revenue focus"),
   _s("product_topic","Product/innovation coverage","product_mentions",0.16,0,3,evidence="product/AI/cloud focus"),
   _s("legal_topic","Legal/regulatory coverage","legal_mentions",0.16,2,0,hib=False,evidence="lawsuit/regulatory focus (lower=better)"),
   _s("analyst_topic","Analyst coverage","analyst_mentions",0.14,0,2,evidence="analyst rating focus"),
   _s("guidance_topic","Guidance coverage","guidance_mentions",0.13,0,1.5,evidence="forward-guidance focus"),
   _s("ma_topic","M&A coverage","ma_mentions",0.09,0,1,evidence="deal activity"),
   _s("mgmt_topic","Management coverage","management_mentions",0.08,0,1,evidence="leadership focus"),
   _s("competitive_topic","Competitive coverage","competitive_mentions",0.08,2,0,hib=False,evidence="rivalry pressure (lower=better)"),
 ]),
 "event_detection": ("Event Detection", 1.40, [
   _s("upgrade","Analyst upgrade","upgrade_flag",0.25,0,1,evidence="upgrade detected"),
   _s("downgrade","Analyst downgrade","downgrade_flag",0.25,1,0,hib=False,evidence="downgrade detected"),
   _s("lawsuit","Lawsuit/litigation","lawsuit_flag",0.20,1,0,hib=False,evidence="legal action detected"),
   _s("guidance_change","Guidance change","guidance_change_flag",0.15,0,1,evidence="guidance revision"),
   _s("product_launch","Product launch","product_launch_flag",0.15,0,1,evidence="launch/unveiling"),
 ]),
 "divergence": ("Sentiment-Price Divergence", 0.90, [
   _s("contrarian","Contrarian signal","contrarian_signal",0.40,-0.5,1.0,evidence="positive news + falling price = potential upside"),
   _s("confirmation","Sentiment-price confirmation","confirmation_score",0.30,0,1,evidence="news and price agree"),
   _s("divergence","Sentiment-price divergence","sentiment_price_divergence",0.30,-0.5,1.0,evidence="news vs price direction"),
 ]),
 "recency": ("Recency & Freshness", 0.30, [
   _s("fresh_share","Fresh-news share","fresh_news_share",0.35,0.1,0.5,evidence="articles in last 2 days"),
   _s("hours_since","Hours since latest","hours_since_latest",0.30,72,6,hib=False,evidence="recency of latest article"),
   _s("stale","Stale coverage","stale_flag",0.20,1,0,hib=False,evidence="no fresh news flag"),
   _s("latest_sent","Latest article sentiment","latest_sentiment",0.15,0.0,1.0,evidence="most recent article tone"),
 ]),
 "narrative": ("Narrative Momentum", 0.20, [
   _s("emerging","Emerging topics","emerging_topics",0.35,0,3,evidence="new themes appearing"),
   _s("kw_diversity","Keyword diversity","keyword_diversity",0.35,2,15,evidence="breadth of narrative"),
   _s("topic_conc","Topic concentration","topic_concentration",0.30,0.6,0.2,hib=False,evidence="single-theme reliance (lower=broader)"),
 ]),
 "risk_signals": ("News Risk Signals", 0.90, [
   _s("fraud_flag","Fraud/litigation flag","fraud_litigation_flag",0.35,1,0,hib=False,evidence="fraud/litigation in negative reasoning"),
   _s("disclosure_flag","Disclosure-risk flag","disclosure_risk_flag",0.30,1,0,hib=False,evidence="disclosure concerns"),
   _s("red_flags","Red-flag count","red_flag_count",0.20,2,0,hib=False,evidence="total risk flags"),
   _s("neg_reasoning","Negative reasoning count","negative_reasoning_count",0.15,10,0,hib=False,evidence="substantiated negative articles"),
 ]),
}
NEWS_INTELLIGENCE = {"label":"News Intelligence","weight":10.0,"categories":CATEGORIES}

def news_rating(score):
    if score is None: return "Unrated"
    if score>=78: return "Very Positive"
    if score>=62: return "Positive"
    if score>=45: return "Mixed"
    if score>=30: return "Negative"
    return "Very Negative"
