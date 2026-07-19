"""News Intelligence — ~52 signals from Polygon news + insights. English-only.
Sentiment from real Polygon insights. Topics/events from keyword+title parsing.
Plus a rule-based 10-point brief (upgrades to AI synthesis at tab #17). No fakes.
"""
import re, math
from datetime import datetime, timezone, timedelta
from collections import Counter

TIER1={"reuters","bloomberg","wall street journal","wsj","financial times","cnbc","the new york times",
       "associated press","barron's","forbes","marketwatch","the motley fool","seeking alpha","yahoo"}
NON_ENGLISH_HINT=re.compile(r'[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]')

def _is_english(a):
    t=(a.get("title") or "")+(a.get("description") or "")
    return not NON_ENGLISH_HINT.search(t)
def _dt(s):
    try: return datetime.fromisoformat(s.replace("Z","+00:00"))
    except: return None
def _sent_val(s):
    return {"positive":1,"negative":-1,"neutral":0}.get((s or "").lower(),0)

def compute_news_features(articles, ticker, price_return_30d=None):
    f={}; brief=[]
    now=datetime.now(timezone.utc)
    arts=[]
    for a in articles:
        if not _is_english(a): continue
        ins=next((i for i in (a.get("insights") or []) if i.get("ticker")==ticker), None)
        sent=ins.get("sentiment") if ins else None
        arts.append({"title":a.get("title",""),"desc":a.get("description",""),
            "pub":(a.get("publisher") or {}).get("name","") if isinstance(a.get("publisher"),dict) else "",
            "dt":_dt(a.get("published_utc","")),"kw":a.get("keywords") or [],
            "sent":sent,"sval":_sent_val(sent),"reason":ins.get("sentiment_reasoning","") if ins else "",
            "url":a.get("article_url","")})
    arts=[a for a in arts if a["dt"] is not None]
    arts.sort(key=lambda x:x["dt"], reverse=True)
    n=len(arts)
    if n==0: return {"available":False}
    def within(days): return [a for a in arts if (now-a["dt"]).days<=days]
    a7=within(7); a30=within(30)
    sv=[a["sval"] for a in arts]; sv30=[a["sval"] for a in a30]

    pos=sum(1 for a in arts if a["sval"]>0); neg=sum(1 for a in arts if a["sval"]<0); neu=n-pos-neg
    f["net_sentiment"]=(pos-neg)/n; f["positive_ratio"]=pos/n; f["negative_ratio"]=neg/n
    f["sentiment_score_mean"]=sum(sv)/n
    f["sentiment_dispersion"]=(sum((x-f["sentiment_score_mean"])**2 for x in sv)/n)**0.5
    f["bullish_bearish_ratio"]=pos/max(neg,1); f["strong_sentiment_share"]=(pos+neg)/n

    s7=[a["sval"] for a in a7]; s_prior=[a["sval"] for a in arts if 7<(now-a["dt"]).days<=30]
    f["sentiment_7d"]=sum(s7)/len(s7) if s7 else None
    f["sentiment_30d"]=sum(sv30)/len(sv30) if sv30 else None
    f["sentiment_prior"]=sum(s_prior)/len(s_prior) if s_prior else None
    if f["sentiment_7d"] is not None and f["sentiment_prior"] is not None:
        f["sentiment_trend"]=f["sentiment_7d"]-f["sentiment_prior"]
    f["recent_vs_baseline_sentiment"]=(f.get("sentiment_7d") or 0)-(f.get("sentiment_30d") or 0)
    if n>=6: f["sentiment_acceleration"]=sum(sv[:3])/3-sum(sv[3:6])/3

    f["article_count_7d"]=len(a7); f["article_count_30d"]=len(a30)
    baseline=len(a30)/30 if a30 else 0
    f["news_velocity"]=len(a7)/7
    f["volume_vs_baseline"]=(len(a7)/7)/baseline if baseline>0 else None
    f["coverage_intensity"]=n
    f["volume_spike"]=1.0 if (baseline>0 and len(a7)/7>2*baseline) else 0.0

    pubs=[a["pub"] for a in arts if a["pub"]]
    f["unique_publishers"]=len(set(pubs))
    tier1=sum(1 for p in pubs if any(t in p.lower() for t in TIER1))
    f["tier1_source_share"]=tier1/len(pubs) if pubs else None
    f["publisher_diversity"]=len(set(pubs))/len(pubs) if pubs else None
    pc=Counter(pubs); f["source_concentration"]=(pc.most_common(1)[0][1]/len(pubs)) if pubs else None

    alltext=" ".join((a["title"]+" "+a["desc"]+" "+" ".join(a["kw"])).lower() for a in arts)
    def cnt(terms): return sum(alltext.count(t) for t in terms)/max(n,1)
    f["earnings_mentions"]=cnt(["earnings","eps","revenue","quarter","profit"])
    f["product_mentions"]=cnt(["product","launch","release","copilot","azure","cloud"])
    f["legal_mentions"]=cnt(["lawsuit","litigation","sec ","fraud","investigation","regulat","antitrust"])
    f["ma_mentions"]=cnt(["acquisition","merger","acquire","buyout"])
    f["management_mentions"]=cnt(["ceo","cfo","executive","management","board"])
    f["guidance_mentions"]=cnt(["guidance","forecast","outlook"])
    f["analyst_mentions"]=cnt(["analyst","upgrade","downgrade","price target","rating"])
    f["competitive_mentions"]=cnt(["competit","rival","market share"])

    def flag(terms): return 1.0 if any(t in alltext for t in terms) else 0.0
    f["lawsuit_flag"]=flag(["class action","lawsuit","securities fraud"])
    f["downgrade_flag"]=flag(["downgrade","cut to sell"])
    f["upgrade_flag"]=flag(["upgrade","raised to buy","initiated buy"])
    f["guidance_change_flag"]=flag(["raises guidance","cuts guidance"])
    f["product_launch_flag"]=flag(["launches","unveils"])
    f["material_event_score"]=(f["lawsuit_flag"]+f["downgrade_flag"]+f["upgrade_flag"]+f["guidance_change_flag"])/4

    if price_return_30d is not None:
        f["price_return_30d"]=price_return_30d; ns=f["net_sentiment"]
        f["sentiment_price_divergence"]=ns-(1 if price_return_30d>0 else -1 if price_return_30d<0 else 0)
        f["contrarian_signal"]=1.0 if (ns>0.2 and price_return_30d<-0.05) else (-1.0 if (ns<-0.2 and price_return_30d>0.05) else 0.0)
        f["confirmation_score"]=1.0 if (ns>0)==(price_return_30d>0) else 0.0

    latest=arts[0]["dt"]
    f["hours_since_latest"]=(now-latest).total_seconds()/3600
    f["fresh_news_share"]=len([a for a in arts if (now-a["dt"]).days<=2])/n
    f["stale_flag"]=1.0 if f["hours_since_latest"]>72 else 0.0
    f["latest_sentiment"]=arts[0]["sval"]

    kw7=Counter(); kw_prior=Counter()
    for a in a7:
        for k in a["kw"]: kw7[k.lower()]+=1
    for a in [x for x in arts if 7<(now-x["dt"]).days<=30]:
        for k in a["kw"]: kw_prior[k.lower()]+=1
    f["emerging_topics"]=len([k for k in kw7 if kw7[k]>=2 and kw_prior.get(k,0)==0])
    f["topic_concentration"]=(kw7.most_common(1)[0][1]/sum(kw7.values())) if kw7 else None
    f["keyword_diversity"]=len(kw7)

    neg_reasons=" ".join(a["reason"].lower() for a in arts if a["sval"]<0)
    f["fraud_litigation_flag"]=1.0 if any(t in neg_reasons for t in ["fraud","litigation","class action","misstatement"]) else 0.0
    f["disclosure_risk_flag"]=1.0 if any(t in neg_reasons for t in ["disclosure","undisclosed","misled","material"]) else 0.0
    f["negative_reasoning_count"]=sum(1 for a in arts if a["sval"]<0 and a["reason"])
    f["red_flag_count"]=f["fraud_litigation_flag"]+f["disclosure_risk_flag"]

    def importance(a):
        s=0
        if any(t in a["pub"].lower() for t in TIER1): s+=2
        if abs(a["sval"])>0: s+=1
        if any(t in (a["title"]+a["desc"]).lower() for t in ["earnings","lawsuit","upgrade","downgrade","guidance","launches","acquisition"]): s+=2
        s+=max(0,3-(now-a["dt"]).days*0.3)
        return s
    ranked=sorted(arts, key=importance, reverse=True)[:10]
    for a in ranked:
        tag={1:"positive",-1:"negative",0:"neutral"}[a["sval"]]
        brief.append({"headline":a["title"],"sentiment":tag,"publisher":a["pub"],
                      "date":a["dt"].strftime("%Y-%m-%d"),"url":a["url"]})
    f["_brief"]=brief; f["_article_count"]=n
    f["_sentiment_dist"]={"positive":pos,"neutral":neu,"negative":neg}
    f["_recent_headlines"]=[{"title":a["title"],"sentiment":a["sent"],"reason":a["reason"][:160],
                             "publisher":a["pub"],"date":a["dt"].strftime("%Y-%m-%d"),"url":a["url"]} for a in arts[:15]]
    return f
