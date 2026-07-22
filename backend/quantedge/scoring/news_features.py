"""News Intelligence — ~52 signals from Polygon news + insights. English-only.
Sentiment from real Polygon insights. Topics/events from keyword+title parsing.
Plus a rule-based 10-point brief (upgrades to AI synthesis at tab #17). No fakes.
"""
import re, math
from datetime import datetime, timezone, timedelta
from collections import Counter

TIER1={"reuters","bloomberg","wall street journal","wsj","financial times","cnbc","the new york times",
       "associated press","barron's","forbes","marketwatch","the motley fool","seeking alpha","yahoo"}

# A handful of tickers whose editorial name genuinely differs from anything
# derivable from the ticker itself. Everything else is inferred from the corpus.
_COMPANY_ALIASES = {
    "GOOGL":["alphabet","google"], "GOOG":["alphabet","google"],
    "META":["meta","facebook"], "TSM":["tsmc","taiwan semi"],
    "BRK.B":["berkshire"], "BRK-B":["berkshire"],
}

_STOPWORDS = {"the","a","an","and","or","but","for","with","from","this","that","what","why","how",
              "is","are","was","were","be","been","has","have","had","will","would","could","should",
              "stock","stocks","shares","share","investors","investing","market","markets","best",
              "buy","sell","hold","today","now","new","one","two","three","after","before","more",
              "than","its","it","he","she","they","you","your","i","in","on","at","to","of","by",
              "up","down","says","said","just","can","not","all","out","over","into","about","q1",
              "q2","q3","q4","ceo","cfo","ai","us","etf","nasdaq","nyse","sp","dow"}

def _infer_company_name(ticker, articles):
    """Work out what the press calls this company, from the headlines themselves.
    Avoids maintaining a lookup table for 6,000 tickers: if fifteen articles about
    PLTR keep saying 'Palantir', the name is discoverable from the corpus."""
    import re as _re
    from collections import Counter as _Counter
    known = _COMPANY_ALIASES.get(ticker.upper())
    if known:
        return set(known)
    counts = _Counter()
    for a in articles[:40]:
        t = a.get("title") or ""
        # Capitalised tokens that are not sentence-initial noise
        for tok in _re.findall(r"\b[A-Z][a-zA-Z'&.-]{2,}\b", t):
            low = tok.lower().strip(".'&-")
            if low and low not in _STOPWORDS and not low.isdigit():
                counts[low] += 1
    if not counts:
        return set()
    top, n = counts.most_common(1)[0]
    # Require the candidate to appear in a meaningful share of headlines
    if n < max(2, len(articles[:40]) * 0.15):
        return set()
    names = {top}
    # Keep a two-word form if the following token also recurs (e.g. "advanced micro")
    for a in articles[:40]:
        t = (a.get("title") or "").lower()
        m = _re.search(_re.escape(top) + r"\s+([a-z][a-z'&.-]{2,})", t)
        if m and counts.get(m.group(1), 0) >= n * 0.6:
            names.add(f"{top} {m.group(1)}")
            break
    return names

# Phrases the summariser uses when the company is only mentioned in passing.
_PASSING = ("mentioned as","mentioned only","listed as","included in the broader",
            "not the focus","without additional commentary","no substantive analysis",
            "brief mention","only in a headline reference","not specifically analyzed",
            "part of the dominant","classified as")

# Events that actually move a share price.
_MATERIAL = {
    "earnings":9,"quarterly results":9,"guidance":9,"revenue":7,"margin":7,"profit":7,
    "outlook":7,"forecast":6,"beat":6,"miss":6,"downgrade":8,"upgrade":8,"price target":7,
    "acquisition":8,"merger":8,"buyback":6,"dividend":6,"lawsuit":6,"investigation":7,
    "recall":7,"antitrust":7,"regulator":6,"sec filing":6,"ceo":6,"cfo":6,"steps down":7,
    "layoff":6,"launch":5,"unveil":5,"supply":5,"shortage":6,"tariff":6,"ban":6,
}
_FLUFF = ("best stocks","should you buy","stocks to buy","etf","mutual fund","index fund",
          "10 stocks","top 5","millionaire","retire","dividend kings","motley fool picks",
          "market size worth","cagr","forecast to 2030","forecast to 2035")

# Source tiers. A wire report and a Motley Fool comparison are not equivalent inputs.
_SOURCE_TIER = {
    "reuters":95,"bloomberg":95,"wall street journal":92,"wsj":92,"financial times":92,
    "cnbc":85,"barron":85,"marketwatch":80,"associated press":90,"ap news":90,
    "investor's business daily":80,"seeking alpha":60,"benzinga":65,"investing.com":65,
    "the motley fool":55,"zacks":60,"simply wall st":50,"insider monkey":45,
    "globenewswire":70,"prnewswire":70,"businesswire":70,"accesswire":55,
}
def _source_weight(pub: str) -> int:
    p = (pub or "").lower()
    for k, v in _SOURCE_TIER.items():
        if k in p:
            return v
    return 60

# Language that distinguishes something that happened from something someone thinks.
_REPORTED = ("reported","announced","posted","filed","said","confirmed","disclosed","unveiled",
             "launched","completed","acquired","agreed","signed","raised guidance","cut guidance",
             "issued","declared","appointed","resigned","stepped down","settled","fined","sued",
             "recalled","approved","rejected","beat","missed","warned")
_SPECULATIVE = ("could","should you","is it time","prediction","predict","vs.","versus","better buy",
                "why you should","here's why","3 reasons","5 reasons","is now the","will it",
                "what to expect","preview","forecast to","best stock","top pick","worth buying",
                "my favorite","i'd buy","think about")
_ANALYST = ("price target","raises target","lowers target","upgrade","downgrade","initiated coverage",
            "reiterates","outperform","underperform","overweight","underweight","buy rating","sell rating")
_EVENT_KINDS = (
    ("EARNINGS",   ("earnings","quarterly results","q1 results","q2 results","q3 results","q4 results","eps","revenue beat","revenue miss")),
    ("GUIDANCE",   ("guidance","outlook","forecast raised","forecast cut","warns")),
    ("DEAL",       ("acquisition","acquires","merger","deal","partnership","contract","agreement","stake")),
    ("LEGAL",      ("lawsuit","sued","investigation","antitrust","doj","sec charges","fine","settlement","probe")),
    ("REGULATORY", ("fda","approval","regulator","ban","tariff","sanction","ruling")),
    ("PERSONNEL",  ("ceo","cfo","resign","steps down","appointed","named chief")),
    ("PRODUCT",    ("launch","unveil","releases","introduces","recall")),
    ("CAPITAL",    ("buyback","repurchase","dividend","split","offering","debt")),
)

def _classify(title: str, reason: str, publisher: str) -> dict:
    """Is this a reported event, an analyst action, or commentary about one?
    The distinction matters: most financial coverage is opinion about events,
    not the events themselves."""
    import re as _re
    t = (title or "").lower(); r = (reason or "").lower(); blob = t + " " + r
    has_figure = bool(_re.search(r"\$\s?[\d,.]+\s*(billion|million|trillion|bn|m\b)?|\d+(\.\d+)?\s?%|\bq[1-4]\b", blob, _re.I))
    reported = sum(1 for w in _REPORTED if w in blob)
    spec = sum(1 for w in _SPECULATIVE if w in t)
    if _re.search(r"\?\s*$", t) or _re.search(r"^\s*\w+:\s*(can|will|is|should|does|why)\b", t) \
       or _re.search(r"\b(can|will|would|might|may)\b[^.]{0,60}\b(justify|beat|reach|hit|survive|continue|last)\b", t):
        spec += 1
    analyst = any(w in blob for w in _ANALYST)

    kind = None
    for k, words in _EVENT_KINDS:
        if any(w in blob for w in words):
            kind = k; break

    announced = bool(_re.search(r"\b(announce[sd]?|unveil[sed]*|report[sed]*|post[sed]*|sign[sed]*|complete[sd]*|acquire[sd]*)\b", t))
    # Future-tense or forecast framing overrides everything: it has not happened yet.
    if _re.search(r"\b(will|would|could|should|expects? to|set to|poised to|on track to)\b", t) \
       or t.strip().startswith(("prediction","forecast","outlook:","preview")) \
       or _re.search(r"\bin (20\d\d|\w+ 20\d\d)\b", t):
        spec += 2
    if announced and kind and spec == 0:
        cls = "EVENT"
    elif analyst and not spec and _re.search(r"(price target|upgrade[sd]?|downgrade[sd]?|initiat|reiterat|raises|lowers|cuts)", t):
        cls = "ANALYST"
    elif spec >= 1 and reported == 0:
        cls = "COMMENTARY"
    elif reported >= 1 and kind and spec == 0 and (has_figure or reported >= 2):
        cls = "EVENT"
    elif spec >= 1:
        cls = "COMMENTARY"
    else:
        cls = "COMMENTARY"
    return {"class": cls, "event_kind": kind if cls == "EVENT" else None,
            "has_figure": has_figure, "source_weight": _source_weight(publisher)}

def _materiality(title: str, reason: str, publisher: str) -> tuple:
    """Score how likely an article is to actually matter for the stock.
    Returns (score 0-100, list of reasons the score fired)."""
    t, r = (title or "").lower(), (reason or "").lower()
    score, why = 40, []
    if any(p in r for p in _PASSING):
        score -= 30; why.append("passing mention")
    hits = [k for k in _MATERIAL if k in t or k in r]
    if hits:
        gain = min(45, sum(_MATERIAL[k] for k in hits[:4]))
        score += gain; why.append(", ".join(hits[:3]))
    if any(fl in t for fl in _FLUFF):
        score -= 25; why.append("listicle or promotional")
    if publisher and any(p in publisher.lower() for p in ("globenewswire","prnewswire","businesswire","accesswire")):
        score -= 12; why.append("press release wire")
    return max(0, min(100, score)), why


def _is_english(a):
    t=a.get("title") or ""   # check TITLE only - descriptions/reasoning are always English
    if not t: return True
    latin=sum(1 for ch in t if ord(ch)<0x250 or ch.isspace())
    return (latin/len(t))>0.85
def _dt(s):
    try: return datetime.fromisoformat(s.replace("Z","+00:00"))
    except: return None
def _sent_val(s):
    return {"positive":1,"negative":-1,"neutral":0}.get((s or "").lower(),0)

def compute_news_features(articles, ticker, price_return_30d=None):
    f={}; brief=[]
    now=datetime.now(timezone.utc)
    SPAM=["tokenized","rtsla","rtoken","tokenised"]  # pure noise: "mentioned as one of 100 tokenized stocks"
    arts=[]
    for a in articles:
        ins=next((i for i in (a.get("insights") or []) if i.get("ticker")==ticker), None)
        sent=ins.get("sentiment") if ins else None
        title=a.get("title",""); reason=ins.get("sentiment_reasoning","") if ins else ""
        # drop only pure tokenized-stock spam (adds no signal); KEEP foreign articles (valid sentiment)
        if any(sp in (title+reason).lower() for sp in SPAM): continue
        arts.append({"title":title,"desc":a.get("description",""),
            "pub":(a.get("publisher") or {}).get("name","") if isinstance(a.get("publisher"),dict) else "",
            "dt":_dt(a.get("published_utc","")),"kw":a.get("keywords") or [],
            "sent":sent,"sval":_sent_val(sent),"reason":reason,
            "url":a.get("article_url",""),"is_en":_is_english(a)})
    arts=[a for a in arts if a["dt"] is not None]
    arts.sort(key=lambda x:x["dt"], reverse=True)
    n=len(arts)
    if n==0: return {"available":False}
    tkr=ticker.lower()
    _company_names = _infer_company_name(ticker, articles)
    for a in arts:
        _t = a["title"].lower()
        _in_title = (tkr in _t) or any(n in _t for n in _company_names)
        a["rel_w"] = 1.0 if _in_title else 0.5
    def within(days): return [a for a in arts if (now-a["dt"]).days<=days]
    a7=within(7); a30=within(30)
    sv=[a["sval"] for a in arts]; sv30=[a["sval"] for a in a30]

    pos=sum(1 for a in arts if a["sval"]>0); neg=sum(1 for a in arts if a["sval"]<0); neu=n-pos-neg
    W=sum(a["rel_w"] for a in arts) or 1
    wpos=sum(a["rel_w"] for a in arts if a["sval"]>0); wneg=sum(a["rel_w"] for a in arts if a["sval"]<0)
    f["net_sentiment"]=(wpos-wneg)/W; f["positive_ratio"]=wpos/W; f["negative_ratio"]=wneg/W
    f["sentiment_score_mean"]=sum(a["sval"]*a["rel_w"] for a in arts)/W
    f["subject_article_count"]=sum(1 for a in arts if a["rel_w"]>=1.0)
    f["sentiment_dispersion"]=(sum((x-f["sentiment_score_mean"])**2 for x in sv)/n)**0.5
    f["bullish_bearish_ratio"]=pos/max(neg,1); f["strong_sentiment_share"]=(pos+neg)/n

    s7=[a["sval"] for a in a7]; s_prior=[a["sval"] for a in arts if 7<(now-a["dt"]).days<=30]
    f["sentiment_7d"]=sum(s7)/len(s7) if s7 else None
    f["sentiment_30d"]=sum(sv30)/len(sv30) if sv30 else None
    f["sentiment_prior"]=sum(s_prior)/len(s_prior) if s_prior else None
    if f["sentiment_7d"] is not None and f["sentiment_prior"] is not None:
        f["sentiment_trend"]=f["sentiment_7d"]-f["sentiment_prior"]
    elif f.get("sentiment_7d") is not None and f.get("sentiment_30d") is not None:
        f["sentiment_trend"]=f["sentiment_7d"]-f["sentiment_30d"]
    else:
        f["sentiment_trend"]=0.0
    f["recent_vs_baseline_sentiment"]=(f.get("sentiment_7d") or 0)-(f.get("sentiment_30d") or 0)
    f["sentiment_acceleration"]=(sum(sv[:3])/3-sum(sv[3:6])/3) if n>=6 else 0.0

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
    # event balance: positive events (upgrade, launch, guidance-up) minus negative (downgrade, lawsuit)
    pos_ev=f["upgrade_flag"]+f["product_launch_flag"]+f["guidance_change_flag"]
    neg_ev=f["downgrade_flag"]+f["lawsuit_flag"]
    f["event_balance"]=(pos_ev-neg_ev)/max(pos_ev+neg_ev,1)  # -1..+1, 0 if no events
    f["neg_event_flag"]=1.0 if neg_ev>0 else 0.0

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
    # risk severity: weighted blend of fraud/disclosure/negative density (0=clean, higher=worse)
    neg_ratio=(sum(1 for a in arts if a["sval"]<0 and a["reason"])/n) if n else 0
    f["negative_reasoning_ratio"]=neg_ratio
    f["risk_severity"]=min(1.0, 0.5*f["fraud_litigation_flag"]+0.3*f["disclosure_risk_flag"]+0.4*neg_ratio)

    # company name hints for relevance (ticker + first word of any title that repeats)
    def relevance(a):
        txt=(a["title"]+" "+a["desc"]).lower()
        # is the ticker or company clearly the SUBJECT (in title, not just mentioned)?
        return 2 if ticker.lower() in a["title"].lower() else (1 if ticker.lower() in txt else 0)
    def importance(a):
        s=relevance(a)*3  # relevance is the biggest factor
        if any(t in a["pub"].lower() for t in TIER1): s+=1
        if abs(a["sval"])>0: s+=1
        if any(t in (a["title"]+a["desc"]).lower() for t in ["earnings","lawsuit","upgrade","downgrade","guidance","launches","acquisition","deliveries","results"]): s+=2
        s+=max(0,3-(now-a["dt"]).days*0.3)
        return s
    # prefer ticker-relevant articles; dedupe by publisher to avoid all-one-source
    brief_pool=[a for a in arts if a.get("is_en")]  # readable brief = English titles only
    ranked_all=sorted(brief_pool, key=importance, reverse=True)
    seen_pub=Counter(); ranked=[]
    for a in ranked_all:
        if seen_pub[a["pub"]]>=4: continue  # max 4 per publisher in brief
        ranked.append(a); seen_pub[a["pub"]]+=1
        if len(ranked)>=10: break
    for a in ranked:
        tag={1:"positive",-1:"negative",0:"neutral"}[a["sval"]]
        brief.append({"headline":a["title"],"sentiment":tag,"publisher":a["pub"],
                      "date":a["dt"].strftime("%Y-%m-%d"),"url":a["url"]})
    f["_brief"]=brief; f["_article_count"]=n
    f["_sentiment_dist"]={"positive":pos,"neutral":neu,"negative":neg}
    # feed: only articles genuinely about the company (rel_w full) or with real sentiment, drop tangential/spam
    # feed shows English + relevant (foreign titles hidden here but their sentiment already counted above)
    for a in arts:
        _m, _w = _materiality(a["title"], a.get("reason",""), a.get("pub",""))
        _cl = _classify(a["title"], a.get("reason",""), a.get("pub",""))
        a["_cls"] = _cl
        # A reported event outranks commentary that merely discusses one.
        _bonus = {"EVENT": 25, "ANALYST": 15, "COMMENTARY": 0}[_cl["class"]]
        _src = (_cl["source_weight"] - 60) * 0.25
        a["_mat"] = max(0, min(100, _m + (12 if a["rel_w"] >= 1.0 else 0) + _bonus + _src))
        a["_mat_why"] = ([_cl["event_kind"].lower()] if _cl["event_kind"] else []) + _w
    # An article whose headline names a different company is about that company,
    # regardless of whether ours appears in the body.
    def _other_subject(a):
        t = a["title"].lower()
        if any(n in t for n in _company_names) or tkr in t:
            return False
        import re as _re2
        if _re2.match(r"^(stock market today|market wrap|markets? close|dow jones today|premarket|midday)", t):
            return True
        return bool(_re2.search(r"\b[A-Z][a-zA-Z]{2,}\s+(stock|shares|earnings)\b", a["title"]))
    feed_arts=[a for a in arts if a.get("is_en") and a["rel_w"]>=1.0 and not _other_subject(a)]
    if len(feed_arts)<6:
        _rest=sorted([a for a in arts if a.get("is_en") and a not in feed_arts],
                     key=lambda x:-x["_mat"])
        feed_arts=feed_arts+_rest[:6-len(feed_arts)]
    feed_arts=sorted(feed_arts, key=lambda x:(-x["_mat"], -x["dt"].timestamp()))
    # Same story, same outlet, reworded headline — keep the higher-scoring one.
    _seen, _dedup = [], []
    for a in feed_arts:
        _w = set(w for w in a["title"].lower().split() if len(w) > 3)
        if any(len(_w & s) / max(1, min(len(_w), len(s))) > 0.45 for s in _seen):
            continue
        _seen.append(_w); _dedup.append(a)
    feed_arts = _dedup
    # Sentiment weighted by materiality — the aggregate should reflect the articles
    # that matter, not be dominated by the filler the ranking already demoted.
    _mw = [(a["_mat"], a["sval"]) for a in arts if a.get("is_en") and a.get("_mat") is not None]
    if _mw:
        _wsum = sum(m for m, _ in _mw) or 1
        f["material_sentiment"] = round(sum(m * s for m, s in _mw) / _wsum, 4)
        _top = sorted(_mw, key=lambda x: -x[0])[:10]
        _tsum = sum(m for m, _ in _top) or 1
        f["top10_sentiment"] = round(sum(m * s for m, s in _top) / _tsum, 4)
        f["material_vs_broad_gap"] = round(f["material_sentiment"] - f.get("sentiment_score_mean", 0), 4)

    f["_recent_headlines"]=[{"title":a["title"],"sentiment":a["sent"],"reason":a["reason"][:160],
                             "publisher":a["pub"],"date":a["dt"].strftime("%Y-%m-%d"),"url":a["url"],
                             "materiality":round(a["_mat"]),"materiality_why":a["_mat_why"],
                             "about_company":a["rel_w"]>=1.0,
                             "kind":a["_cls"]["class"],"event_kind":a["_cls"]["event_kind"],
                             "source_weight":a["_cls"]["source_weight"]} for a in feed_arts[:20]]
    return f
