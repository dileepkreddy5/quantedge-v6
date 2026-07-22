"""News Intelligence endpoint — GET /api/v6/news/{ticker}.
Fetches English Polygon news + insights, computes ~49 signals + 10-point brief,
scores the catalog. Reusable for conviction aggregator."""
from __future__ import annotations
import math, datetime as dt
from typing import Optional, Dict, Any, List
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from quantedge.scoring.news_features import compute_news_features
from quantedge.scoring.compute import score_signal
from quantedge.scoring.cat_news_v6 import CATEGORIES, news_rating
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()
_POLY="https://api.polygon.io"

def _san(o):
    if isinstance(o,float): return o if math.isfinite(o) else None
    if isinstance(o,dict): return {k:_san(v) for k,v in o.items()}
    if isinstance(o,(list,tuple)): return [_san(v) for v in o]
    return o

def _roll(children):
    scored=[c for c in children if c.get("score") is not None]
    tw=sum(c["weight"] for c in children); aw=sum(c["weight"] for c in scored)
    if not scored or tw==0: return None,0.0
    return round(sum(c["score"]*c["weight"] for c in scored)/aw,1), round(aw/tw,3)

def score_news(features):
    cats=[]
    for cid,(label,wt,sigs) in CATEGORIES.items():
        scored=[]
        for spec in sigs:
            val=features.get(spec["field"])
            res=score_signal(val,spec,None)
            scored.append({"id":spec["id"],"label":spec["label"],"weight":spec["weight"],
                           "status":spec["status"],"evidence":spec["evidence"],"raw_value":val,**res})
        cs,cc=_roll(scored)
        cats.append({"id":cid,"label":label,"weight":wt,"score":cs,"confidence":cc,
                     "n_signals":len(sigs),"n_scored":sum(1 for s in scored if s["score"] is not None),
                     "signals":scored})
    ns,nc=_roll(cats)
    return {"label":"News Intelligence","weight":10.0,"score":ns,"confidence":nc,"categories":cats}

async def _fetch_news(ticker, api_key, limit=100):
    async with httpx.AsyncClient(timeout=20) as c:
        url=f"{_POLY}/v2/reference/news?ticker={ticker}&limit={limit}&order=desc&sort=published_utc&apiKey={api_key}"
        r=await c.get(url)
        return (r.json() or {}).get("results",[]) if r.status_code==200 else []

async def _price_return_30d(ticker, api_key):
    try:
        end=dt.date.today(); start=end-dt.timedelta(days=45)
        async with httpx.AsyncClient(timeout=12) as c:
            u=f"{_POLY}/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=60&apiKey={api_key}"
            r=await c.get(u)
            if r.status_code==200:
                bars=(r.json() or {}).get("results",[])
                if len(bars)>=20: return bars[-1]["c"]/bars[0]["c"]-1
    except Exception: pass
    return None

async def compute_news_intelligence(ticker: str, api_key: str) -> Dict[str,Any]:
    ticker=ticker.upper().strip()
    articles=await _fetch_news(ticker, api_key)
    if not articles:
        return {"ticker":ticker,"available":False,"reason":"no news coverage found"}
    pr30=await _price_return_30d(ticker, api_key)
    feats=compute_news_features(articles, ticker, price_return_30d=pr30)
    if feats.get("available") is False:
        return {"ticker":ticker,"available":False,"reason":"no English-language coverage with insights"}
    tree=score_news(feats)
    n_scored=sum(c["n_scored"] for c in tree["categories"])
    n_total=sum(c["n_signals"] for c in tree["categories"])
    return {"ticker":ticker,"available":True,"intelligence":"news",
            "score":tree["score"],"confidence":tree["confidence"],
            "news_rating":news_rating(tree["score"]),
            "weight_in_conviction":10.0,
            "coverage":{"scored":n_scored,"total":n_total},
            "article_count":feats.get("_article_count"),
            "sentiment_dist":feats.get("_sentiment_dist"),
            "brief":feats.get("_brief"),
            "recent_headlines":feats.get("_recent_headlines"),
            "tree":tree,
            "key_metrics":{k:feats.get(k) for k in
                ["net_sentiment","positive_ratio","negative_ratio","sentiment_trend",
                 "article_count_7d","tier1_source_share","contrarian_signal","fraud_litigation_flag",
                 "price_return_30d","news_velocity",
                 "material_sentiment","top10_sentiment","material_vs_broad_gap"]}}

@router.get("/news/{ticker}")
async def get_news(ticker: str, http_request: Request,
                   current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    if not api_key: raise HTTPException(503,"data source unavailable")
    result=await compute_news_intelligence(ticker, api_key)
    return {"data":_san(result)}
