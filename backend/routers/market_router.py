"""Market Intelligence endpoint — GET /api/v6/market/{ticker}.
Peer-relative scoring of momentum/trend/liquidity from the pre-computed peer_stats
bucketed factors, plus a fresh GARCH/HMM/Kalman regime read from price history.
Single ownership: price-based signals only. Reusable for the conviction aggregator.
"""
from __future__ import annotations
import math, datetime as dt
from typing import Optional, Dict, Any, List
import httpx
from fastapi import APIRouter, HTTPException, Request, Depends
from loguru import logger

from quantedge.scoring.market_features import score_market, market_rating
from auth.cognito_auth import get_optional_user, CognitoUser
from core.config import settings

router = APIRouter()
_POLY = "https://api.polygon.io"

def _san(o):
    if isinstance(o, float): return o if math.isfinite(o) else None
    if isinstance(o, dict): return {k: _san(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_san(v) for v in o]
    return o

async def _regime_read(ticker: str, api_key: str) -> Dict[str, Any]:
    """Fresh GARCH volatility + HMM regime from ~1yr daily returns (reuse real engines)."""
    out={}
    try:
        import pandas as pd, numpy as np
        end=dt.date.today(); start=end-dt.timedelta(days=420)
        async with httpx.AsyncClient(timeout=15) as c:
            u=f"{_POLY}/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=500&apiKey={api_key}"
            r=await c.get(u)
            if r.status_code!=200: return out
            bars=(r.json() or {}).get("results",[])
        if len(bars)<60: return out
        closes=pd.Series([b["c"] for b in bars])
        volume=pd.Series([b.get("v",0) for b in bars])
        returns=closes.pct_change().dropna()
        try:
            from ml.models.regime_volatility import GJRGARCHModel, HMMRegimeClassifier, KalmanTrendFilter
            g=GJRGARCHModel(); gr=g.fit(returns)
            out["garch"]={"current_vol":gr.get("current_vol") or gr.get("annualized_vol"),
                          "vol_regime":gr.get("vol_regime") or gr.get("regime")}
            h=HMMRegimeClassifier(); h.fit(returns, volume)
            hr=h.predict_current_regime(returns, volume)
            out["regime"]={"current":hr.get("current_regime"),"confidence":hr.get("confidence") or hr.get("probability")}
            k=KalmanTrendFilter(); kr=k.fit(returns)
            out["kalman"]={"trend":kr.get("trend") or kr.get("trend_direction"),"state":kr.get("state")}
        except Exception as e:
            logger.info(f"market regime engines: {e}")
    except Exception as e:
        logger.info(f"market regime read failed {ticker}: {e}")
    return out

async def _benchmarks_and_position(ticker: str, api_key: str) -> Dict[str, Any]:
    """Relative strength vs SPY/QQQ/XLK + 52-week price position. Real from Polygon aggs."""
    out={"relative_strength":{}, "price_position":{}}
    try:
        end=dt.date.today(); start=end-dt.timedelta(days=400)
        async with httpx.AsyncClient(timeout=15) as c:
            async def series(sym):
                u=f"{_POLY}/v2/aggs/ticker/{sym}/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=500&apiKey={api_key}"
                r=await c.get(u)
                if r.status_code!=200: return []
                return [b["c"] for b in (r.json() or {}).get("results",[])]
            stock=await series(ticker)
            if len(stock)<60: return out
            # 52-week position
            hi=max(stock); lo=min(stock); cur=stock[-1]
            out["price_position"]={"pct_from_52w_high":round((cur-hi)/hi*100,1),
                "pct_from_52w_low":round((cur-lo)/lo*100,1),
                "range_percentile":round((cur-lo)/(hi-lo)*100,1) if hi>lo else None}
            # relative strength: 3-month return vs benchmarks
            def ret_3m(px): 
                n=min(63,len(px)-1); return (px[-1]/px[-1-n]-1) if n>0 else None
            s3=ret_3m(stock)
            for bench in ["SPY","QQQ","XLK"]:
                bpx=await series(bench)
                b3=ret_3m(bpx) if len(bpx)>60 else None
                if s3 is not None and b3 is not None:
                    out["relative_strength"][bench]=round((s3-b3)*100,1)
    except Exception as e:
        logger.info(f"market benchmarks failed {ticker}: {e}")
    return out

def _market_reasons(tree, ladder, rg):
    """Rule-based explanation of the market read from real signals."""
    r=[]
    m6=ladder.get("mom_6m"); m12=ladder.get("mom_12_1"); m1=ladder.get("mom_1m")
    if m6 is not None and m6<-5: r.append(f"Lagging 6-month ({m6:.0f}%)")
    if m12 is not None and m12<-10: r.append(f"Weak 12-month trend ({m12:.0f}%)")
    if m1 is not None and m1>0 and m6 is not None and m6<0: r.append("Short-term bounce vs weak medium-term")
    liq=next((c["score"] for c in tree["categories"] if c["id"]=="liquidity_flow"),None)
    if liq is not None and liq>60: r.append("Healthy liquidity (not distribution-driven)")
    if rg.get("regime",{}).get("current","").startswith("BULL"): r.append("Market regime still constructive")
    if not r: r.append("Market signals broadly neutral")
    return r

async def _price_volume_si(ticker: str, api_key: str):
    """Fetch ~1.5yr daily closes+volumes + latest short-interest records. Real."""
    closes=[]; volumes=[]; si_records=[]
    try:
        end=dt.date.today(); start=end-dt.timedelta(days=550)
        async with httpx.AsyncClient(timeout=15) as c:
            u=f"{_POLY}/v2/aggs/ticker/{ticker}/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=800&apiKey={api_key}"
            r=await c.get(u)
            if r.status_code==200:
                bars=(r.json() or {}).get("results",[])
                closes=[b["c"] for b in bars]; volumes=[b.get("v",0) for b in bars]
            # short interest (latest few records)
            rs=await c.get(f"{_POLY}/stocks/v1/short-interest?ticker={ticker}&limit=12&sort=settlement_date.desc&apiKey={api_key}")
            if rs.status_code==200:
                si_records=(rs.json() or {}).get("results",[])
    except Exception as e:
        logger.info(f"market price/volume/si fetch failed {ticker}: {e}")
    return closes, volumes, si_records

async def _spy_closes(api_key: str):
    try:
        end=dt.date.today(); start=end-dt.timedelta(days=550)
        async with httpx.AsyncClient(timeout=12) as c:
            u=f"{_POLY}/v2/aggs/ticker/SPY/range/1/day/{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=800&apiKey={api_key}"
            r=await c.get(u)
            if r.status_code==200: return [b["c"] for b in (r.json() or {}).get("results",[])]
    except Exception: pass
    return []

async def compute_market_intelligence(ticker: str, api_key: str, pool=None) -> Dict[str, Any]:
    ticker=ticker.upper().strip()
    me_factors=None; peer_list=[]; bucket=None
    if pool is not None:
        try:
            from services.peer_store import PeerStore
            pdata=await PeerStore(pool).get_peers(ticker)
            if pdata.get("available"):
                bucket=pdata.get("bucket")
                me_row=pdata.get("me",{})
                me_factors=me_row.get("factors") or {}
                if isinstance(me_factors,str):
                    import json; me_factors=json.loads(me_factors)
                for row in pdata.get("peers",[]):
                    pf=row.get("factors") or {}
                    if isinstance(pf,str):
                        import json; pf=json.loads(pf)
                    peer_list.append(pf)
        except Exception as e:
            logger.info(f"market: peers unavailable {ticker}: {e}")
    if not me_factors:
        return {"ticker":ticker,"available":False,"reason":"ticker not in peer universe (run the peer scan)"}
    tree=score_market(me_factors, peer_list)
    regime=await _regime_read(ticker, api_key)
    # Deep signals — volatility, trading risk, volume, short interest (all real)
    from quantedge.scoring.market_deep import volatility_suite, trading_risk, volume_liquidity, short_interest_signals
    closes, volumes, si_records = await _price_volume_si(ticker, api_key)
    spy = await _spy_closes(api_key)
    deep={}
    if len(closes)>=60:
        deep["volatility"]=volatility_suite(closes, spy)
        deep["trading_risk"]=trading_risk(closes)
        deep["volume"]=volume_liquidity(closes, volumes)
    deep["short_interest"]=short_interest_signals(si_records)
    bench=await _benchmarks_and_position(ticker, api_key)
    n_scored=sum(c["n_scored"] for c in tree["categories"])
    n_total=sum(c["n_signals"] for c in tree["categories"])
    # momentum ladder for the frontend
    ladder={k:me_factors.get(k) for k in ["mom_1m","mom_3m","mom_6m","mom_12_1"]}
    return {"ticker":ticker,"available":True,"intelligence":"market",
            "score":tree["score"],"confidence":tree["confidence"],
            "market_rating":market_rating(tree["score"]),
            "weight_in_conviction":5.0,"sector_bucket":bucket,
            "peer_count":len(peer_list),
            "coverage":{"scored":n_scored,"total":n_total},
            "tree":tree,"regime":regime,"momentum_ladder":ladder,
            "volatility":deep.get("volatility"),"trading_risk":deep.get("trading_risk"),
            "volume":deep.get("volume"),"short_interest":deep.get("short_interest"),
            "relative_strength":bench.get("relative_strength"),"price_position":bench.get("price_position"),
            "reasons":_market_reasons(tree, ladder, regime),
            "key_metrics":{"hurst":me_factors.get("hurst"),"sharpe_3m":me_factors.get("sharpe_3m"),
                "ma_alignment":me_factors.get("ma_alignment"),"amihud":me_factors.get("amihud"),
                "pct_above_ma50":me_factors.get("pct_above_ma50"),"pct_above_ma200":me_factors.get("pct_above_ma200")}}

@router.get("/market/{ticker}")
async def get_market(ticker: str, http_request: Request,
                     current_user: Optional[CognitoUser]=Depends(get_optional_user)):
    api_key=getattr(settings,"POLYGON_API_KEY","") or ""
    pool=getattr(http_request.app.state,"db",None)
    result=await compute_market_intelligence(ticker, api_key, pool)
    return {"data":_san(result)}
