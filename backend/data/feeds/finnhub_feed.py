"""
QuantEdge v6.0 — Finnhub Feed (free tier)
==========================================
Real structured data the Polygon $29 plan lacks:
  - upcoming + recent earnings dates (calendar/earnings)
  - EPS actual-vs-estimate surprises (stock/earnings)
  - analyst buy/hold/sell recommendation trend (stock/recommendation)

Free tier: 60 calls/min, auth via ?token=KEY. Base https://finnhub.io/api/v1
All methods are best-effort: any failure returns an empty/neutral result so the
news endpoint never breaks because Finnhub hiccuped or rate-limited.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import httpx
from loguru import logger

FINNHUB_BASE = "https://finnhub.io/api/v1"


class FinnhubFeed:
    def __init__(self, api_key: str, redis_client=None):
        self.api_key = api_key or ""
        self.redis = redis_client

    async def _get(self, client: httpx.AsyncClient, path: str, params: Dict) -> Optional[dict]:
        if not self.api_key:
            return None
        params = {**params, "token": self.api_key}
        try:
            r = await client.get(f"{FINNHUB_BASE}{path}", params=params, timeout=12)
            if r.status_code == 429:
                logger.warning("Finnhub rate-limited (429)")
                return None
            if r.status_code != 200:
                return None
            return r.json()
        except Exception as e:
            logger.warning(f"Finnhub {path} error: {e}")
            return None

    async def get_events(self, ticker: str) -> Dict:
        ticker = ticker.upper().strip()
        cache_key = f"finnhub:events:v1:{ticker}"
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        out: Dict = {
            "next_earnings_date": None,
            "last_eps_actual": None,
            "last_eps_estimate": None,
            "eps_surprise_pct": None,
            "analyst": None,
        }

        today = datetime.now(timezone.utc).date()
        frm = (today - timedelta(days=7)).isoformat()
        to = (today + timedelta(days=120)).isoformat()

        async with httpx.AsyncClient() as client:
            cal = await self._get(client, "/calendar/earnings",
                                  {"symbol": ticker, "from": frm, "to": to})
            if cal and isinstance(cal.get("earningsCalendar"), list):
                upcoming = [e for e in cal["earningsCalendar"]
                            if e.get("date") and e["date"] >= today.isoformat()]
                upcoming.sort(key=lambda e: e["date"])
                if upcoming:
                    out["next_earnings_date"] = upcoming[0]["date"]

            hist = await self._get(client, "/stock/earnings", {"symbol": ticker, "limit": 1})
            if isinstance(hist, list) and hist:
                h = hist[0]
                act, est = h.get("actual"), h.get("estimate")
                out["last_eps_actual"] = act
                out["last_eps_estimate"] = est
                if act is not None and est not in (None, 0):
                    try:
                        out["eps_surprise_pct"] = round((act - est) / abs(est) * 100, 1)
                    except Exception:
                        pass

            rec = await self._get(client, "/stock/recommendation", {"symbol": ticker})
            if isinstance(rec, list) and rec:
                r0 = rec[0]
                out["analyst"] = {
                    "strongBuy": r0.get("strongBuy", 0),
                    "buy": r0.get("buy", 0),
                    "hold": r0.get("hold", 0),
                    "sell": r0.get("sell", 0),
                    "strongSell": r0.get("strongSell", 0),
                    "period": r0.get("period", ""),
                }

        if self.redis:
            try:
                await self.redis.setex(cache_key, 3600, json.dumps(out))
            except Exception:
                pass
        return out

    @staticmethod
    def events_to_facts(ev: Dict) -> List[str]:
        facts = []
        if ev.get("next_earnings_date"):
            facts.append(f"Next earnings date: {ev['next_earnings_date']}.")
        if ev.get("eps_surprise_pct") is not None:
            a, e = ev.get("last_eps_actual"), ev.get("last_eps_estimate")
            direction = "beat" if ev["eps_surprise_pct"] >= 0 else "missed"
            facts.append(
                f"Most recent quarter: EPS {a} vs {e} estimate "
                f"({direction} by {abs(ev['eps_surprise_pct'])}%)."
            )
        an = ev.get("analyst")
        if an:
            total = an["strongBuy"] + an["buy"] + an["hold"] + an["sell"] + an["strongSell"]
            if total > 0:
                buys = an["strongBuy"] + an["buy"]
                sells = an["sell"] + an["strongSell"]
                facts.append(
                    f"Analyst ratings ({an.get('period','recent')}): "
                    f"{buys} buy, {an['hold']} hold, {sells} sell (of {total} analysts)."
                )
        return facts
