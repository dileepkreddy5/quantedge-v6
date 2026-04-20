"""
AnalystRatingsEngine backed by Finnhub free tier.

Replaces hardcoded fake ratings with real data:
  - Real consensus counts (buy/hold/sell/strongBuy/strongSell)
  - Real historical recommendation trends
  - Real earnings surprise data

Finnhub free tier limits:
  - 60 calls/minute (plenty of headroom with Redis caching)
  - No access to: individual analyst names, price targets, upgrade/downgrade feed

Graceful degradation: when a field is unavailable, returns None so the frontend
renders "—" instead of fake placeholder data.
"""
import os
import json
import logging
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import httpx

logger = logging.getLogger(__name__)


FINNHUB_BASE = "https://finnhub.io/api/v1"
HTTP_TIMEOUT = 8.0
DEFAULT_TTL_SECONDS = 24 * 60 * 60  # 24hr — analyst data moves slowly


SCORE_TO_LABEL = {
    1: ("STRONG SELL", "#ff6080"),
    2: ("SELL",         "#ff8090"),
    3: ("HOLD",         "#e8b84b"),
    4: ("BUY",          "#40dda0"),
    5: ("STRONG BUY",   "#00c896"),
}


class AnalystRatingsEngine:
    """
    Real analyst ratings from Finnhub free tier.

        engine = AnalystRatingsEngine(api_key=..., redis_client=...)
        result = engine.fetch("NVDA")
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        redis_client: Any = None,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> None:
        self.api_key = api_key or os.getenv("FINNHUB_API_KEY", "")
        self.redis = redis_client
        self.ttl = ttl_seconds

    def fetch(self, ticker: str) -> Dict[str, Any]:
        ticker = ticker.upper().strip()

        if not self.api_key:
            logger.warning("FINNHUB_API_KEY not set — returning empty analyst result")
            return self._empty_result(ticker)

        cached = self._cache_get(ticker)
        if cached:
            return cached

        recs = self._fetch_recommendations(ticker)
        earnings = self._fetch_earnings_surprise(ticker)

        result = self._build_result(ticker, recs, earnings)
        self._cache_set(ticker, result)
        return result

    def _fetch_recommendations(self, ticker: str) -> List[Dict[str, Any]]:
        url = f"{FINNHUB_BASE}/stock/recommendation"
        params = {"symbol": ticker, "token": self.api_key}
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                r = client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Finnhub recommendations for {ticker} failed: {e}")
            return []

    def _fetch_earnings_surprise(self, ticker: str) -> List[Dict[str, Any]]:
        url = f"{FINNHUB_BASE}/stock/earnings"
        params = {"symbol": ticker, "token": self.api_key}
        try:
            with httpx.Client(timeout=HTTP_TIMEOUT) as client:
                r = client.get(url, params=params)
                r.raise_for_status()
                data = r.json()
                return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning(f"Finnhub earnings for {ticker} failed: {e}")
            return []

    def _build_result(
        self,
        ticker: str,
        recs: List[Dict[str, Any]],
        earnings: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        latest = recs[0] if recs else {}
        strong_buy = int(latest.get("strongBuy", 0) or 0)
        buy = int(latest.get("buy", 0) or 0)
        hold = int(latest.get("hold", 0) or 0)
        sell = int(latest.get("sell", 0) or 0)
        strong_sell = int(latest.get("strongSell", 0) or 0)
        n_analysts = strong_buy + buy + hold + sell + strong_sell

        if n_analysts > 0:
            weighted = (
                strong_sell * 1 + sell * 2 + hold * 3 + buy * 4 + strong_buy * 5
            ) / n_analysts
            score_int = max(1, min(5, round(weighted)))
            label, color = SCORE_TO_LABEL[score_int]
        else:
            weighted = 3.0
            label, color = "NO COVERAGE", "#9d8b7a"

        upgrades_30d = 0
        downgrades_30d = 0
        if len(recs) >= 2:
            prev = recs[1]
            prev_n = sum(int(prev.get(k, 0) or 0) for k in ("strongBuy","buy","hold","sell","strongSell"))
            if prev_n > 0:
                prev_weighted = (
                    int(prev.get("strongSell", 0) or 0) * 1
                    + int(prev.get("sell", 0) or 0) * 2
                    + int(prev.get("hold", 0) or 0) * 3
                    + int(prev.get("buy", 0) or 0) * 4
                    + int(prev.get("strongBuy", 0) or 0) * 5
                ) / prev_n
                delta = weighted - prev_weighted
                if delta >= 0.10:
                    upgrades_30d = 1
                elif delta <= -0.10:
                    downgrades_30d = 1

        buy_count = strong_buy + buy
        sell_count = strong_sell + sell
        hold_count = hold

        next_earnings_date, days_to = self._next_earnings_date(earnings)
        eps_est, eps_prev, eps_growth = self._next_eps_from_history(earnings)
        rev_est = rev_prev = rev_growth = None

        return {
            "ticker": ticker,
            "source": "finnhub",
            "ratings": [],  # Finnhub free tier does not expose individual rows
            "consensus": {
                "label":          label,
                "color":          color,
                "score":          round(weighted, 2),
                "n_analysts":     n_analysts,
                "strong_buy":     strong_buy,
                "buy":            buy,
                "hold":           hold,
                "sell":           sell,
                "strong_sell":    strong_sell,
                "buy_count":      buy_count,
                "hold_count":     hold_count,
                "sell_count":     sell_count,
                "avg_target":     None,
                "high_target":    None,
                "low_target":     None,
                "upgrades_30d":   upgrades_30d,
                "downgrades_30d": downgrades_30d,
            },
            "trend": [
                {
                    "period":     r.get("period"),
                    "strong_buy": int(r.get("strongBuy", 0) or 0),
                    "buy":        int(r.get("buy", 0) or 0),
                    "hold":       int(r.get("hold", 0) or 0),
                    "sell":       int(r.get("sell", 0) or 0),
                    "strong_sell":int(r.get("strongSell", 0) or 0),
                }
                for r in recs[:6]
            ],
            "earnings": {
                "date":           next_earnings_date,
                "days_to":        days_to,
                "eps_estimate":   eps_est,
                "eps_prev_year":  eps_prev,
                "eps_growth":     eps_growth,
                "rev_estimate":   rev_est,
                "rev_prev_year":  rev_prev,
                "rev_growth":     rev_growth,
                "surprise_history": [
                    {
                        "period":           e.get("period"),
                        "actual":           e.get("actual"),
                        "estimate":         e.get("estimate"),
                        "surprise_percent": e.get("surprisePercent"),
                    }
                    for e in earnings[:4]
                ],
            },
        }

    def _next_earnings_date(self, earnings: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[int]]:
        """
        Finnhub free tier doesn't have a forward earnings calendar endpoint.
        Estimate next earnings by adding ~90 days to the most recent report.
        """
        if not earnings:
            return None, None
        latest = earnings[0]
        period = latest.get("period")
        if not period:
            return None, None
        try:
            last_report = datetime.strptime(period, "%Y-%m-%d").date()
            next_est = last_report + timedelta(days=90)
            days_to = (next_est - date.today()).days
            return next_est.isoformat(), days_to
        except Exception:
            return None, None

    def _next_eps_from_history(
        self, earnings: List[Dict[str, Any]]
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Forward EPS estimate not in free tier. Use most recent actual as the
        "previous" comparison point; growth computed vs same quarter last year
        (idx 4 if available).
        """
        if not earnings:
            return None, None, None
        latest_actual = earnings[0].get("actual")
        prev_year = None
        if len(earnings) >= 5:
            prev_year = earnings[4].get("actual")
        eps_growth = None
        if latest_actual is not None and prev_year and prev_year != 0:
            eps_growth = round((latest_actual - prev_year) / abs(prev_year) * 100, 1)
        return None, latest_actual, eps_growth

    def _cache_get(self, ticker: str) -> Optional[Dict[str, Any]]:
        if not self.redis:
            return None
        try:
            key = f"analyst_ratings:finnhub:{ticker}"
            raw = self.redis.get(key)
            if raw:
                return json.loads(raw)
        except Exception as e:
            logger.debug(f"Cache get failed for {ticker}: {e}")
        return None

    def _cache_set(self, ticker: str, payload: Dict[str, Any]) -> None:
        if not self.redis:
            return
        try:
            key = f"analyst_ratings:finnhub:{ticker}"
            self.redis.setex(key, self.ttl, json.dumps(payload, default=str))
        except Exception as e:
            logger.debug(f"Cache set failed for {ticker}: {e}")

    def _empty_result(self, ticker: str) -> Dict[str, Any]:
        return {
            "ticker": ticker,
            "source": "unavailable",
            "ratings": [],
            "consensus": {
                "label": "NO COVERAGE",
                "color": "#9d8b7a",
                "score": 3.0,
                "n_analysts": 0,
                "strong_buy": 0, "buy": 0, "hold": 0, "sell": 0, "strong_sell": 0,
                "buy_count": 0, "hold_count": 0, "sell_count": 0,
                "avg_target": None, "high_target": None, "low_target": None,
                "upgrades_30d": 0, "downgrades_30d": 0,
            },
            "trend": [],
            "earnings": {
                "date": None, "days_to": None,
                "eps_estimate": None, "eps_prev_year": None, "eps_growth": None,
                "rev_estimate": None, "rev_prev_year": None, "rev_growth": None,
                "surprise_history": [],
            },
        }
