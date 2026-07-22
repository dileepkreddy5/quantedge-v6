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

        # ── Computed analytics from real data (no new source) ──
        _sh = [e for e in earnings[:8]]
        _beats = [e for e in _sh if (e.get("surprisePercent") or 0) > 0]
        _surprises = [e.get("surprisePercent") for e in _sh if e.get("surprisePercent") is not None]
        _beat_rate = round(len(_beats)/len(_sh), 3) if _sh else None
        _avg_surprise = round(sum(_surprises)/len(_surprises), 2) if _surprises else None
        # surprise trend: recent 2 vs older 2 (positive = improving execution)
        _surprise_trend = None
        if len(_surprises) >= 4:
            _surprise_trend = round((sum(_surprises[:2])/2) - (sum(_surprises[2:4])/2), 2)
        # consensus revision momentum: net (strongBuy+buy) share change over trend window
        _rev_mom = None; _rev_direction = "stable"
        if len(recs) >= 2:
            def _bull_share(r):
                n = sum(int(r.get(k,0) or 0) for k in ["strongBuy","buy","hold","sell","strongSell"]) or 1
                return (int(r.get("strongBuy",0) or 0) + int(r.get("buy",0) or 0)) / n
            _now = _bull_share(recs[0]); _prev = _bull_share(recs[min(3, len(recs)-1)])
            _rev_mom = round(_now - _prev, 3)
            _rev_direction = "improving" if _rev_mom > 0.02 else ("deteriorating" if _rev_mom < -0.02 else "stable")
        # rating dispersion / conviction: how concentrated (Herfindahl-style)
        _disp = None
        if n_analysts > 0:
            _shares = [c/n_analysts for c in [strong_buy, buy, hold, sell, strong_sell]]
            _hhi = sum(s*s for s in _shares)  # 1.0=all one rating (high conviction), 0.2=evenly split
            _disp = round(_hhi, 3)
        analytics = {
            "beat_rate": _beat_rate,
            "n_quarters": len(_sh),
            "avg_surprise_pct": _avg_surprise,
            "surprise_trend": _surprise_trend,
            "revision_momentum": _rev_mom,
            "revision_direction": _rev_direction,
            "rating_conviction": _disp,
            "bull_share": round((strong_buy + buy) / n_analysts, 3) if n_analysts > 0 else None,
        }
        return {
            "ticker": ticker,
            "source": "finnhub",
            "ratings": [],  # Finnhub free tier does not expose individual rows
            "analytics": analytics,
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
                # last_reported, not next. The old "date" key held last report
                # + 90 days and read as a forward date; a consumer had no way
                # to tell a fabricated date from a real one.
                "last_reported":  next_earnings_date,
                "days_since":     (-days_to if days_to is not None else None),
                "next_scheduled": None,   # needs FinnhubFeed.get_events()
                "date":           next_earnings_date,   # deprecated alias
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
        Return the LAST REPORTED period, not a guess at the next one.

        This previously returned last_report + 90 days and served it as the
        next earnings date. That is a fabricated figure: quarters are not 90
        days apart, companies move their dates, and when the estimate landed in
        the past the field shipped a negative days_to under a forward label
        (META: 2026-06-29, days_to -23).

        Finnhub does expose /calendar/earnings — FinnhubFeed.get_events()
        implements it — so a real forward date is obtainable and should be
        wired through rather than inferred here. Until then, report what is
        known and leave the future null.
        """
        if not earnings:
            return None, None
        period = earnings[0].get("period")
        if not period:
            return None, None
        try:
            last_report = datetime.strptime(period, "%Y-%m-%d").date()
            return last_report.isoformat(), -(date.today() - last_report).days
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
