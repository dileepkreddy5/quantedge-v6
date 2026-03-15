"""
QuantEdge v6.0 — Market Data Feed
====================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.

Thin wrapper classes that delegate to polygon_feed.py.
Class names and method signatures are identical to v5.0 so
analysis_v6.py imports work without any changes.

Data source: Polygon.io REST API (Starter Plus — $29/mo)
yFinance is completely removed.

Classes (backward-compatible interface):
    MarketDataFeed      → wraps PolygonMarketFeed
    FundamentalDataFeed → wraps PolygonFundamentalFeed
    OptionsDataFeed     → wraps PolygonOptionsFeed
    SentimentDataFeed   → wraps PolygonNewsFeed (news only; Reddit removed)

Redis client is optional. When None, Polygon calls are made without caching.
The redis client is injected at runtime by QuantEdgeAnalyzerV6 when available.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from loguru import logger

from core.config import settings
from data.feeds.polygon_feed import (
    PolygonMarketFeed,
    PolygonFundamentalFeed,
    PolygonOptionsFeed,
    PolygonNewsFeed,
)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _polygon_api_key() -> str:
    """Return POLYGON_API_KEY from settings; raise clearly if missing."""
    key = settings.POLYGON_API_KEY
    if not key:
        raise RuntimeError(
            "POLYGON_API_KEY is not set. "
            "Add it to your environment or AWS Secrets Manager."
        )
    return key


# ---------------------------------------------------------------------------
# MarketDataFeed
# ---------------------------------------------------------------------------

class MarketDataFeed:
    """
    Primary market data source backed by Polygon.io.
    Backward-compatible replacement for yFinance-based v5 class.
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client

    def _feed(self) -> PolygonMarketFeed:
        return PolygonMarketFeed(
            api_key=_polygon_api_key(),
            redis_client=self._redis,
        )

    async def get_price_history(
        self,
        ticker: str,
        years: int = 10,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data from Polygon.
        Returns DataFrame with lowercase columns: open, high, low, close, volume, returns
        Caches in Redis with key polygon:v1:{ticker}:ohlcv (TTL 1 hour).
        Retries with exponential backoff on HTTP 429.
        """
        try:
            df = await self._feed().get_price_history(ticker=ticker, years=years)
            if df.empty:
                logger.warning(f"Empty price history returned for {ticker}")
                return pd.DataFrame()

            # Normalise column names to lowercase
            df.columns = [c.lower() for c in df.columns]

            # Ensure 'returns' column exists
            if "returns" not in df.columns:
                if "close" in df.columns:
                    df["returns"] = df["close"].pct_change().fillna(0)

            logger.info(
                f"✅ Price data: {ticker} — {len(df)} rows, "
                f"{df.index[0].date()} to {df.index[-1].date()}"
            )
            return df

        except Exception as exc:
            logger.error(f"Price history error {ticker}: {exc}")
            return pd.DataFrame()

    async def get_realtime_quote(self, ticker: str) -> Dict:
        """
        Current price, volume, bid/ask from Polygon snapshot API.
        Falls back to last close from price history if snapshot unavailable.
        """
        try:
            feed = self._feed()
            if hasattr(feed, "get_snapshot"):
                snap = await feed.get_snapshot(ticker)
                if snap:
                    return snap

            # Fallback: pull last row from price history
            df = await self.get_price_history(ticker, years=1)
            if df.empty:
                return {}

            last = df.iloc[-1]
            return {
                "price": float(last.get("close", 0)),
                "open": float(last.get("open", 0)),
                "high": float(last.get("high", 0)),
                "low": float(last.get("low", 0)),
                "volume": float(last.get("volume", 0)),
                "avg_volume": None,
                "market_cap": None,
                "bid": None,
                "ask": None,
                "bid_size": None,
                "ask_size": None,
                "pre_market": None,
                "after_hours": None,
            }

        except Exception as exc:
            logger.error(f"Realtime quote error {ticker}: {exc}")
            return {}

    def inject_redis(self, redis_client) -> None:
        """Called after app.state.redis is available."""
        self._redis = redis_client


# ---------------------------------------------------------------------------
# FundamentalDataFeed
# ---------------------------------------------------------------------------

class FundamentalDataFeed:
    """Fundamental data backed by Polygon.io financials API."""

    def __init__(self, redis_client=None):
        self._redis = redis_client

    def _feed(self) -> PolygonFundamentalFeed:
        return PolygonFundamentalFeed(
            api_key=_polygon_api_key(),
            redis_client=self._redis,
        )

    async def get_fundamentals(self, ticker: str) -> Dict:
        """
        Returns fundamental dict.
        Required keys per spec: pe_ratio, pb_ratio, eps_ttm, revenue_growth,
        debt_to_equity, current_ratio, roe, roa, gross_margin, market_cap
        Caches with key polygon:v1:{ticker}:fundamentals (TTL 24 hours).
        """
        try:
            data = await self._feed().get_fundamentals(ticker=ticker)
            if not data:
                logger.warning(f"Empty fundamentals returned for {ticker}")
                return {"name": ticker}

            normalised: Dict = {}

            # Identity
            normalised["name"] = data.get("name") or ticker
            for k in ("sector", "industry", "exchange", "country"):
                if data.get(k):
                    normalised[k] = data[k]
            normalised["description"] = (data.get("description", "") or "")[:500]

            # All numeric keys — copy if present
            numeric_keys = [
                "pe_ratio", "pb_ratio", "price_to_book", "forward_pe",
                "peg_ratio", "price_to_sales", "ev_ebitda", "ev_revenue",
                "enterprise_value", "market_cap",
                "gross_margin", "operating_margin", "ebitda_margin",
                "net_margin", "roe", "roa", "roic",
                "revenue_growth", "earnings_growth", "revenue_ttm",
                "ebitda", "free_cash_flow", "eps_ttm", "eps_forward",
                "total_debt", "total_cash", "debt_to_equity",
                "current_ratio", "quick_ratio",
                "dividend_yield", "dividend_rate", "payout_ratio",
                "week_52_high", "week_52_low", "beta",
                "short_interest", "shares_outstanding", "float_shares",
                "institutional_ownership", "insider_ownership",
            ]
            for k in numeric_keys:
                v = data.get(k)
                if v is not None:
                    try:
                        normalised[k] = float(v)
                    except (TypeError, ValueError):
                        pass

            # Alias: price_to_book → pb_ratio
            if "price_to_book" in normalised and "pb_ratio" not in normalised:
                normalised["pb_ratio"] = normalised["price_to_book"]

            # Earnings date
            if data.get("earnings_date"):
                normalised["earnings_date"] = str(data["earnings_date"])

            # Derived metrics
            if normalised.get("free_cash_flow") and normalised.get("market_cap"):
                normalised["fcf_yield"] = (
                    normalised["free_cash_flow"] / normalised["market_cap"]
                )
            if normalised.get("free_cash_flow") and normalised.get("revenue_ttm"):
                normalised["fcf_margin"] = (
                    normalised["free_cash_flow"] / normalised["revenue_ttm"]
                )

            return {k: v for k, v in normalised.items() if v is not None}

        except Exception as exc:
            logger.error(f"Fundamentals error {ticker}: {exc}")
            return {"name": ticker}

    def inject_redis(self, redis_client) -> None:
        self._redis = redis_client


# ---------------------------------------------------------------------------
# OptionsDataFeed
# ---------------------------------------------------------------------------

class OptionsDataFeed:
    """Options chain data backed by Polygon.io options API."""

    def __init__(self, redis_client=None):
        self._redis = redis_client

    def _feed(self) -> PolygonOptionsFeed:
        return PolygonOptionsFeed(
            api_key=_polygon_api_key(),
            redis_client=self._redis,
        )

    async def get_chain(self, ticker: str, max_expiries: int = 6) -> pd.DataFrame:
        """
        Returns options chain DataFrame.
        Columns: strike, expiry, call_iv, put_iv,
                 call_delta, put_delta, call_gamma, put_gamma,
                 call_volume, put_volume, call_oi, put_oi,
                 days_to_expiry, iv, open_interest_call, open_interest_put,
                 volume_call, volume_put
        Caches with key polygon:v1:{ticker}:options (TTL 15 minutes).
        """
        try:
            df = await self._feed().get_chain(ticker=ticker)
            if df.empty:
                logger.warning(f"Empty options chain for {ticker}")
                return pd.DataFrame()

            # Normalise column name aliases
            col_aliases = {
                "impliedVolatility": "iv",
                "implied_volatility": "iv",
                "openInterest": "open_interest_call",
                "volume": "volume_call",
            }
            df = df.rename(columns=col_aliases)

            # Ensure backward-compat columns exist with NaN defaults
            required_cols = [
                "call_iv", "put_iv",
                "call_delta", "put_delta",
                "call_gamma", "put_gamma",
                "call_volume", "put_volume",
                "call_oi", "put_oi",
                "strike", "expiry", "days_to_expiry",
            ]
            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan

            # Keep only rows with positive DTE
            df = df[df["days_to_expiry"] > 0].copy()

            logger.info(f"✅ Options: {ticker} — {len(df)} strikes")
            return df

        except Exception as exc:
            logger.error(f"Options chain error {ticker}: {exc}")
            return pd.DataFrame()

    def inject_redis(self, redis_client) -> None:
        self._redis = redis_client


# ---------------------------------------------------------------------------
# SentimentDataFeed
# ---------------------------------------------------------------------------

class SentimentDataFeed:
    """
    News sentiment data backed by Polygon.io news API.

    Reddit removed in v6 — Polygon provides institutional-quality
    news sourcing (SEC, WSJ, Reuters, Bloomberg aggregated).

    Method signature get_news_and_reddit() kept for backward compatibility.
    """

    def __init__(self, redis_client=None):
        self._redis = redis_client

    def _feed(self) -> PolygonNewsFeed:
        return PolygonNewsFeed(
            api_key=_polygon_api_key(),
            redis_client=self._redis,
        )

    async def get_news_and_reddit(self, ticker: str) -> Dict:
        """
        Returns {'news': [...], 'reddit': []}.
        news: list of article dicts with keys:
              title, description, publisher, timestamp, url, sentiment_score
        reddit: always empty list (removed in v6)
        Caches with key polygon:v1:{ticker}:news (TTL 30 minutes).
        """
        try:
            articles = await self._feed().get_news(ticker=ticker, limit=50)
            if not articles:
                logger.warning(f"No news articles returned for {ticker}")
                articles = []

            normalised_news = []
            for art in articles:
                publisher = art.get("publisher", {})
                publisher_name = (
                    publisher.get("name", "")
                    if isinstance(publisher, dict)
                    else str(publisher)
                )
                normalised_news.append({
                    "title": art.get("title", ""),
                    "description": art.get("description", "") or art.get("summary", ""),
                    "publisher": publisher_name,
                    "timestamp": art.get("published_utc", ""),
                    "url": art.get("article_url", art.get("url", "")),
                    "sentiment_score": art.get("sentiment_score"),
                })

            logger.info(f"✅ News: {ticker} — {len(normalised_news)} articles")
            return {"news": normalised_news, "reddit": []}

        except Exception as exc:
            logger.error(f"Sentiment/news error {ticker}: {exc}")
            return {"news": [], "reddit": []}

    async def get_news(self, ticker: str, limit: int = 50) -> List[Dict]:
        """Direct access to news list (used by FinBERT pipeline)."""
        result = await self.get_news_and_reddit(ticker)
        return result.get("news", [])

    def inject_redis(self, redis_client) -> None:
        self._redis = redis_client
