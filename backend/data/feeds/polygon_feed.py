"""
QuantEdge v6.0 — Polygon.io Data Feed
=======================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.

Replaces yFinance entirely. Uses Polygon.io Starter Plus ($29/mo).
All methods cache in Redis and retry with exponential backoff on rate limits.

Classes:
    PolygonMarketFeed      — OHLCV price history (10 years)
    PolygonFundamentalFeed — Financial ratios and fundamentals
    PolygonOptionsFeed     — Full options chain with Greeks
    PolygonNewsFeed        — News articles for FinBERT sentiment
    PolygonWebSocketManager — Real-time price streaming via Polygon WebSocket

Usage:
    from data.feeds.polygon_feed import (
        PolygonMarketFeed, PolygonFundamentalFeed,
        PolygonOptionsFeed, PolygonNewsFeed,
        PolygonWebSocketManager,
    )
    feed = PolygonMarketFeed(api_key=settings.POLYGON_API_KEY, redis_client=redis)
    df = await feed.get_price_history("AAPL")
"""

import asyncio
import json
import time
import math
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any
import aiohttp
import pandas as pd
import numpy as np
from loguru import logger

# ── Constants ─────────────────────────────────────────────────
POLYGON_BASE = "https://api.polygon.io"
POLYGON_WS   = "wss://socket.polygon.io/stocks"

CACHE_TTL_OHLCV        = 3600       # 1 hour
CACHE_TTL_FUNDAMENTALS = 86400      # 24 hours
CACHE_TTL_OPTIONS      = 900        # 15 minutes
CACHE_TTL_NEWS         = 1800       # 30 minutes

MAX_RETRIES  = 5
BACKOFF_BASE = 2.0   # seconds; doubles per retry


# ── Shared HTTP session factory ───────────────────────────────

def _make_session(api_key: str) -> aiohttp.ClientSession:
    return aiohttp.ClientSession(
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        },
        timeout=aiohttp.ClientTimeout(total=30),
    )


# ── Retry helper ─────────────────────────────────────────────

async def _get_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    params: Optional[Dict] = None,
    max_retries: int = MAX_RETRIES,
) -> Optional[Dict]:
    """
    GET request with exponential backoff on 429 (rate limit) and 5xx errors.
    Returns parsed JSON dict, or None on permanent failure.
    """
    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 200:
                    return await resp.json()
                if resp.status == 429:
                    wait = BACKOFF_BASE ** attempt
                    logger.warning(f"Polygon rate limit hit — waiting {wait:.1f}s (attempt {attempt+1})")
                    await asyncio.sleep(wait)
                    continue
                if resp.status == 403:
                    logger.error(f"Polygon 403 Forbidden: check API key and plan. URL: {url}")
                    return None
                if resp.status == 404:
                    logger.warning(f"Polygon 404 Not Found: {url}")
                    return None
                if resp.status >= 500:
                    wait = BACKOFF_BASE ** attempt
                    logger.warning(f"Polygon {resp.status} server error — retrying in {wait:.1f}s")
                    await asyncio.sleep(wait)
                    continue
                logger.error(f"Polygon unexpected status {resp.status}: {url}")
                return None
        except asyncio.TimeoutError:
            wait = BACKOFF_BASE ** attempt
            logger.warning(f"Polygon timeout — retrying in {wait:.1f}s")
            await asyncio.sleep(wait)
        except Exception as e:
            logger.error(f"Polygon request error: {e}")
            return None
    return None


# ── PolygonMarketFeed ─────────────────────────────────────────

class PolygonMarketFeed:
    """
    OHLCV daily price history via Polygon /v2/aggs/ticker endpoint.

    Caches result in Redis as gzipped JSON for 1 hour.
    Returns DataFrame columns: open, high, low, close, volume
    Index: DatetimeIndex (UTC)
    """

    def __init__(self, api_key: str, redis_client=None):
        self.api_key = api_key
        self.redis = redis_client

    async def get_price_history(
        self,
        ticker: str,
        years: int = 10,
    ) -> pd.DataFrame:
        """
        Fetch daily OHLCV for `ticker` going back `years` years.
        Caches with key polygon:v1:{ticker}:ohlcv — TTL 1 hour.
        """
        cache_key = f"polygon:v1:{ticker.upper()}:ohlcv"

        # Try cache first
        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return pd.read_json(cached, orient="split")
            except Exception:
                pass

        # Fetch from Polygon
        df = await self._fetch_aggregates(ticker, years)

        # Cache the result
        if not df.empty and self.redis:
            try:
                await self.redis.setex(cache_key, CACHE_TTL_OHLCV, df.to_json(orient="split"))
            except Exception:
                pass

        return df

    async def _fetch_aggregates(self, ticker: str, years: int) -> pd.DataFrame:
        """
        Polygon /v2/aggs/ticker/{ticker}/range/1/day/{from}/{to}
        Paginates automatically if result set is large.
        """
        end_date   = date.today()
        start_date = end_date - timedelta(days=years * 365)
        from_str   = start_date.strftime("%Y-%m-%d")
        to_str     = end_date.strftime("%Y-%m-%d")

        url    = f"{POLYGON_BASE}/v2/aggs/ticker/{ticker.upper()}/range/1/day/{from_str}/{to_str}"
        params = {"adjusted": "true", "sort": "asc", "limit": 50000}

        all_results = []

        async with _make_session(self.api_key) as session:
            while url:
                data = await _get_with_retry(session, url, params)
                if data is None:
                    break
                results = data.get("results", [])
                if not results:
                    break
                all_results.extend(results)
                # Pagination
                next_url = data.get("next_url")
                if next_url:
                    url    = next_url
                    params = {}   # next_url already has all params
                else:
                    break

        if not all_results:
            logger.warning(f"Polygon: no OHLCV results for {ticker}")
            return pd.DataFrame()

        df = pd.DataFrame(all_results)
        # Polygon columns: t=timestamp(ms), o=open, h=high, l=low, c=close, v=volume, vw=vwap
        df = df.rename(columns={
            "t": "timestamp", "o": "open", "h": "high",
            "l": "low", "c": "close", "v": "volume", "vw": "vwap",
        })
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()

        keep = [c for c in ["open", "high", "low", "close", "volume", "vwap"] if c in df.columns]
        df = df[keep].dropna(subset=["close"])
        df = df[df["close"] > 0]

        logger.info(f"Polygon OHLCV {ticker}: {len(df)} rows — {df.index[0].date()} to {df.index[-1].date()}")
        return df


# ── PolygonFundamentalFeed ────────────────────────────────────

class PolygonFundamentalFeed:
    """
    Fundamental financial data via Polygon /vX/reference/financials endpoint.
    Also fetches ticker details for P/E, market cap from /v3/reference/tickers.
    Caches for 24 hours.
    """

    def __init__(self, api_key: str, redis_client=None):
        self.api_key = api_key
        self.redis = redis_client

    async def get_fundamentals(self, ticker: str) -> Dict:
        """
        Returns dict with keys:
            pe_ratio, pb_ratio, eps_ttm, revenue_growth, debt_to_equity,
            current_ratio, roe, roa, gross_margin, market_cap,
            name, sector, industry, exchange, beta
        """
        cache_key = f"polygon:v1:{ticker.upper()}:fundamentals"

        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        result = {}

        async with _make_session(self.api_key) as session:
            # Ticker details: name, market_cap, description, sector
            detail_data = await _get_with_retry(
                session,
                f"{POLYGON_BASE}/v3/reference/tickers/{ticker.upper()}",
            )
            if detail_data and "results" in detail_data:
                res = detail_data["results"]
                result["name"]         = res.get("name", ticker)
                result["sector"]       = res.get("sic_description", "Unknown")
                result["industry"]     = res.get("sic_description", "Unknown")
                result["exchange"]     = res.get("primary_exchange", "")
                result["market_cap"]   = res.get("market_cap")
                result["description"]  = res.get("description", "")

            # Snapshot — current price + prev close for P/E approximation
            snapshot = await _get_with_retry(
                session,
                f"{POLYGON_BASE}/v2/snapshot/locale/us/markets/stocks/tickers/{ticker.upper()}",
            )
            if snapshot and "ticker" in snapshot:
                snap_ticker = snapshot["ticker"]
                day = snap_ticker.get("day", {})
                result["price"]      = day.get("c")
                result["prev_close"] = snap_ticker.get("prevDay", {}).get("c")
                result["volume"]     = day.get("v")
                result["vwap"]       = day.get("vw")

                # P/E ratio from Polygon ticker details v3
                # Polygon provides pe_ratio in the ticker details endpoint
                details_v3 = await _get_with_retry(
                    session,
                    f"{POLYGON_BASE}/v3/reference/tickers/{ticker.upper()}",
                )
                if details_v3 and "results" in details_v3:
                    res3 = details_v3["results"]
                    # Polygon Starter Plus returns these in branding/metadata
                    result["weight_index"] = res3.get("weight_index")
                    result["list_date"]    = res3.get("list_date")
                    result["share_class_shares_outstanding"] = res3.get("share_class_shares_outstanding")
                    result["weighted_shares_outstanding"]    = res3.get("weighted_shares_outstanding")

                    # Compute market cap from share count + price if not already set
                    price = result.get("price", 0) or 0
                    shares = res3.get("weighted_shares_outstanding") or res3.get("share_class_shares_outstanding")
                    if shares and price and not result.get("market_cap"):
                        result["market_cap"] = float(shares) * float(price)

            # Financials endpoint for income statement / balance sheet
            financials = await _get_with_retry(
                session,
                f"{POLYGON_BASE}/vX/reference/financials",
                params={
                    "ticker": ticker.upper(),
                    "timeframe": "annual",
                    "order": "desc",
                    "limit": 4,
                    "sort": "filing_date",
                },
            )
            if financials and financials.get("results"):
                results = financials["results"]
                latest = results[0] if results else {}
                fin = latest.get("financials", {})

                # Income statement
                income = fin.get("income_statement", {})
                result["revenue"]         = _safe(income, "revenues")
                result["gross_profit"]    = _safe(income, "gross_profit")
                result["operating_income"]= _safe(income, "operating_income_loss")
                result["net_income"]      = _safe(income, "net_income_loss")
                result["eps_ttm"]         = _safe(income, "basic_earnings_per_share") or _safe(income, "diluted_earnings_per_share")
                result["net_income"]      = _safe(income, "net_income_loss") or _safe(income, "net_income_loss_attributable_to_parent")
                result["total_equity"]    = _safe(balance, "equity") or _safe(balance, "equity_attributable_to_parent")
                result["total_debt"]      = (_safe(balance, "long_term_debt") or 0) + (_safe(balance, "current_portion_of_long_term_debt") or 0)
                result["total_cash"]      = _safe(balance, "cash_and_cash_equivalents_including_restricted_cash") or _safe(balance, "cash_and_equivalents")
                result["ebitda"]          = (_safe(income, "operating_income_loss") or 0) + (_safe(income, "depreciation_and_amortization") or 0)

                rev_curr = _safe(income, "revenues")
                # Revenue growth: compare to 1-year-ago filing
                if len(results) >= 2:
                    income_prev = results[1].get("financials", {}).get("income_statement", {})
                    rev_prev = _safe(income_prev, "revenues")
                    if rev_prev and rev_curr and rev_prev != 0:
                        result["revenue_growth"] = (rev_curr - rev_prev) / abs(rev_prev)

                # Balance sheet
                balance = fin.get("balance_sheet", {})
                total_assets  = _safe(balance, "assets")
                total_equity  = _safe(balance, "equity")
                total_liab    = _safe(balance, "liabilities")
                curr_assets   = _safe(balance, "current_assets")
                curr_liab     = _safe(balance, "current_liabilities")

                if total_equity and total_equity != 0:
                    if total_liab is not None:
                        result["debt_to_equity"] = total_liab / abs(total_equity)
                    if result.get("net_income") is not None:
                        result["roe"] = result["net_income"] / abs(total_equity)
                if total_assets and total_assets != 0 and result.get("net_income") is not None:
                    result["roa"] = result["net_income"] / total_assets
                if curr_liab and curr_liab != 0 and curr_assets is not None:
                    result["current_ratio"] = curr_assets / curr_liab

                # Gross margin
                if rev_curr and rev_curr != 0 and result.get("gross_profit") is not None:
                    result["gross_margin"] = result["gross_profit"] / rev_curr

        # ── Derived ratios ──────────────────────────────────
        # Compute P/E, P/B, P/S from available data + price
        price = result.get("price", 0) or 0
        if price > 0:
            # P/E = price / EPS_TTM
            eps = result.get("eps_ttm")
            if eps and eps != 0:
                result["pe_ratio"] = round(price / eps, 2)

            # P/B = price / (equity / shares)
            equity = result.get("total_equity") or result.get("equity")
            shares = result.get("weighted_shares_outstanding") or result.get("share_class_shares_outstanding")
            if equity and shares and float(shares) > 0:
                book_per_share = float(equity) / float(shares)
                if book_per_share > 0:
                    result["price_to_book"] = round(price / book_per_share, 2)
                    result["pb_ratio"]      = result["price_to_book"]

            # P/S = market_cap / revenue_ttm
            market_cap = result.get("market_cap", 0) or 0
            revenue    = result.get("revenue") or result.get("revenue_ttm", 0) or 0
            if market_cap > 0 and revenue > 0:
                result["price_to_sales"] = round(market_cap / revenue, 2)

            # EV/EBITDA approximation
            ebitda = result.get("ebitda", 0) or 0
            total_debt = result.get("total_debt", 0) or 0
            total_cash = result.get("total_cash", 0) or 0
            if ebitda > 0 and market_cap > 0:
                ev = market_cap + total_debt - total_cash
                result["ev_ebitda"] = round(ev / ebitda, 2)

            # Operating margin
            operating_income = result.get("operating_income", 0) or 0
            if revenue > 0 and operating_income:
                result["operating_margin"] = round(operating_income / revenue, 6)

            # Net margin
            net_income = result.get("net_income", 0) or 0
            if revenue > 0 and net_income:
                result["net_margin"] = round(net_income / revenue, 6)

            # Revenue TTM alias
            if revenue and not result.get("revenue_ttm"):
                result["revenue_ttm"] = revenue

        # Cache result
        if result and self.redis:
            try:
                await self.redis.setex(cache_key, CACHE_TTL_FUNDAMENTALS, json.dumps(result, default=str))
            except Exception:
                pass

        return result


def _safe(d: Dict, key: str) -> Optional[float]:
    """Safely extract a numeric value from a Polygon financials dict."""
    val = d.get(key, {})
    if isinstance(val, dict):
        v = val.get("value")
    else:
        v = val
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


# ── PolygonOptionsFeed ────────────────────────────────────────

class PolygonOptionsFeed:
    """
    Options chain data via Polygon /v3/reference/options/contracts endpoint.
    Returns DataFrame with strikes, expiries, IVs, Greeks, volume, OI.
    Caches for 15 minutes.
    """

    def __init__(self, api_key: str, redis_client=None):
        self.api_key = api_key
        self.redis = redis_client

    async def get_chain(self, ticker: str) -> pd.DataFrame:
        """
        Returns options chain DataFrame with columns:
            strike, expiry, option_type, call_iv, put_iv,
            call_delta, put_delta, call_gamma, put_gamma,
            call_volume, put_volume, call_oi, put_oi
        """
        cache_key = f"polygon:v1:{ticker.upper()}:options"

        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return pd.read_json(cached, orient="split")
            except Exception:
                pass

        df = await self._fetch_chain(ticker)

        if not df.empty and self.redis:
            try:
                await self.redis.setex(cache_key, CACHE_TTL_OPTIONS, df.to_json(orient="split"))
            except Exception:
                pass

        return df

    async def _fetch_chain(self, ticker: str) -> pd.DataFrame:
        """
        Paginate through /v3/reference/options/contracts for current chain.
        Then fetch snapshot for Greeks/IV.
        """
        rows = []
        today = date.today()
        exp_cutoff = (today + timedelta(days=90)).strftime("%Y-%m-%d")

        async with _make_session(self.api_key) as session:
            url = f"{POLYGON_BASE}/v3/reference/options/contracts"
            params = {
                "underlying_ticker": ticker.upper(),
                "expiration_date.gte": today.strftime("%Y-%m-%d"),
                "expiration_date.lte": exp_cutoff,
                "limit": 250,
                "sort": "expiration_date",
            }

            page_count = 0
            while url and page_count < 10:
                data = await _get_with_retry(session, url, params if page_count == 0 else None)
                if data is None or not data.get("results"):
                    break
                for contract in data["results"]:
                    rows.append({
                        "ticker":       contract.get("ticker", ""),
                        "strike":       contract.get("strike_price"),
                        "expiry":       contract.get("expiration_date"),
                        "option_type":  contract.get("contract_type", ""),  # 'call' or 'put'
                        "shares_per":   contract.get("shares_per_contract", 100),
                    })
                url = data.get("next_url")
                params = None
                page_count += 1

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)

        # Fetch snapshot for IV and Greeks on all contracts
        tickers_str = ",".join(df["ticker"].dropna().unique()[:100])
        if tickers_str:
            async with _make_session(self.api_key) as session:
                snap = await _get_with_retry(
                    session,
                    f"{POLYGON_BASE}/v3/snapshot/options/{ticker.upper()}",
                    params={"limit": 250},
                )
                if snap and snap.get("results"):
                    snap_map = {}
                    for r in snap["results"]:
                        details = r.get("details", {})
                        greeks  = r.get("greeks", {})
                        snap_map[details.get("ticker", "")] = {
                            "iv":    r.get("implied_volatility"),
                            "delta": greeks.get("delta"),
                            "gamma": greeks.get("gamma"),
                            "theta": greeks.get("theta"),
                            "vega":  greeks.get("vega"),
                            "volume": r.get("day", {}).get("volume"),
                            "open_interest": r.get("open_interest"),
                        }
                    df["iv"]            = df["ticker"].map(lambda t: snap_map.get(t, {}).get("iv"))
                    df["delta"]         = df["ticker"].map(lambda t: snap_map.get(t, {}).get("delta"))
                    df["gamma"]         = df["ticker"].map(lambda t: snap_map.get(t, {}).get("gamma"))
                    df["theta"]         = df["ticker"].map(lambda t: snap_map.get(t, {}).get("theta"))
                    df["vega"]          = df["ticker"].map(lambda t: snap_map.get(t, {}).get("vega"))
                    df["volume"]        = df["ticker"].map(lambda t: snap_map.get(t, {}).get("volume"))
                    df["open_interest"] = df["ticker"].map(lambda t: snap_map.get(t, {}).get("open_interest"))

        logger.info(f"Polygon options {ticker}: {len(df)} contracts")
        return df


# ── PolygonNewsFeed ───────────────────────────────────────────

class PolygonNewsFeed:
    """
    News articles via Polygon /v2/reference/news endpoint.
    Returns list of dicts for FinBERT sentiment scoring.
    Caches for 30 minutes.
    """

    def __init__(self, api_key: str, redis_client=None):
        self.api_key = api_key
        self.redis = redis_client

    async def get_news(self, ticker: str, limit: int = 50) -> List[Dict]:
        """
        Returns list of dicts: title, description, published_utc, article_url,
                               publisher, author, sentiment_score (Polygon's NLP, optional)
        """
        cache_key = f"polygon:v1:{ticker.upper()}:news"

        if self.redis:
            try:
                cached = await self.redis.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass

        articles = []
        async with _make_session(self.api_key) as session:
            data = await _get_with_retry(
                session,
                f"{POLYGON_BASE}/v2/reference/news",
                params={
                    "ticker": ticker.upper(),
                    "order": "desc",
                    "limit": min(limit, 50),
                    "sort": "published_utc",
                },
            )
            if data and data.get("results"):
                for article in data["results"]:
                    # Polygon Starter Plus may include insights (NLP pre-scored sentiment)
                    insights = article.get("insights", [])
                    poly_sentiment = None
                    for ins in insights:
                        if ins.get("ticker") == ticker.upper():
                            sentiment_str = ins.get("sentiment", "neutral")
                            poly_sentiment = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}.get(
                                sentiment_str.lower(), 0.0
                            )
                    articles.append({
                        "title":           article.get("title", ""),
                        "description":     article.get("description", ""),
                        "published_utc":   article.get("published_utc", ""),
                        "article_url":     article.get("article_url", ""),
                        "publisher":       article.get("publisher", {}).get("name", ""),
                        "author":          article.get("author", ""),
                        "polygon_sentiment": poly_sentiment,   # pre-scored by Polygon NLP (may be None)
                    })

        if articles and self.redis:
            try:
                await self.redis.setex(cache_key, CACHE_TTL_NEWS, json.dumps(articles))
            except Exception:
                pass

        logger.info(f"Polygon news {ticker}: {len(articles)} articles")
        return articles


# ── PolygonWebSocketManager ───────────────────────────────────

class PolygonWebSocketManager:
    """
    Polygon WebSocket real-time price feed.
    Opens one upstream WebSocket connection per ticker.
    Publishes price ticks to Redis pub/sub channel: ws:price:{ticker}
    so unlimited browser clients can subscribe via the FastAPI WebSocket endpoint.

    Reconnects automatically on disconnect with exponential backoff.

    Usage:
        manager = PolygonWebSocketManager(api_key=settings.POLYGON_API_KEY, redis_client=redis)
        asyncio.create_task(manager.subscribe("AAPL"))
        asyncio.create_task(manager.subscribe("NVDA"))
    """

    def __init__(self, api_key: str, redis_client):
        self.api_key = api_key
        self.redis = redis_client
        self._active: Dict[str, bool] = {}

    async def subscribe(self, ticker: str) -> None:
        """
        Subscribe to real-time trades for `ticker`.
        Runs indefinitely — reconnects on disconnect.
        Each trade tick is published to Redis channel ws:price:{ticker}.
        """
        ticker = ticker.upper()
        self._active[ticker] = True
        channel = f"ws:price:{ticker}"
        retry_delay = 1.0

        while self._active.get(ticker, False):
            try:
                import websockets
                async with websockets.connect(POLYGON_WS) as ws:
                    # Wait for connected message
                    msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    data = json.loads(msg)
                    if not any(m.get("status") == "connected" for m in data):
                        logger.warning(f"Polygon WS: unexpected first message: {data}")

                    # Authenticate
                    await ws.send(json.dumps({"action": "auth", "params": self.api_key}))
                    auth_msg = await asyncio.wait_for(ws.recv(), timeout=10)
                    auth_data = json.loads(auth_msg)
                    if not any(m.get("status") == "auth_success" for m in auth_data):
                        logger.error(f"Polygon WS auth failed: {auth_data}")
                        await asyncio.sleep(retry_delay)
                        retry_delay = min(retry_delay * 2, 60)
                        continue

                    # Subscribe to trades
                    await ws.send(json.dumps({"action": "subscribe", "params": f"T.{ticker}"}))
                    logger.info(f"Polygon WS: subscribed to T.{ticker}")
                    retry_delay = 1.0  # Reset on success

                    async for raw_msg in ws:
                        if not self._active.get(ticker, False):
                            break
                        try:
                            events = json.loads(raw_msg)
                            for event in events:
                                if event.get("ev") == "T":  # Trade event
                                    payload = json.dumps({
                                        "ticker":    ticker,
                                        "price":     event.get("p", 0),
                                        "size":      event.get("s", 0),
                                        "timestamp": event.get("t", 0),
                                        "source":    "polygon_ws",
                                    })
                                    await self.redis.publish(channel, payload)
                        except Exception as e:
                            logger.warning(f"Polygon WS message parse error: {e}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self._active.get(ticker, False):
                    logger.warning(f"Polygon WS {ticker} disconnected: {e} — reconnecting in {retry_delay:.1f}s")
                    await asyncio.sleep(retry_delay)
                    retry_delay = min(retry_delay * 2, 60)

        logger.info(f"Polygon WS: stopped subscription for {ticker}")

    def unsubscribe(self, ticker: str) -> None:
        """Signal the subscription loop to stop."""
        self._active[ticker.upper()] = False
