"""
QuantEdge v6.0 — News Momentum (Analyst Revision Proxy)
========================================================
We don't have FactSet analyst revision data on Polygon Starter.
This module extracts a lower-fidelity proxy from Polygon's news API:

  1. News volume trend (accelerating = attention shift)
  2. Sentiment drift (7d avg vs 30d baseline)
  3. Revision-keyword extraction (upgrade/downgrade/target/beat/miss)

Output: 0-100 news momentum score per ticker.

Honest caveat: This is ~30-40% as good as real analyst revision data.
Academic anchor: Barber & Odean (2008) "All That Glitters",
Tetlock (2007) "Giving Content to Investor Sentiment"
"""

import os
import re
import asyncio
import aiohttp
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from collections import Counter
from loguru import logger


POLYGON_BASE = "https://api.polygon.io"


# Bullish revision keywords (weight +1)
BULLISH_KW = [
    "upgrade", "upgraded", "raises price target", "raise price target",
    "raised price target", "raised target", "beats estimates", "beat estimates",
    "exceeds expectations", "exceeded expectations", "boosts guidance",
    "raised guidance", "raises guidance", "strong buy", "outperform",
    "record revenue", "record earnings", "tops estimates", "topped",
    "bullish", "rally", "surge", "soar",
]

# Bearish revision keywords (weight -1)
BEARISH_KW = [
    "downgrade", "downgraded", "cuts price target", "cut price target",
    "lowered price target", "lowered target", "misses estimates",
    "missed estimates", "disappointing results", "weak guidance",
    "lowered guidance", "cut guidance", "underperform", "sell rating",
    "bearish", "plunge", "tumble", "crash", "slump",
    "warning", "investigation", "lawsuit", "probe", "recall",
    "misses expectations", "missed expectations",
]


@dataclass
class NewsMomentumSignal:
    ticker: str
    score: float = 50.0                     # 0-100 composite
    article_count_30d: int = 0
    article_count_7d: int = 0
    volume_trend: float = 1.0               # 7d rate / 30d baseline rate
    sentiment_30d_mean: float = 0.0         # -1 to +1
    sentiment_7d_mean: float = 0.0
    sentiment_drift: float = 0.0            # 7d mean - 30d mean
    bullish_keyword_hits: int = 0
    bearish_keyword_hits: int = 0
    net_keyword_score: int = 0              # bullish - bearish
    top_headlines: List[Dict] = field(default_factory=list)
    data_quality: str = "unknown"


# ══════════════════════════════════════════════════════════════
# POLYGON NEWS FETCHER
# ══════════════════════════════════════════════════════════════
async def fetch_news(
    ticker: str,
    api_key: str,
    session: aiohttp.ClientSession,
    days: int = 30,
    limit: int = 100,
) -> List[Dict]:
    """Fetch recent news articles for ticker from Polygon."""
    published_gte = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    url = f"{POLYGON_BASE}/v2/reference/news"
    params = {
        "ticker": ticker.upper(),
        "published_utc.gte": published_gte,
        "limit": limit,
        "order": "desc",
        "sort": "published_utc",
        "apiKey": api_key,
    }

    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                logger.debug(f"News fetch {resp.status} for {ticker}")
                return []
            data = await resp.json()
    except Exception as e:
        logger.debug(f"News fetch failed for {ticker}: {e}")
        return []

    return data.get("results", [])


# ══════════════════════════════════════════════════════════════
# ANALYSIS
# ══════════════════════════════════════════════════════════════
def count_keywords(text: str, keywords: List[str]) -> int:
    """Case-insensitive keyword count in text."""
    if not text:
        return 0
    lower = text.lower()
    hits = 0
    for kw in keywords:
        if kw in lower:
            hits += 1
    return hits


def extract_sentiment(article: Dict, ticker: str) -> Optional[float]:
    """
    Extract sentiment for this specific ticker from article insights.
    Polygon articles include 'insights' array with per-ticker sentiment.
    Returns float in [-1, +1] or None.
    """
    insights = article.get("insights") or []
    for ins in insights:
        if ins.get("ticker", "").upper() == ticker.upper():
            sentiment = ins.get("sentiment")
            if sentiment == "positive":
                return 1.0
            elif sentiment == "negative":
                return -1.0
            elif sentiment == "neutral":
                return 0.0
    return None


def score_news_momentum(signal: NewsMomentumSignal) -> float:
    """
    Composite 0-100 score.
    Weights:
      sentiment_drift:  30% (directional change matters most)
      net_keywords:     30% (revision language is discrete signal)
      volume_trend:     20% (attention shifts predict moves)
      sentiment_7d:     20% (current state)
    """
    if signal.article_count_30d < 3:
        return 50.0  # neutral — insufficient data

    # Sentiment drift: -1 to +1 range → 0-100
    drift_score = 50 + signal.sentiment_drift * 50
    drift_score = max(0, min(100, drift_score))

    # Net keywords: normalize by total keyword mentions
    total_kw = signal.bullish_keyword_hits + signal.bearish_keyword_hits
    if total_kw > 0:
        kw_ratio = signal.net_keyword_score / total_kw  # -1 to +1
        kw_score = 50 + kw_ratio * 50
    else:
        kw_score = 50
    kw_score = max(0, min(100, kw_score))

    # Volume trend: 7d rate / 30d baseline. 1.0 = steady, >1.5 = accelerating
    if signal.volume_trend >= 2.0:
        vol_score = 80
    elif signal.volume_trend >= 1.5:
        vol_score = 70
    elif signal.volume_trend >= 1.0:
        vol_score = 55
    elif signal.volume_trend >= 0.5:
        vol_score = 45
    else:
        vol_score = 30

    # Current sentiment
    sent_score = 50 + signal.sentiment_7d_mean * 50
    sent_score = max(0, min(100, sent_score))

    composite = (
        0.30 * drift_score +
        0.30 * kw_score +
        0.20 * vol_score +
        0.20 * sent_score
    )
    return round(composite, 1)


def analyze_articles(ticker: str, articles: List[Dict]) -> NewsMomentumSignal:
    """Compute news momentum signal from a list of Polygon articles."""
    signal = NewsMomentumSignal(ticker=ticker)

    if not articles:
        signal.data_quality = "no_news"
        return signal

    now = datetime.now(timezone.utc)
    cutoff_7d = now - timedelta(days=7)

    articles_30d: List[Dict] = []
    articles_7d: List[Dict] = []
    sentiments_30d: List[float] = []
    sentiments_7d: List[float] = []
    bullish_hits_total = 0
    bearish_hits_total = 0

    for a in articles:
        pub_str = a.get("published_utc")
        if not pub_str:
            continue
        try:
            pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        articles_30d.append(a)
        sent = extract_sentiment(a, ticker)
        if sent is not None:
            sentiments_30d.append(sent)

        title = a.get("title", "")
        desc = a.get("description", "")
        text = f"{title} {desc}"
        bull = count_keywords(text, BULLISH_KW)
        bear = count_keywords(text, BEARISH_KW)
        bullish_hits_total += bull
        bearish_hits_total += bear

        if pub_dt >= cutoff_7d:
            articles_7d.append(a)
            if sent is not None:
                sentiments_7d.append(sent)

    signal.article_count_30d = len(articles_30d)
    signal.article_count_7d = len(articles_7d)

    # Volume trend: 7d articles/day vs 30d articles/day
    rate_7d = signal.article_count_7d / 7.0
    rate_30d = signal.article_count_30d / 30.0
    signal.volume_trend = round(rate_7d / rate_30d, 3) if rate_30d > 0 else 1.0

    # Sentiment means
    signal.sentiment_30d_mean = round(sum(sentiments_30d) / len(sentiments_30d), 4) if sentiments_30d else 0.0
    signal.sentiment_7d_mean = round(sum(sentiments_7d) / len(sentiments_7d), 4) if sentiments_7d else 0.0
    signal.sentiment_drift = round(signal.sentiment_7d_mean - signal.sentiment_30d_mean, 4)

    # Keywords
    signal.bullish_keyword_hits = bullish_hits_total
    signal.bearish_keyword_hits = bearish_hits_total
    signal.net_keyword_score = bullish_hits_total - bearish_hits_total

    # Top headlines for inspection
    for a in articles_30d[:5]:
        signal.top_headlines.append({
            "title": a.get("title", "")[:140],
            "published": a.get("published_utc", ""),
            "sentiment": extract_sentiment(a, ticker),
            "url": a.get("article_url", ""),
        })

    signal.score = score_news_momentum(signal)
    signal.data_quality = (
        "excellent" if signal.article_count_30d >= 20 else
        "good" if signal.article_count_30d >= 10 else
        "partial" if signal.article_count_30d >= 3 else
        "insufficient"
    )
    return signal


# ══════════════════════════════════════════════════════════════
# ENGINE
# ══════════════════════════════════════════════════════════════
class NewsMomentumEngine:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")

    async def analyze(
        self,
        ticker: str,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> NewsMomentumSignal:
        if not self.api_key:
            return NewsMomentumSignal(ticker=ticker, data_quality="no_api_key")

        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            articles = await fetch_news(ticker, self.api_key, session, days=30, limit=100)
            return analyze_articles(ticker, articles)
        finally:
            if close_session:
                await session.close()


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════
async def _test(ticker: str = "NVDA"):
    import sys
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set"); sys.exit(1)

    engine = NewsMomentumEngine(api_key=api_key)
    print(f"Analyzing news momentum for {ticker}...")
    sig = await engine.analyze(ticker)

    print(f"\n{'='*70}")
    print(f"  NEWS MOMENTUM: {sig.ticker}")
    print(f"{'='*70}")
    print(f"  Score:               {sig.score} / 100")
    print(f"  Data quality:        {sig.data_quality}")
    print(f"\n  Article volume:")
    print(f"    Last 30 days:      {sig.article_count_30d}")
    print(f"    Last 7 days:       {sig.article_count_7d}")
    print(f"    Volume trend:      {sig.volume_trend}x (7d rate / 30d rate)")
    print(f"\n  Sentiment:")
    print(f"    30d mean:          {sig.sentiment_30d_mean:+.3f}")
    print(f"    7d mean:           {sig.sentiment_7d_mean:+.3f}")
    print(f"    Drift:             {sig.sentiment_drift:+.3f}")
    print(f"\n  Revision keywords:")
    print(f"    Bullish hits:      {sig.bullish_keyword_hits}")
    print(f"    Bearish hits:      {sig.bearish_keyword_hits}")
    print(f"    Net:               {sig.net_keyword_score:+d}")
    if sig.top_headlines:
        print(f"\n  Recent headlines:")
        for h in sig.top_headlines:
            s = h.get("sentiment")
            emoji = "+" if s == 1.0 else "-" if s == -1.0 else "."
            print(f"    [{emoji}] {h['title']}")


async def _test_batch():
    api_key = os.getenv("POLYGON_API_KEY", "")
    tickers = ["NVDA", "AAPL", "MSFT", "META", "TSLA", "INTC", "F", "AMZN"]

    engine = NewsMomentumEngine(api_key=api_key)
    async with aiohttp.ClientSession() as session:
        results = await asyncio.gather(*[engine.analyze(t, session) for t in tickers])

    print(f"\n{'='*90}")
    print(f"  NEWS MOMENTUM BATCH")
    print(f"{'='*90}")
    print(f"  {'Ticker':<8}{'Score':>7} {'30d':>5}{'7d':>5} {'VTrend':>7} "
          f"{'SentDr':>8} {'Bull':>5}{'Bear':>5}")
    print(f"  {'-'*8}{'-'*7} {'-'*5}{'-'*5} {'-'*7} {'-'*8} {'-'*5}{'-'*5}")
    for s in results:
        print(f"  {s.ticker:<8}{s.score:>7.1f} {s.article_count_30d:>5}"
              f"{s.article_count_7d:>5} {s.volume_trend:>7.2f} "
              f"{s.sentiment_drift:>+8.3f} {s.bullish_keyword_hits:>5}"
              f"{s.bearish_keyword_hits:>5}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        asyncio.run(_test_batch())
    else:
        t = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
        asyncio.run(_test(t))
