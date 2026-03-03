"""
QuantEdge v6.0 — Celery Background Tasks & Scheduler
=====================================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.
Proprietary & Confidential.

Handles scheduled background tasks:
  - Market data refresh every 15 minutes
  - Sentiment refresh every 30 minutes
  - Daily model retraining at 6pm ET
  - Weekly full universe retrain on Sundays

Setup:
    # Start Celery worker
    celery -A tasks worker --loglevel=info

    # Start Celery Beat scheduler
    celery -A tasks beat --loglevel=info

    # Or combined (dev only):
    celery -A tasks worker --beat --loglevel=info
"""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List
from loguru import logger

from celery import Celery
from celery.schedules import crontab

# ── Celery App ────────────────────────────────────────────────
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "quantedge",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["tasks"]
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="America/New_York",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# ── Beat Schedule ─────────────────────────────────────────────
app.conf.beat_schedule = {
    # Refresh OHLCV data every 15 minutes (market hours only)
    "refresh-market-data": {
        "task": "tasks.refresh_market_data",
        "schedule": crontab(minute="*/15", hour="9-16", day_of_week="1-5"),
        "args": (["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"], 5),
    },
    # Refresh sentiment every 30 minutes
    "refresh-sentiment": {
        "task": "tasks.refresh_sentiment_cache",
        "schedule": crontab(minute="*/30"),
        "args": (["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"],),
    },
    # Daily model refresh at 6pm ET (after market close)
    "daily-model-refresh": {
        "task": "tasks.daily_model_refresh",
        "schedule": crontab(hour=18, minute=0, day_of_week="1-5"),
        "args": (["AAPL", "MSFT", "NVDA", "TSLA"],),
    },
    # Weekly full retrain Sunday at 2am
    "weekly-retrain": {
        "task": "tasks.weekly_full_retrain",
        "schedule": crontab(hour=2, minute=0, day_of_week="sun"),
        "args": (),
    },
    # Health check every 5 minutes
    "health-ping": {
        "task": "tasks.health_ping",
        "schedule": crontab(minute="*/5"),
    },
}


# ── Task: Refresh Market Data ─────────────────────────────────
@app.task(bind=True, max_retries=3, default_retry_delay=60)
def refresh_market_data(self, tickers: List[str], days: int = 5):
    """Fetch recent OHLCV and store to DB."""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ml.data_pipeline.fetch_and_store import (
            fetch_ohlcv, store_ohlcv, get_engine, create_tables
        )
        engine = get_engine()
        create_tables(engine)

        success = 0
        for ticker in tickers:
            df = fetch_ohlcv(ticker, days)
            if df is not None:
                rows = store_ohlcv(df, engine)
                if rows > 0:
                    success += 1

        engine.dispose()
        logger.info(f"✅ Market data refresh: {success}/{len(tickers)} tickers updated")
        return {"status": "ok", "updated": success, "tickers": tickers}
    except Exception as e:
        logger.error(f"Market data refresh failed: {e}")
        raise self.retry(exc=e)


# ── Task: Refresh Sentiment Cache ─────────────────────────────
@app.task(bind=True, max_retries=2, default_retry_delay=120)
def refresh_sentiment_cache(self, tickers: List[str]):
    """Refresh sentiment analysis cache in Redis."""
    try:
        import redis
        import json
        from ml.price_oracle.sentiment_nlp import SentimentEngine

        r = redis.from_url(REDIS_URL)
        engine = SentimentEngine()

        updated = 0
        for ticker in tickers:
            try:
                result = engine.analyze(ticker)
                cache_key = f"sentiment:{ticker}"
                r.setex(cache_key, 1800, json.dumps(result))  # 30-min TTL
                updated += 1
                logger.debug(f"Cached sentiment for {ticker}: {result.get('signal')}")
            except Exception as e:
                logger.warning(f"Sentiment cache failed for {ticker}: {e}")

        logger.info(f"✅ Sentiment cache refreshed: {updated}/{len(tickers)}")
        return {"status": "ok", "updated": updated}
    except Exception as e:
        logger.error(f"Sentiment cache task failed: {e}")
        raise self.retry(exc=e)


# ── Task: Daily Model Refresh ─────────────────────────────────
@app.task(bind=True, max_retries=1)
def daily_model_refresh(self, tickers: List[str]):
    """Retrain models on specified tickers daily."""
    try:
        sys.path.insert(0, str(Path(__file__).parent / "ml" / "training"))
        from ml.training.run_training import train_single_ticker

        results = []
        for ticker in tickers:
            logger.info(f"Daily retrain: {ticker}")
            r = train_single_ticker(ticker, days=500, skip_data=False)
            results.append(r)

        logger.info(f"✅ Daily retrain complete: {len(tickers)} tickers")
        return {"status": "ok", "results": results}
    except Exception as e:
        logger.error(f"Daily retrain failed: {e}")
        return {"status": "error", "message": str(e)}


# ── Task: Weekly Full Universe Retrain ────────────────────────
@app.task(bind=True, max_retries=1)
def weekly_full_retrain(self):
    """Full universe retrain on Sundays."""
    from ml.data_pipeline.fetch_and_store import SP500_TOP50
    try:
        sys.path.insert(0, str(Path(__file__).parent / "ml" / "training"))
        from ml.training.run_training import train_universe

        results = train_universe(SP500_TOP50, days=1000, skip_data=False)
        success = sum(1 for r in results
                      if r.get('xgboost', {}).get('status') == 'ok')

        logger.info(f"✅ Weekly retrain: {success}/{len(SP500_TOP50)} models updated")
        return {"status": "ok", "success": success, "total": len(SP500_TOP50)}
    except Exception as e:
        logger.error(f"Weekly retrain failed: {e}")
        return {"status": "error", "message": str(e)}


# ── Task: Health Ping ─────────────────────────────────────────
@app.task
def health_ping():
    """Ensure Celery is alive."""
    return {"status": "alive", "time": datetime.now().isoformat()}


if __name__ == "__main__":
    # Manual trigger for testing
    print("Triggering market data refresh...")
    result = refresh_market_data.delay(["AAPL", "MSFT"], days=5)
    print(f"Task ID: {result.id}")
