"""
QuantEdge v6.0 — FastAPI Application
=====================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.

Lifespan initializes in this exact order (per spec Section 5.1 main_v6.py):
  1. Redis pool          — app.state.redis
  2. DB pool             — app.state.db
  3. FinBERT pipeline    — app.state.finbert
  4. QuantEdgeAnalyzerV6 — app.state.analyzer
  5. ML model check      — train any missing model files before serving
  6. SignalTracker        — app.state.signal_tracker
  7. APScheduler         — OutcomeFillerJob daily at 18:00 ET
  8. Cache warmer background task
Shutdown:
  - Cancel warmer, stop scheduler, close Redis pool, close DB pool
"""

from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional
import os
import time
import asyncio
import json
import redis.asyncio as aioredis
import asyncpg
from loguru import logger

from core.config import settings
from auth.cognito_auth import get_current_user, CognitoUser
from routers import analysis, auth_router, watchlist, portfolio
from routers.analysis_v6 import router as v6_router, QuantEdgeAnalyzerV6
from routers.performance_router import router as performance_router
from routers.quality_router import router as quality_router
from routers.screener_router import router as screener_router
from ml.price_oracle.router import router as oracle_router
from services.signal_tracker import SignalTracker, OutcomeFillerJob


# ── Model file paths ───────────────────────────────────────────
_MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./models"))
XGB_MODEL_PATH  = _MODEL_DIR / "xgb_model.joblib"
LGB_MODEL_PATH  = _MODEL_DIR / "lgb_model.joblib"
LSTM_MODEL_PATH = _MODEL_DIR / "lstm_weights.pt"


async def _ensure_models_trained(analyzer: QuantEdgeAnalyzerV6) -> None:
    """
    Check if all 3 model files exist in MODEL_DIR.
    If any are missing, train on AAPL before serving traffic.
    This prevents the first real request from hanging during ECS startup
    and causing health check failure / restart loop.
    """
    missing = []
    if not XGB_MODEL_PATH.exists():
        missing.append("xgb_model.joblib")
    if not LGB_MODEL_PATH.exists():
        missing.append("lgb_model.joblib")
    if not LSTM_MODEL_PATH.exists():
        missing.append("lstm_weights.pt")

    if not missing:
        logger.info(f"✅ All ML model files present in {_MODEL_DIR}")
        return

    logger.warning(f"⚠️  Missing model files: {missing}. Training on AAPL now...")
    try:
        from ml.models.real_ml_engine import ModelTrainer
        from data.feeds.polygon_feed import PolygonMarketFeed, PolygonFundamentalFeed

        price_feed = PolygonMarketFeed(api_key=settings.POLYGON_API_KEY, redis_client=None)
        fund_feed  = PolygonFundamentalFeed(api_key=settings.POLYGON_API_KEY, redis_client=None)

        price_data   = await price_feed.get_price_history("AAPL", years=10)
        fundamentals = await fund_feed.get_fundamentals("AAPL")

        trainer = ModelTrainer()
        metrics = trainer.train_all(
            ticker="AAPL",
            price_data=price_data,
            fundamentals=fundamentals,
        )
        logger.info(f"✅ Model training complete: {metrics}")
    except Exception as e:
        logger.error(f"⚠️  Startup model training failed: {e} — models will train on first prediction request")


# ── Cache Warmer ───────────────────────────────────────────────
async def _cache_warmer(redis, analyzer: QuantEdgeAnalyzerV6) -> None:
    """
    Pre-warm analysis cache for top-8 tickers.
    Waits 60s after startup, then sweeps every 15 minutes.
    Only warms tickers with a cache miss.
    """
    TOP_TICKERS = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ", "MSFT", "AMZN", "META"]
    await asyncio.sleep(60)

    while True:
        for ticker in TOP_TICKERS:
            try:
                cache_key = f"analysis:v6:{ticker}"
                if not await redis.exists(cache_key):
                    logger.info(f"Cache warmer: warming {ticker}")
                    data = await analyzer.run_full_analysis(
                        ticker=ticker,
                        include_options=True,
                        include_sentiment=True,
                        mc_paths=10_000,
                    )
                    await redis.setex(cache_key, 300, json.dumps(data, default=str))
                    logger.info(f"Cache warmer: {ticker} warmed")
                    await asyncio.sleep(5)
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"Cache warmer error {ticker}: {e}")
        try:
            await asyncio.sleep(900)
        except asyncio.CancelledError:
            return


# ── Lifespan ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 QuantEdge v6.0 starting...")

    # 1. Redis connection pool
    app.state.redis = await aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
    )
    logger.info(f"✅ Redis pool: {settings.REDIS_URL[:40]}...")

    # 2. PostgreSQL — optional, graceful fallback if RDS is stopped
    app.state.db = None
    try:
        db_url = settings.DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
        app.state.db = await asyncpg.create_pool(
            db_url, min_size=1, max_size=5, command_timeout=10,
        )
        logger.info("✅ PostgreSQL pool connected")
        async with app.state.db.acquire() as conn:
            await conn.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    ticker VARCHAR(10) NOT NULL,
                    generated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    hmm_regime VARCHAR(30) NOT NULL,
                    hmm_confidence FLOAT NOT NULL,
                    garch_vol_forecast FLOAT, garch_regime VARCHAR(20), hmm_state_probs JSONB,
                    kalman_trend FLOAT, kalman_uncertainty FLOAT,
                    lstm_pred_5d FLOAT, lstm_pred_21d FLOAT, lstm_pred_63d FLOAT, lstm_uncertainty FLOAT,
                    xgb_signal FLOAT, xgb_confidence FLOAT, xgb_shap_values JSONB,
                    lgb_signal FLOAT, lgb_confidence FLOAT,
                    ensemble_signal FLOAT NOT NULL, ensemble_direction VARCHAR(10), weights_used JSONB NOT NULL,
                    cvar_95 FLOAT, vol_scale FLOAT, recommended_position FLOAT,
                    ret_5d FLOAT, ret_21d FLOAT, ret_63d FLOAT, barrier_hit VARCHAR(20), ic_contribution FLOAT
                )""")
            await conn.execute("CREATE TABLE IF NOT EXISTS performance_daily (date DATE PRIMARY KEY, ic_21d FLOAT, icir_21d FLOAT, hit_rate FLOAT, n_signals INTEGER, model_ics JSONB)")
            await conn.execute("CREATE TABLE IF NOT EXISTS regime_performance (regime VARCHAR(30) NOT NULL, period_start DATE NOT NULL, period_end DATE, mean_ic FLOAT, icir FLOAT, hit_rate FLOAT, n_signals INTEGER, PRIMARY KEY (regime, period_start))")
            await conn.execute("CREATE TABLE IF NOT EXISTS model_weights_history (id UUID PRIMARY KEY DEFAULT gen_random_uuid(), recorded_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), regime VARCHAR(30) NOT NULL, weights JSONB NOT NULL, ic_basis FLOAT, n_signals_used INTEGER)")
        logger.info("✅ Database tables verified/created")
    except Exception as e:
        logger.warning(f"PostgreSQL unavailable (RDS stopped) — running without DB: {e}")
        app.state.db = None

        # 3. FinBERT — loaded lazily on first request to avoid startup timeout
    app.state.finbert = None
    logger.info("✅ FinBERT will load on first analysis request (lazy)")

    # 4. QuantEdgeAnalyzerV6 — all ML models loaded into memory now
    app.state.analyzer = QuantEdgeAnalyzerV6()
# Inject the shared Redis pool into all data feeds
# so every Polygon call is cached — no cold hits per request
    app.state.analyzer.market_feed.inject_redis(app.state.redis)
    app.state.analyzer.fund_feed.inject_redis(app.state.redis)
    app.state.analyzer.options_feed.inject_redis(app.state.redis)
    app.state.analyzer.sentiment_feed.inject_redis(app.state.redis)
    logger.info("✅ Redis injected into all data feeds")
    logger.info("✅ QuantEdgeAnalyzerV6 initialized")

    # 5. ML model files — trained lazily on first analysis request
    logger.info("✅ ML models will train on first analysis request (lazy)")

    # 6. SignalTracker — uses shared DB pool
    app.state.signal_tracker = SignalTracker(db_pool=app.state.db) if app.state.db else None
    logger.info("✅ SignalTracker initialized")

    # 7. APScheduler — OutcomeFillerJob daily at 18:00 ET
    app.state.scheduler = None
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        import pytz

        if not app.state.signal_tracker:
            raise Exception("No DB — skipping scheduler")
        if not app.state.signal_tracker:
            raise Exception("No DB")
        outcome_job = OutcomeFillerJob(signal_tracker=app.state.signal_tracker)
        scheduler = AsyncIOScheduler(timezone=pytz.utc)
        scheduler.add_job(
            outcome_job.run,
            trigger=CronTrigger(hour=23, minute=0, timezone=pytz.timezone("US/Eastern")),
            id="outcome_filler",
            name="Daily outcome filler 18:00 ET",
            replace_existing=True,
        )
        scheduler.start()
        app.state.scheduler = scheduler
        logger.info("✅ APScheduler started — OutcomeFillerJob at 18:00 ET daily")
    except ImportError:
        logger.warning("⚠️  apscheduler not installed — add it to requirements.txt")
    except Exception as e:
        logger.error(f"APScheduler error: {e}")

    # 8. Cache warmer — pre-warms top tickers so first user request is instant
    async def _startup_warmer():
        await asyncio.sleep(120)  # wait for ECS health checks + allow manual cache clear first
        TOP = ["AAPL", "MSFT", "NVDA", "TSLA", "SPY", "QQQ", "AMZN", "META"]
        for ticker in TOP:
            try:
                cache_key = f"analysis:v6:{ticker}"
                if not await app.state.redis.exists(cache_key):
                    logger.info(f"Cache warmer: pre-warming {ticker}...")
                    data = await app.state.analyzer.run_full_analysis(
                        ticker=ticker,
                        include_options=False,
                        include_sentiment=True,
                        mc_paths=1000,
                    )
                    await app.state.redis.setex(
                        cache_key, 3600,
                        json.dumps(data, default=str)
                    )
                    logger.info(f"Cache warmer: ✅ {ticker} warmed")
                else:
                    logger.info(f"Cache warmer: {ticker} already cached")
                await asyncio.sleep(10)  # space out requests
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.warning(f"Cache warmer error {ticker}: {e}")
                await asyncio.sleep(5)

    app.state.warmer_task = asyncio.create_task(_startup_warmer())
    logger.info("✅ Cache warmer started — will pre-warm AAPL MSFT NVDA TSLA SPY QQQ AMZN META")

    logger.info("✅ QuantEdge v6.0 ready")
    yield

    # Shutdown
    logger.info("⏹ QuantEdge shutting down...")

    
    if app.state.warmer_task:
        app.state.warmer_task.cancel()
        try:
            await app.state.warmer_task
        except asyncio.CancelledError:
            pass

    if app.state.scheduler:
        app.state.scheduler.shutdown(wait=False)

    await app.state.redis.aclose()
    logger.info("✅ Redis pool closed")

    if app.state.db:
        await app.state.db.close()
    logger.info("✅ PostgreSQL pool closed")


# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="QuantEdge™ Institutional Analytics API",
    description="© 2026 Dileep Kumar Reddy Kapu. Proprietary & Confidential.",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS.split(",") if isinstance(settings.CORS_ORIGINS, str) else settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "font-src 'self' https://fonts.gstatic.com data:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "connect-src 'self' https://quant.dileepkapu.com wss://quant.dileepkapu.com"
    )
    return response


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    try:
        redis = request.app.state.redis
        client_ip = request.client.host
        key = f"ratelimit:{client_ip}"
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, 60)
        if count > 200:
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Max 200 requests/minute."},
            )
    except Exception:
        pass
    return await call_next(request)


@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    response.headers["X-Response-Time"] = f"{(time.time()-start)*1000:.1f}ms"
    return response


# ── Routers ────────────────────────────────────────────────────
app.include_router(auth_router.router,   prefix="/auth",               tags=["Authentication"])
app.include_router(analysis.router,      prefix="/api",                tags=["Analysis v5"])
app.include_router(v6_router,            prefix="/api/v6",             tags=["Analysis v6"])
app.include_router(performance_router,   prefix="/api/v6/performance", tags=["Performance"])
app.include_router(quality_router,       prefix="/api/v6",             tags=["Quality"])
app.include_router(screener_router,      prefix="/api/v6",             tags=["Screener"])
app.include_router(oracle_router,        prefix="/api/v1/oracle",      tags=["Price Oracle"])
app.include_router(watchlist.router,     prefix="/api",                tags=["Watchlist"])
app.include_router(portfolio.router,     prefix="/api",                tags=["Portfolio"])


@app.get("/health", tags=["System"])
async def health_check(request: Request):
    """ALB health check. Must return {"status":"healthy","redis":"ok","version":"6.0.0"}"""
    try:
        await request.app.state.redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "healthy" if redis_ok else "degraded",
        "version": "6.0.0",
        "platform": "QuantEdge™",
        "owner": "Dileep Kumar Reddy Kapu",
        "redis": "ok" if redis_ok else "error",
        "timestamp": time.time(),
    }


@app.websocket("/ws/stream/{ticker}")
async def websocket_stream(websocket: WebSocket, ticker: str):
    """
    Real-time price stream via Redis pub/sub channel ws:price:{ticker}.
    PolygonWebSocketManager publishes ticks to this channel.
    One upstream Polygon connection feeds unlimited browser clients.
    """
    await websocket.accept()
    ticker = ticker.upper()
    channel = f"ws:price:{ticker}"

    try:
        pubsub_conn = await aioredis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        pubsub = pubsub_conn.pubsub()
        await pubsub.subscribe(channel)

        while True:
            try:
                message = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=10.0,
                )
                if message and message.get("type") == "message":
                    await websocket.send_json(json.loads(message["data"]))
                else:
                    await websocket.send_json({
                        "type": "heartbeat",
                        "ticker": ticker,
                        "timestamp": time.time(),
                    })
            except asyncio.TimeoutError:
                await websocket.send_json({
                    "type": "heartbeat",
                    "ticker": ticker,
                    "timestamp": time.time(),
                })
    except Exception:
        logger.info(f"WebSocket disconnected: {ticker}")
    finally:
        try:
            await pubsub.unsubscribe(channel)
            await pubsub_conn.aclose()
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_v6:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=2,
        log_level="info",
    )
