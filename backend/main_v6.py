"""
QuantEdge v6.0 — FastAPI Application
=====================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.
Proprietary & Confidential.

Main API server for QuantEdge institutional quant platform.
Runs on ECS Fargate behind ALB + CloudFront.

Endpoints:
  POST /auth/login                      — Cognito MFA authentication
  POST /auth/refresh                    — Refresh JWT token
  DELETE /auth/logout                   — Invalidate session
  POST /api/analyze                     — Core analysis (v5 compatible)
  POST /api/v6/analyze                  — Full v6 institutional analysis
  POST /api/v1/oracle/analyze/{ticker}  — Price Oracle: full analysis
  POST /api/v1/oracle/sentiment/{ticker}— Price Oracle: sentiment only
  POST /api/v1/oracle/analysts/{ticker} — Price Oracle: analyst ratings
  GET  /api/watchlist                   — Personal watchlist
  POST /api/watchlist/add               — Add to watchlist
  GET  /api/portfolio                   — Portfolio analytics
  WS   /ws/stream/{ticker}              — Real-time price stream
  GET  /health                          — Health check
"""

from fastapi import FastAPI, Depends, HTTPException, Request, WebSocket, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Optional
import time
import asyncio
import redis.asyncio as aioredis
from loguru import logger

from core.config import settings
from auth.cognito_auth import get_current_user, CognitoUser
from routers import analysis, auth_router, watchlist, portfolio
from routers.analysis_v6 import router as v6_router

# Price Oracle routers
from ml.price_oracle.router import router as oracle_router


# ── Cache Warmer ───────────────────────────────────────────────
async def _cache_warmer(redis):
    TOP_TICKERS = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ", "MSFT", "AMZN", "META"]
    while True:
        try:
            await asyncio.sleep(900)
            for ticker in TOP_TICKERS:
                cache_key = f"analysis:v6:{ticker}"
                exists = await redis.exists(cache_key)
                if not exists:
                    await redis.setex(f"warm_needed:{ticker}", 60, "1")
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.warning(f"Cache warmer error: {e}")


# ── Lifespan: startup + shutdown ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 QuantEdge v6.0 starting...")
    logger.info("© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.")
    app.state.redis = await aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
    )
    logger.info(f"✅ Redis connected: {settings.REDIS_URL[:30]}...")
    warmer_task = asyncio.create_task(_cache_warmer(app.state.redis))
    app.state.warmer_task = warmer_task
    logger.info("✅ QuantEdge v6.0 ready — institutional quant platform is live")
    yield
    # Shutdown
    logger.info("⏹ QuantEdge shutting down...")
    warmer_task.cancel()
    try:
        await warmer_task
    except asyncio.CancelledError:
        pass
    await app.state.redis.aclose()


# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="QuantEdge™ Institutional Analytics API",
    description="© 2026 Dileep Kumar Reddy Kapu. Proprietary & Confidential.",
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# ── Gzip ───────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Security Headers ───────────────────────────────────────────
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

# ── Rate Limiting ──────────────────────────────────────────────
@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.url.path == "/health":
        return await call_next(request)
    # Simple IP-based rate limiting via Redis
    try:
        redis = request.app.state.redis
        client_ip = request.client.host
        key = f"ratelimit:{client_ip}"
        count = await redis.incr(key)
        if count == 1:
            await redis.expire(key, 60)  # 1-minute window
        if count > 200:  # 200 req/min per IP
            return JSONResponse(
                status_code=429,
                content={"detail": "Rate limit exceeded. Max 200 requests/minute."}
            )
    except Exception:
        pass
    return await call_next(request)

# ── Request Timing ─────────────────────────────────────────────
@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = (time.time() - start) * 1000
    response.headers["X-Response-Time"] = f"{elapsed:.1f}ms"
    return response


# ── Routers ────────────────────────────────────────────────────
app.include_router(auth_router.router, prefix="/auth", tags=["Authentication"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis v5"])
app.include_router(v6_router, prefix="/api/v6", tags=["Analysis v6"])
app.include_router(oracle_router, prefix="/api/v1/oracle", tags=["Price Oracle"])
app.include_router(watchlist.router, prefix="/api", tags=["Watchlist"])
app.include_router(portfolio.router, prefix="/api", tags=["Portfolio"])


# ── Health Check ───────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check(request: Request):
    """Health check endpoint. Used by ALB target group."""
    try:
        redis = request.app.state.redis
        await redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False

    return {
        "status": "healthy" if redis_ok else "degraded",
        "version": "6.0.0",
        "platform": "QuantEdge™",
        "owner": "Dileep Kumar Reddy Kapu",
        "redis": "ok" if redis_ok else "error",
        "timestamp": time.time()
    }


# ── WebSocket: Real-Time Price Stream ─────────────────────────
@app.websocket("/ws/stream/{ticker}")
async def websocket_stream(websocket: WebSocket, ticker: str):
    """Stream real-time price updates for a ticker."""
    await websocket.accept()
    logger.info(f"WebSocket connected: {ticker}")
    try:
        while True:
            try:
                import yfinance as yf
                t = yf.Ticker(ticker)
                info = t.fast_info
                price = float(getattr(info, 'last_price', 0) or 0)
                change = float(getattr(info, 'regular_market_change', 0) or 0)
                change_pct = float(getattr(info, 'regular_market_change_percent', 0) or 0)

                await websocket.send_json({
                    "ticker": ticker,
                    "price": price,
                    "change": change,
                    "change_pct": change_pct,
                    "timestamp": time.time()
                })
            except Exception as e:
                await websocket.send_json({"error": str(e), "ticker": ticker})

            await asyncio.sleep(5)  # 5-second updates
    except Exception:
        logger.info(f"WebSocket disconnected: {ticker}")
    finally:
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
        log_level="info"
    )
