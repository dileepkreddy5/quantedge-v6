"""
QuantEdge v5.0 — FastAPI Application
=====================================
Main API server for Dileep's institutional quant platform.
Runs on ECS Fargate behind ALB + CloudFront.

Endpoints:
  POST /auth/login           — Cognito MFA authentication
  POST /auth/refresh         — Refresh JWT token
  DELETE /auth/logout        — Invalidate session
  POST /api/analyze          — Full institutional analysis
  GET  /api/watchlist        — Personal watchlist
  POST /api/watchlist/add    — Add to watchlist
  GET  /api/portfolio        — Portfolio analytics
  WS   /ws/stream/{ticker}   — Real-time price stream
  GET  /health               — Health check
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


# ── Lifespan: startup + shutdown ───────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 QuantEdge v5.0 starting...")
    app.state.redis = await aioredis.from_url(
        settings.REDIS_URL,
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
    )
    logger.info(f"✅ Redis connected: {settings.REDIS_URL[:20]}...")
    logger.info("✅ QuantEdge ready — Dileep's quant platform is live")
    yield
    # Shutdown
    logger.info("⏹ QuantEdge shutting down...")
    await app.state.redis.close()


# ── App Factory ────────────────────────────────────────────────
app = FastAPI(
    title="QuantEdge v5.0",
    description="Institutional Quantitative Analytics — Dileep Kumar Reddy Kapu",
    version="5.0.0",
    docs_url=None,        # Disabled in production (security)
    redoc_url=None,
    openapi_url=None,     # No public API schema
    lifespan=lifespan,
)


# ── Middleware Stack ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "PUT"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)


# ── Security Middleware ────────────────────────────────────────
@app.middleware("http")
async def security_headers(request: Request, call_next):
    """Add security headers to all responses"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    # Security headers (OWASP recommendations)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self' https://api.anthropic.com;"
    )
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-App-Version"] = "5.0.0"

    # Log request
    logger.info(
        f"{request.method} {request.url.path} "
        f"status={response.status_code} "
        f"time={process_time:.3f}s"
    )
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Redis-based rate limiting: 60 req/min per IP"""
    # Skip health checks
    if request.url.path == "/health":
        return await call_next(request)

    client_ip = request.headers.get("X-Forwarded-For", request.client.host).split(",")[0].strip()
    redis_key = f"rl:{client_ip}"

    try:
        redis = request.app.state.redis
        current = await redis.incr(redis_key)
        if current == 1:
            await redis.expire(redis_key, 60)

        if current > settings.RATE_LIMIT_PER_MINUTE:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded. Try again in 60 seconds."},
                headers={"Retry-After": "60"},
            )
    except Exception:
        pass  # If Redis fails, don't block requests

    return await call_next(request)


@app.middleware("http")
async def ip_allowlist_middleware(request: Request, call_next):
    """Optional IP allowlist — only Dileep's IPs can access"""
    if not settings.ALLOWED_IPS:
        return await call_next(request)

    if request.url.path == "/health":
        return await call_next(request)

    client_ip = request.headers.get("X-Forwarded-For", request.client.host).split(",")[0].strip()
    if client_ip not in settings.ALLOWED_IPS:
        logger.warning(f"⛔ Blocked IP: {client_ip} → {request.url.path}")
        return JSONResponse(
            status_code=status.HTTP_403_FORBIDDEN,
            content={"error": "Access restricted"},
        )
    return await call_next(request)


# ── Routes ────────────────────────────────────────────────────
app.include_router(auth_router.router, prefix="/auth", tags=["Authentication"])
app.include_router(analysis.router, prefix="/api", tags=["Analysis"])
app.include_router(watchlist.router, prefix="/api", tags=["Watchlist"])
app.include_router(portfolio.router, prefix="/api", tags=["Portfolio"])


# ── Health & Root ──────────────────────────────────────────────
@app.get("/health")
async def health_check(request: Request):
    """Health check for ALB and ECS"""
    try:
        await request.app.state.redis.ping()
        redis_ok = True
    except Exception:
        redis_ok = False
    return {
        "status": "healthy",
        "version": "5.0.0",
        "redis": "connected" if redis_ok else "error",
        "owner": "Dileep Kumar Reddy Kapu",
        "platform": "quant.dileepkapu.com",
    }


@app.get("/")
async def root():
    return {"message": "QuantEdge v5.0 — Institutional Quantitative Analytics"}


# ── WebSocket: Real-time Price Streaming ───────────────────────
@app.websocket("/ws/stream/{ticker}")
async def websocket_stream(
    websocket: WebSocket,
    ticker: str,
    token: str,  # JWT passed as query param for WS auth
):
    """
    Real-time price streaming via WebSocket.
    Requires valid JWT token as query param.
    Streams OHLCV + signal updates every 5 seconds.
    """
    from auth.cognito_auth import verify_cognito_token
    try:
        payload = verify_cognito_token(token)
        if payload.get("username") != settings.OWNER_USERNAME:
            await websocket.close(code=4001, reason="Unauthorized")
            return
    except Exception:
        await websocket.close(code=4001, reason="Invalid token")
        return

    await websocket.accept()
    logger.info(f"📡 WebSocket opened: {ticker}")

    try:
        import yfinance as yf
        while True:
            try:
                stock = yf.Ticker(ticker)
                info = stock.fast_info
                data = {
                    "ticker": ticker.upper(),
                    "price": float(info.last_price) if info.last_price else None,
                    "volume": float(info.shares) if hasattr(info, "shares") else None,
                    "timestamp": int(time.time()),
                    "market_cap": float(info.market_cap) if hasattr(info, "market_cap") else None,
                }
                await websocket.send_json(data)
            except Exception as e:
                await websocket.send_json({"error": str(e), "ticker": ticker})

            await asyncio.sleep(5)
    except Exception:
        logger.info(f"📡 WebSocket closed: {ticker}")


# ── Exception Handlers ─────────────────────────────────────────
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "Endpoint not found"})


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error. This has been logged."}
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=2,
        log_level="info",
        access_log=True,
    )
