#!/usr/bin/env python3
"""
QuantEdge v6.0 — Pre-Launch Check
====================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.

Run before EVERY deployment. All 9 checks must pass.
Exit code 0 = safe to deploy.
Exit code 1 = do NOT deploy — fix errors above first.

Usage:
    export POLYGON_API_KEY=...
    export ANTHROPIC_API_KEY=...
    export REDIS_URL=redis://localhost:6379/0
    export DATABASE_URL=postgresql://quantedge:quantedge@localhost:5432/quantedge
    export MODEL_DIR=./models
    python scripts/pre_launch_check.py
"""

import asyncio
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Tuple

# ── Colour helpers ──────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg: str) -> None:
    print(f"  {GREEN}✅ PASS{RESET}  {msg}")

def fail(msg: str) -> None:
    print(f"  {RED}❌ FAIL{RESET}  {msg}")

def info(msg: str) -> None:
    print(f"  {YELLOW}ℹ️  INFO{RESET}  {msg}")

def section(n: int, title: str) -> None:
    print(f"\n{BOLD}Check {n}/9 — {title}{RESET}")


# ── Environment ──────────────────────────────────────────────────────────────

REDIS_URL     = os.environ.get("REDIS_URL",     "redis://localhost:6379/0")
DATABASE_URL  = os.environ.get("DATABASE_URL",  "postgresql://quantedge:quantedge@localhost:5432/quantedge")
POLYGON_KEY   = os.environ.get("POLYGON_API_KEY", "")
ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
MODEL_DIR     = Path(os.environ.get("MODEL_DIR", "./models"))

REQUIRED_TABLES = ["signals", "performance_daily", "regime_performance", "model_weights_history"]
REQUIRED_MODELS = ["xgb_model.joblib", "lgb_model.joblib", "lstm_weights.pt"]


# ── Check functions ──────────────────────────────────────────────────────────

async def check_redis() -> Tuple[bool, str]:
    """Check 1 — Redis connection ping."""
    try:
        import redis.asyncio as aioredis
        client = await aioredis.from_url(REDIS_URL, socket_connect_timeout=5)
        await client.ping()
        await client.aclose()
        return True, f"Redis connected at {REDIS_URL}"
    except Exception as exc:
        return False, f"Redis ping failed: {exc}"


async def check_postgres() -> Tuple[bool, str]:
    """Check 2 — PostgreSQL connection and all 4 required tables exist."""
    try:
        import asyncpg
        db_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(db_url, timeout=10)

        missing = []
        for table in REQUIRED_TABLES:
            row = await conn.fetchrow(
                "SELECT EXISTS (SELECT 1 FROM information_schema.tables "
                "WHERE table_name = $1 AND table_schema = 'public')",
                table,
            )
            if not row["exists"]:
                missing.append(table)

        await conn.close()

        if missing:
            return False, f"Missing tables: {missing}. Run: psql $DATABASE_URL < db/schema.sql"
        return True, f"All 4 tables present: {REQUIRED_TABLES}"
    except Exception as exc:
        return False, f"PostgreSQL connection failed: {exc}"


async def check_polygon() -> Tuple[bool, str]:
    """Check 3 — Polygon API: fetch 5 days of AAPL OHLCV data."""
    if not POLYGON_KEY:
        return False, "POLYGON_API_KEY is not set"
    try:
        import aiohttp
        from datetime import date, timedelta
        end_date   = date.today()
        start_date = end_date - timedelta(days=10)

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/AAPL/range/1/day/"
            f"{start_date}/{end_date}"
            f"?adjusted=true&sort=asc&limit=10&apiKey={POLYGON_KEY}"
        )

        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 401:
                    return False, "Polygon API key is invalid (HTTP 401)"
                if resp.status == 403:
                    return False, "Polygon API key lacks required permissions (HTTP 403)"
                if resp.status != 200:
                    return False, f"Polygon API returned HTTP {resp.status}"
                data = await resp.json()

        results = data.get("results", [])
        if not results:
            return False, f"Polygon returned 0 bars for AAPL. Status: {data.get('status')}"

        return True, f"Polygon OK — AAPL returned {len(results)} bars"
    except Exception as exc:
        return False, f"Polygon API check failed: {exc}"


async def check_anthropic() -> Tuple[bool, str]:
    """Check 4 — Anthropic API: send one test message."""
    if not ANTHROPIC_KEY:
        return False, "ANTHROPIC_API_KEY is not set"
    try:
        from anthropic import AsyncAnthropic
        client = AsyncAnthropic(api_key=ANTHROPIC_KEY)
        msg = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=20,
            messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        )
        text = msg.content[0].text.strip().lower() if msg.content else ""
        if not text:
            return False, "Anthropic API returned empty response"
        return True, f"Anthropic API OK — response: '{text[:40]}'"
    except Exception as exc:
        return False, f"Anthropic API check failed: {exc}"


def check_model_files() -> Tuple[bool, str]:
    """Check 5 — ML model files exist in MODEL_DIR."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    missing = []
    found   = []
    for fname in REQUIRED_MODELS:
        path = MODEL_DIR / fname
        if path.exists() and path.stat().st_size > 0:
            found.append(fname)
        else:
            missing.append(fname)

    if missing:
        return (
            False,
            f"Missing model files in {MODEL_DIR}: {missing}. "
            "Train models first: on startup the system trains automatically if missing. "
            "Or trigger via /api/v6/admin/train if the server is running."
        )
    return True, f"All 3 model files present in {MODEL_DIR}"


async def check_full_analysis() -> Tuple[bool, str]:
    """
    Check 6 — Full AAPL analysis pipeline (mc_paths=100 for speed).
    Check 7 — Result has all 11 panels non-null.
    Returns two pass/fail decisions bundled together.
    """
    if not POLYGON_KEY:
        return False, "Skipped (POLYGON_API_KEY not set)"
    try:
        # Add backend to path so imports work when run from project root
        backend_dir = Path(__file__).resolve().parent.parent / "backend"
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))

        from routers.analysis_v6 import QuantEdgeAnalyzerV6

        analyzer = QuantEdgeAnalyzerV6()
        result = await asyncio.wait_for(
            analyzer.run_analysis("AAPL", mc_paths=100),
            timeout=180.0,
        )

        if not result:
            return False, "Analysis returned empty result"

        # Check 11 required panels are present and non-null
        required_panels = [
            "price_data",
            "regime_analysis",
            "garch_analysis",
            "ml_predictions",
            "risk_metrics",
            "portfolio_construction",
            "sentiment_analysis",
            "options_analysis",
            "fundamental_analysis",
            "investment_thesis",
            "signal_metadata",
        ]
        missing_panels = [p for p in required_panels if result.get(p) is None]

        if missing_panels:
            return False, f"Analysis ran but {len(missing_panels)} panels are null: {missing_panels}"

        return True, f"Full AAPL analysis passed — all {len(required_panels)} panels non-null"

    except asyncio.TimeoutError:
        return False, "Full analysis timed out after 180 seconds"
    except Exception as exc:
        return False, f"Full analysis error: {exc}"


async def check_signal_tracker() -> Tuple[bool, str]:
    """Check 8 — Signal tracker: write, read, delete one test record."""
    try:
        import asyncpg
        db_url = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
        conn = await asyncpg.connect(db_url, timeout=10)

        test_id = str(uuid.uuid4())
        test_ticker = "PRELAUNCH_TEST"

        # Write
        inserted_id = await conn.fetchval(
            """
            INSERT INTO signals (
                id, ticker, hmm_regime, hmm_confidence,
                ensemble_signal, weights_used
            )
            VALUES (
                $1::uuid, $2, 'Bull_Trending', 0.80,
                0.42, '{"test": true}'::jsonb
            )
            RETURNING id::text
            """,
            test_id, test_ticker,
        )

        if str(inserted_id) != test_id:
            await conn.close()
            return False, f"Signal tracker: insert returned wrong ID ({inserted_id})"

        # Read
        row = await conn.fetchrow(
            "SELECT ticker, ensemble_signal FROM signals WHERE id = $1::uuid",
            test_id,
        )
        if not row or row["ticker"] != test_ticker:
            await conn.close()
            return False, "Signal tracker: could not read back test record"

        # Delete
        await conn.execute(
            "DELETE FROM signals WHERE id = $1::uuid", test_id
        )

        # Verify deletion
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM signals WHERE id = $1::uuid", test_id
        )
        await conn.close()

        if count != 0:
            return False, "Signal tracker: test record was not deleted"

        return True, "Signal tracker write / read / delete OK"

    except Exception as exc:
        return False, f"Signal tracker check failed: {exc}"


async def check_finbert() -> Tuple[bool, str]:
    """Check 9 — FinBERT: run one sentence through the transformers pipeline."""
    try:
        # Import in a thread executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()

        def _run_finbert():
            from transformers import pipeline as hf_pipeline
            finbert = hf_pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                device=-1,  # CPU
            )
            text = "Apple reported record quarterly revenue beating analyst expectations."
            results = finbert([text], truncation=True, max_length=512)
            return results

        results = await loop.run_in_executor(None, _run_finbert)

        if not results or not isinstance(results, list):
            return False, "FinBERT returned no results"

        label = results[0].get("label", "UNKNOWN")
        score = results[0].get("score", 0.0)
        return True, f"FinBERT OK — label='{label}' score={score:.3f}"

    except Exception as exc:
        return False, f"FinBERT check failed: {exc}"


# ── Main ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  QuantEdge v6.0 — Pre-Launch Check{RESET}")
    print(f"{BOLD}  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}{RESET}")
    print(f"{BOLD}{'='*60}{RESET}")

    results = []  # list of (passed: bool, label: str)

    # ── Check 1: Redis ───────────────────────────────────────────────────────
    section(1, "Redis Connection")
    t0 = time.time()
    passed, msg = await check_redis()
    elapsed = time.time() - t0
    (ok if passed else fail)(f"{msg}  ({elapsed:.1f}s)")
    results.append(("Redis", passed))

    # ── Check 2: PostgreSQL ──────────────────────────────────────────────────
    section(2, "PostgreSQL — 4 required tables")
    t0 = time.time()
    passed, msg = await check_postgres()
    elapsed = time.time() - t0
    (ok if passed else fail)(f"{msg}  ({elapsed:.1f}s)")
    results.append(("PostgreSQL", passed))

    # ── Check 3: Polygon API ─────────────────────────────────────────────────
    section(3, "Polygon API — AAPL 5-day fetch")
    t0 = time.time()
    passed, msg = await check_polygon()
    elapsed = time.time() - t0
    (ok if passed else fail)(f"{msg}  ({elapsed:.1f}s)")
    results.append(("Polygon API", passed))

    # ── Check 4: Anthropic API ───────────────────────────────────────────────
    section(4, "Anthropic API — test message")
    t0 = time.time()
    passed, msg = await check_anthropic()
    elapsed = time.time() - t0
    (ok if passed else fail)(f"{msg}  ({elapsed:.1f}s)")
    results.append(("Anthropic API", passed))

    # ── Check 5: ML model files ──────────────────────────────────────────────
    section(5, "ML model files — xgb, lgb, lstm")
    t0 = time.time()
    passed, msg = check_model_files()
    elapsed = time.time() - t0
    (ok if passed else fail)(f"{msg}  ({elapsed:.1f}s)")
    results.append(("ML model files", passed))

    # ── Check 6 & 7: Full analysis + 11 panels ───────────────────────────────
    section(6, "Full AAPL analysis pipeline (mc_paths=100)")
    info("This may take 30–120 seconds. ML models run from scratch.")
    t0 = time.time()
    passed, msg = await check_full_analysis()
    elapsed = time.time() - t0
    (ok if passed else fail)(f"{msg}  ({elapsed:.1f}s)")
    results.append(("Full analysis + 11 panels", passed))

    # ── Check 8: Signal tracker ──────────────────────────────────────────────
    section(8, "Signal tracker — write / read / delete")
    t0 = time.time()
    passed, msg = await check_signal_tracker()
    elapsed = time.time() - t0
    (ok if passed else fail)(f"{msg}  ({elapsed:.1f}s)")
    results.append(("Signal tracker", passed))

    # ── Check 9: FinBERT ─────────────────────────────────────────────────────
    section(9, "FinBERT — ProsusAI/finbert sentiment pipeline")
    info("First run downloads ~440MB model. Subsequent runs use cache.")
    t0 = time.time()
    passed, msg = await check_finbert()
    elapsed = time.time() - t0
    (ok if passed else fail)(f"{msg}  ({elapsed:.1f}s)")
    results.append(("FinBERT", passed))

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{BOLD}{'='*60}{RESET}")
    all_passed = all(r[1] for r in results)
    failed_checks = [name for name, p in results if not p]

    if all_passed:
        print(f"\n{GREEN}{BOLD}  ✅ PRE-LAUNCH CHECK PASSED — safe to deploy{RESET}\n")
    else:
        print(f"\n{RED}{BOLD}  ❌ PRE-LAUNCH CHECK FAILED — do NOT deploy{RESET}")
        print(f"{RED}  Failed checks: {failed_checks}{RESET}\n")

    # Summary table
    print(f"{'Check':<30} {'Result':>10}")
    print("-" * 42)
    for name, passed in results:
        status = f"{GREEN}PASS{RESET}" if passed else f"{RED}FAIL{RESET}"
        print(f"  {name:<28} {status}")
    print()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    asyncio.run(main())
