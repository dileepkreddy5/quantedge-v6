"""
QuantEdge v6.0 — Signal Tracker
==================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.

Records every ML prediction to PostgreSQL using a shared connection pool.
Called as a FastAPI BackgroundTask — never blocks the main response.

Fills forward returns daily and computes IC so the research record
is always accurate and up to date.

Architecture:
    SignalTracker        — insert / fill / IC compute using app.state.db pool
    OutcomeFillerJob     — APScheduler wrapper, runs daily at 18:00 ET

Usage in analysis_v6.py endpoint:
    background_tasks.add_task(
        request.app.state.signal_tracker.record_signal,
        ticker=ticker,
        analysis_result=data,
        regime=regime,
        regime_confidence=regime_confidence,
        weights_used=weights_used,
    )
"""

import asyncio
import json
import uuid
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional
import asyncpg
from loguru import logger
from scipy import stats
import numpy as np


class SignalTracker:
    """
    Writes every ML analysis to the `signals` table and maintains
    the `performance_daily` table by filling forward returns and
    computing Spearman IC.

    Uses the shared asyncpg.Pool from app.state.db — zero per-call
    connections.
    """

    def __init__(self, db_pool: asyncpg.Pool):
        self.pool = db_pool

    # ── record_signal ─────────────────────────────────────────
    async def record_signal(
        self,
        ticker: str,
        analysis_result: dict,
        regime: str,
        regime_confidence: float,
        weights_used: dict,
    ) -> str:
        """
        Insert one row into the `signals` table.

        Extracts all model outputs and risk metrics from analysis_result.
        Returns the new signal UUID. Never raises — logs and silently
        returns empty string on failure so the main response is unaffected.
        """
        signal_id = str(uuid.uuid4())
        try:
            # Extract model outputs
            ml = analysis_result.get("ml_predictions", {})
            garch = ml.get("garch", ml.get("volatility", {}))
            kalman = ml.get("kalman", ml.get("kalman_filter", {}))
            lstm = ml.get("lstm", {})
            xgb = ml.get("xgboost", ml.get("xgb", {}))
            lgb = ml.get("lightgbm", ml.get("lgb", {}))
            ens = ml.get("ensemble", {})
            risk = analysis_result.get("risk_metrics", analysis_result.get("risk", {}))

            # Ensemble signal — try multiple key names for robustness
            ensemble_signal = float(
                ens.get("signal", ens.get("ensemble_signal", ens.get("pred_21d", 0.0))) or 0.0
            )
            ensemble_direction = _direction(ensemble_signal)

            # HMM state probs
            hmm_data = ml.get("hmm", ml.get("regime", {}))
            state_probs = hmm_data.get("state_probs", hmm_data.get("probabilities"))
            state_probs_json = json.dumps(state_probs) if state_probs else None

            # SHAP
            shap_data = xgb.get("shap_values", xgb.get("shap_top_drivers"))
            shap_json = json.dumps(shap_data) if shap_data else None

            async with self.pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO signals (
                        id, ticker, generated_at,
                        hmm_regime, hmm_confidence,
                        garch_vol_forecast, garch_regime,
                        hmm_state_probs,
                        kalman_trend, kalman_uncertainty,
                        lstm_pred_5d, lstm_pred_21d, lstm_pred_63d, lstm_uncertainty,
                        xgb_signal, xgb_confidence, xgb_shap_values,
                        lgb_signal, lgb_confidence,
                        ensemble_signal, ensemble_direction, weights_used,
                        cvar_95, vol_scale, recommended_position
                    ) VALUES (
                        $1,$2,$3,$4,$5,$6,$7,$8,$9,$10,
                        $11,$12,$13,$14,$15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25
                    )
                    """,
                    signal_id,
                    ticker.upper(),
                    datetime.utcnow(),
                    # Regime
                    str(regime)[:30],
                    float(regime_confidence or 0.0),
                    # GARCH
                    _f(garch.get("vol_forecast", garch.get("annualized_vol"))),
                    str(garch.get("regime", garch.get("garch_regime", "")))[:20] or None,
                    # HMM state probs
                    state_probs_json,
                    # Kalman
                    _f(kalman.get("trend", kalman.get("kalman_trend"))),
                    _f(kalman.get("uncertainty", kalman.get("kalman_uncertainty"))),
                    # LSTM
                    _f(lstm.get("pred_5d")),
                    _f(lstm.get("pred_21d")),
                    _f(lstm.get("pred_63d")),
                    _f(lstm.get("uncertainty", lstm.get("epistemic_uncertainty"))),
                    # XGBoost
                    _f(xgb.get("signal_strength", xgb.get("pred_21d"))),
                    _f(xgb.get("confidence")),
                    shap_json,
                    # LightGBM
                    _f(lgb.get("signal_strength", lgb.get("pred_21d"))),
                    _f(lgb.get("confidence")),
                    # Ensemble
                    ensemble_signal,
                    ensemble_direction,
                    json.dumps(weights_used),
                    # Risk
                    _f(risk.get("cvar_95", risk.get("cvar"))),
                    _f(risk.get("vol_scale", risk.get("volatility_scale"))),
                    _f(risk.get("recommended_position", risk.get("position_size"))),
                )

            logger.info(f"signal_tracker: recorded {signal_id[:8]} {ticker} {ensemble_direction}")
            return signal_id

        except Exception as e:
            # Signal tracking MUST NOT crash the main request path
            logger.warning(f"signal_tracker: failed to record {ticker}: {e}")
            return ""

    # ── fill_outcomes ─────────────────────────────────────────
    async def fill_outcomes(self) -> int:
        """
        Find all signals that have matured (5, 21, or 63 trading days old)
        and whose return columns are still NULL. Fetch the actual forward
        return from Polygon (or compute from price history in Redis cache)
        and update the row.

        Returns the count of rows updated.
        """
        updated = 0
        try:
            async with self.pool.acquire() as conn:
                # Find signals needing outcome fill
                rows = await conn.fetch(
                    """
                    SELECT id, ticker, generated_at,
                           ret_5d, ret_21d, ret_63d
                    FROM signals
                    WHERE (ret_5d IS NULL OR ret_21d IS NULL OR ret_63d IS NULL)
                      AND generated_at < NOW() - INTERVAL '5 days'
                    ORDER BY generated_at ASC
                    LIMIT 500
                    """
                )

                if not rows:
                    logger.info("signal_tracker.fill_outcomes: nothing to fill")
                    return 0

                # Group tickers to minimize API calls
                tickers: Dict[str, List[asyncpg.Record]] = {}
                for row in rows:
                    tickers.setdefault(row["ticker"], []).append(row)

                for ticker, ticker_rows in tickers.items():
                    try:
                        price_map = await _fetch_price_map(ticker)
                        if not price_map:
                            continue

                        for row in ticker_rows:
                            signal_date = row["generated_at"].date()
                            updates: Dict[str, Optional[float]] = {}

                            for horizon, col in [(5, "ret_5d"), (21, "ret_21d"), (63, "ret_63d")]:
                                if row[col] is not None:
                                    continue  # already filled
                                target_date = _add_trading_days(signal_date, horizon)
                                if target_date > date.today():
                                    continue  # not yet matured

                                entry_price = price_map.get(signal_date)
                                exit_price = price_map.get(target_date)
                                if entry_price and exit_price and entry_price > 0:
                                    updates[col] = (exit_price - entry_price) / entry_price

                            if updates:
                                set_clause = ", ".join(f"{col} = ${i+2}" for i, col in enumerate(updates))
                                values = list(updates.values())
                                await conn.execute(
                                    f"UPDATE signals SET {set_clause} WHERE id = $1",
                                    row["id"],
                                    *values,
                                )
                                updated += 1

                    except Exception as e:
                        logger.warning(f"fill_outcomes: error on {ticker}: {e}")
                        continue

        except Exception as e:
            logger.error(f"fill_outcomes: fatal error: {e}")

        logger.info(f"signal_tracker.fill_outcomes: updated {updated} rows")
        return updated

    # ── compute_daily_ic ──────────────────────────────────────
    async def compute_daily_ic(self) -> dict:
        """
        Compute Spearman IC between ensemble_signal and ret_21d
        using a rolling 63-day window of matured signals.

        Writes result to performance_daily table.
        Returns ic_21d, icir_21d, hit_rate, n_signals.
        """
        result = {"ic_21d": None, "icir_21d": None, "hit_rate": None, "n_signals": 0}
        try:
            async with self.pool.acquire() as conn:
                # Get signals with matured 21d returns, rolling 63 trading days
                rows = await conn.fetch(
                    """
                    SELECT ensemble_signal, ret_21d, xgb_signal, lgb_signal,
                           lstm_pred_21d, garch_vol_forecast, kalman_trend,
                           generated_at
                    FROM signals
                    WHERE ret_21d IS NOT NULL
                      AND generated_at >= NOW() - INTERVAL '90 days'
                    ORDER BY generated_at DESC
                    LIMIT 500
                    """
                )

                if len(rows) < 5:
                    logger.info("compute_daily_ic: not enough matured signals yet")
                    return result

                signals = np.array([float(r["ensemble_signal"] or 0) for r in rows])
                returns = np.array([float(r["ret_21d"] or 0) for r in rows])

                # Spearman IC
                ic, pval = stats.spearmanr(signals, returns)
                ic_21d = float(ic) if not np.isnan(ic) else 0.0

                # Hit rate
                hit_rate = float(np.mean((signals > 0) == (returns > 0)))

                # ICIR: IC / std(IC) — use rolling 21-day ICs if enough data
                ic_series = []
                for i in range(0, min(len(rows) - 5, 42), 1):
                    chunk_s = signals[i:i+21]
                    chunk_r = returns[i:i+21]
                    if len(chunk_s) >= 5:
                        ic_i, _ = stats.spearmanr(chunk_s, chunk_r)
                        if not np.isnan(ic_i):
                            ic_series.append(ic_i)

                icir = float(np.mean(ic_series) / np.std(ic_series)) if len(ic_series) >= 3 and np.std(ic_series) > 0 else None

                # Per-model ICs
                model_ics = {}
                for model_col, model_name in [
                    ("xgb_signal", "xgb"),
                    ("lgb_signal", "lgb"),
                    ("lstm_pred_21d", "lstm"),
                    ("kalman_trend", "kalman"),
                ]:
                    model_vals = np.array([float(r[model_col] or 0) for r in rows])
                    m_ic, _ = stats.spearmanr(model_vals, returns)
                    if not np.isnan(m_ic):
                        model_ics[model_name] = round(float(m_ic), 4)

                result = {
                    "ic_21d": round(ic_21d, 4),
                    "icir_21d": round(icir, 4) if icir else None,
                    "hit_rate": round(hit_rate, 4),
                    "n_signals": len(rows),
                    "model_ics": model_ics,
                }

                # Write to performance_daily
                today = date.today()
                await conn.execute(
                    """
                    INSERT INTO performance_daily (date, ic_21d, icir_21d, hit_rate, n_signals, model_ics)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (date) DO UPDATE SET
                        ic_21d = EXCLUDED.ic_21d,
                        icir_21d = EXCLUDED.icir_21d,
                        hit_rate = EXCLUDED.hit_rate,
                        n_signals = EXCLUDED.n_signals,
                        model_ics = EXCLUDED.model_ics
                    """,
                    today,
                    result["ic_21d"],
                    result["icir_21d"],
                    result["hit_rate"],
                    result["n_signals"],
                    json.dumps(model_ics),
                )

                logger.info(
                    f"compute_daily_ic: IC={ic_21d:.4f} ICIR={icir} "
                    f"hit_rate={hit_rate:.3f} n={len(rows)}"
                )

        except Exception as e:
            logger.error(f"compute_daily_ic: error: {e}")

        return result


# ── OutcomeFillerJob ───────────────────────────────────────────

class OutcomeFillerJob:
    """
    APScheduler-compatible job that fills forward returns daily at 18:00 ET
    and immediately computes/writes IC.

    Register in main_v6.py lifespan:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        scheduler = AsyncIOScheduler(timezone="America/New_York")
        scheduler.add_job(
            OutcomeFillerJob(signal_tracker).run,
            trigger="cron",
            hour=18, minute=0,
        )
        scheduler.start()
    """

    def __init__(self, signal_tracker: SignalTracker):
        self.tracker = signal_tracker

    async def run(self) -> None:
        logger.info("OutcomeFillerJob: starting daily run")
        try:
            n_filled = await self.tracker.fill_outcomes()
            logger.info(f"OutcomeFillerJob: filled {n_filled} outcomes")

            ic_result = await self.tracker.compute_daily_ic()
            logger.info(f"OutcomeFillerJob: IC={ic_result.get('ic_21d')} n={ic_result.get('n_signals')}")
        except Exception as e:
            logger.error(f"OutcomeFillerJob: error: {e}")


# ── Helpers ───────────────────────────────────────────────────

def _f(val) -> Optional[float]:
    """Safely cast to float, returning None on failure."""
    try:
        return float(val) if val is not None else None
    except (TypeError, ValueError):
        return None


def _direction(signal: float) -> str:
    if signal > 0.02:
        return "LONG"
    if signal < -0.02:
        return "SHORT"
    return "NEUTRAL"


def _add_trading_days(start: date, n: int) -> date:
    """Approximate: add n trading days (skipping weekends, no holiday calendar)."""
    d = start
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon–Fri
            added += 1
    return d


async def _fetch_price_map(ticker: str) -> Dict[date, float]:
    """
    Fetch closing price history for a ticker using Polygon REST.
    Returns {date: close_price} dict.
    Falls back to empty dict on any error.
    """
    import os
    import aiohttp
    from datetime import date as date_type

    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        return {}

    end_date = date.today()
    start_date = end_date - timedelta(days=120)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{ticker.upper()}/range/1/day"
        f"/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
    )
    params = {"adjusted": "true", "sort": "asc", "limit": 200}
    headers = {"Authorization": f"Bearer {api_key}"}

    price_map: Dict[date, float] = {}
    try:
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for bar in data.get("results", []):
                        ts = date_type.fromtimestamp(bar["t"] / 1000)
                        price_map[ts] = float(bar["c"])
    except Exception as e:
        logger.warning(f"_fetch_price_map {ticker}: {e}")

    return price_map
