"""
QuantEdge v6.0 — Performance Router
======================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.

4 read-only endpoints for the live track record dashboard.
All data comes from PostgreSQL — never blocks, uses asyncpg pool.
All endpoints require Cognito authentication.

Endpoints:
    GET /api/v6/performance/track_record
    GET /api/v6/performance/regime_ic
    GET /api/v6/performance/signal_history/{ticker}
    GET /api/v6/performance/decay_curve/{model}
"""

import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import asyncpg
from fastapi import APIRouter, Depends, HTTPException, Request
from loguru import logger

from auth.cognito_auth import get_current_user, CognitoUser

router = APIRouter(prefix="/api/v6/performance", tags=["performance"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pool(request: Request) -> asyncpg.Pool:
    """Extract the asyncpg pool attached to app state."""
    pool = getattr(request.app.state, "db", None)
    if pool is None:
        raise HTTPException(
            status_code=503,
            detail="Database pool not initialised. Check startup logs.",
        )
    return pool


def _safe_float(v: Any) -> Optional[float]:
    """Convert to float; return None on NaN or None."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# GET /api/v6/performance/track_record
# ---------------------------------------------------------------------------

@router.get("/track_record")
async def track_record(
    request: Request,
    current_user: CognitoUser = Depends(get_current_user),
) -> Dict:
    """
    Aggregate performance statistics since inception.

    Returns:
        ic_21d         — mean Spearman IC on 21-day forward returns
        icir_21d       — IC / std(IC), rolling 63-day
        hit_rate       — fraction of signals where sign(ensemble_signal) == sign(ret_21d)
        sharpe_since_inception — IC-based Sharpe (annualised)
        n_signals      — total signals with realised 21d outcomes
        since_date     — date of first signal
        last_updated   — date of most recent performance_daily row
    """
    pool = _pool(request)
    try:
        async with pool.acquire() as conn:
            # Latest row from performance_daily for ic_21d / icir / hit_rate
            perf_row = await conn.fetchrow(
                """
                SELECT date, ic_21d, icir_21d, hit_rate, n_signals
                FROM performance_daily
                ORDER BY date DESC
                LIMIT 1
                """
            )

            # Count total matured signals
            totals = await conn.fetchrow(
                """
                SELECT
                    COUNT(*)                                    AS n_signals,
                    MIN(generated_at)::date                    AS since_date,
                    AVG(CASE
                        WHEN ensemble_signal IS NOT NULL
                         AND ret_21d IS NOT NULL
                         AND SIGN(ensemble_signal) = SIGN(ret_21d) THEN 1.0
                        WHEN ensemble_signal IS NOT NULL
                         AND ret_21d IS NOT NULL THEN 0.0
                    END)                                       AS hit_rate_all
                FROM signals
                WHERE ret_21d IS NOT NULL
                """
            )

            # Compute annualised Sharpe from IC series
            ic_rows = await conn.fetch(
                """
                SELECT ic_21d
                FROM performance_daily
                WHERE ic_21d IS NOT NULL
                ORDER BY date DESC
                LIMIT 252
                """
            )
            ic_values = [float(r["ic_21d"]) for r in ic_rows if r["ic_21d"] is not None]

            sharpe = None
            if len(ic_values) >= 2:
                import statistics
                mean_ic = statistics.mean(ic_values)
                std_ic = statistics.stdev(ic_values)
                if std_ic > 0:
                    # Approximate: signals generated ~daily, annualise by sqrt(252)
                    sharpe = round((mean_ic / std_ic) * (252 ** 0.5), 3)

        result = {
            "ic_21d": _safe_float(perf_row["ic_21d"]) if perf_row else None,
            "icir_21d": _safe_float(perf_row["icir_21d"]) if perf_row else None,
            "hit_rate": _safe_float(totals["hit_rate_all"]) if totals else None,
            "sharpe_since_inception": sharpe,
            "n_signals": int(totals["n_signals"]) if totals and totals["n_signals"] else 0,
            "since_date": str(totals["since_date"]) if totals and totals["since_date"] else None,
            "last_updated": str(perf_row["date"]) if perf_row and perf_row["date"] else None,
        }
        return result

    except asyncpg.PostgresError as exc:
        logger.error(f"track_record DB error: {exc}")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


# ---------------------------------------------------------------------------
# GET /api/v6/performance/regime_ic
# ---------------------------------------------------------------------------

@router.get("/regime_ic")
async def regime_ic(
    request: Request,
    current_user: CognitoUser = Depends(get_current_user),
) -> Dict:
    """
    5×5 IC matrix: 5 HMM regimes × 5 models.

    Returns:
        {
          "Bull_Trending": {"xgb": 0.067, "lgb": 0.071, "lstm": 0.058,
                            "garch": 0.031, "kalman": 0.044, "n": 42},
          "Bull_Volatile":  {...},
          "Mean_Reverting": {...},
          "Bear_Trending":  {...},
          "Crisis":         {...},
        }

    Computed as Spearman rank correlation between each model's signal
    and realised ret_21d, grouped by HMM regime.
    Requires scipy on the DB query server — computed in Python from raw rows.
    """
    pool = _pool(request)

    regimes = [
        "Bull_Trending", "Bull_Volatile", "Mean_Reverting",
        "Bear_Trending", "Crisis",
    ]
    model_signal_cols = {
        "xgb": "xgb_signal",
        "lgb": "lgb_signal",
        "lstm": "lstm_pred_21d",
        "garch": "garch_vol_forecast",
        "kalman": "kalman_trend",
    }

    try:
        from scipy import stats as scipy_stats

        result: Dict[str, Dict] = {}

        async with pool.acquire() as conn:
            for regime in regimes:
                rows = await conn.fetch(
                    """
                    SELECT
                        xgb_signal, lgb_signal,
                        lstm_pred_21d, garch_vol_forecast,
                        kalman_trend, ret_21d
                    FROM signals
                    WHERE hmm_regime = $1
                      AND ret_21d IS NOT NULL
                    ORDER BY generated_at DESC
                    LIMIT 500
                    """,
                    regime,
                )

                if len(rows) < 5:
                    result[regime] = {m: None for m in model_signal_cols}
                    result[regime]["n"] = len(rows)
                    continue

                import numpy as np
                ret_21d_arr = np.array([float(r["ret_21d"]) for r in rows])
                regime_result: Dict = {}

                for model_name, col in model_signal_cols.items():
                    signals = np.array([
                        float(r[col]) if r[col] is not None else np.nan
                        for r in rows
                    ])
                    valid = ~np.isnan(signals)
                    if valid.sum() < 5:
                        regime_result[model_name] = None
                        continue
                    ic, pval = scipy_stats.spearmanr(signals[valid], ret_21d_arr[valid])
                    regime_result[model_name] = round(float(ic), 4) if not math.isnan(ic) else None

                regime_result["n"] = len(rows)
                result[regime] = regime_result

        return result

    except asyncpg.PostgresError as exc:
        logger.error(f"regime_ic DB error: {exc}")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    except Exception as exc:
        logger.error(f"regime_ic computation error: {exc}")
        raise HTTPException(status_code=500, detail=f"Computation error: {exc}")


# ---------------------------------------------------------------------------
# GET /api/v6/performance/signal_history/{ticker}
# ---------------------------------------------------------------------------

@router.get("/signal_history/{ticker}")
async def signal_history(
    ticker: str,
    request: Request,
    current_user: CognitoUser = Depends(get_current_user),
    limit: int = 100,
) -> List[Dict]:
    """
    Last N signals for a specific ticker, with all model outputs and outcomes.

    Returns list of dicts with:
        id, ticker, generated_at, hmm_regime, hmm_confidence,
        ensemble_signal, ensemble_direction, recommended_position,
        xgb_signal, xgb_confidence,
        lgb_signal, lgb_confidence,
        lstm_pred_5d, lstm_pred_21d, lstm_pred_63d, lstm_uncertainty,
        garch_vol_forecast, kalman_trend,
        cvar_95, vol_scale,
        ret_5d, ret_21d, ret_63d,   ← null until outcomes filled
        ic_contribution
    """
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 10:
        raise HTTPException(status_code=400, detail="Invalid ticker symbol")

    pool = _pool(request)
    cap = min(max(1, limit), 500)  # clamp 1–500

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    id::text                AS id,
                    ticker,
                    generated_at,
                    hmm_regime,
                    hmm_confidence,
                    ensemble_signal,
                    ensemble_direction,
                    recommended_position,
                    xgb_signal,
                    xgb_confidence,
                    lgb_signal,
                    lgb_confidence,
                    lstm_pred_5d,
                    lstm_pred_21d,
                    lstm_pred_63d,
                    lstm_uncertainty,
                    garch_vol_forecast,
                    kalman_trend,
                    kalman_uncertainty,
                    cvar_95,
                    vol_scale,
                    ret_5d,
                    ret_21d,
                    ret_63d,
                    barrier_hit,
                    ic_contribution
                FROM signals
                WHERE ticker = $1
                ORDER BY generated_at DESC
                LIMIT $2
                """,
                ticker,
                cap,
            )

        return [
            {
                "id": r["id"],
                "ticker": r["ticker"],
                "generated_at": r["generated_at"].isoformat() if r["generated_at"] else None,
                "hmm_regime": r["hmm_regime"],
                "hmm_confidence": _safe_float(r["hmm_confidence"]),
                "ensemble_signal": _safe_float(r["ensemble_signal"]),
                "ensemble_direction": r["ensemble_direction"],
                "recommended_position": _safe_float(r["recommended_position"]),
                "xgb_signal": _safe_float(r["xgb_signal"]),
                "xgb_confidence": _safe_float(r["xgb_confidence"]),
                "lgb_signal": _safe_float(r["lgb_signal"]),
                "lgb_confidence": _safe_float(r["lgb_confidence"]),
                "lstm_pred_5d": _safe_float(r["lstm_pred_5d"]),
                "lstm_pred_21d": _safe_float(r["lstm_pred_21d"]),
                "lstm_pred_63d": _safe_float(r["lstm_pred_63d"]),
                "lstm_uncertainty": _safe_float(r["lstm_uncertainty"]),
                "garch_vol_forecast": _safe_float(r["garch_vol_forecast"]),
                "kalman_trend": _safe_float(r["kalman_trend"]),
                "kalman_uncertainty": _safe_float(r["kalman_uncertainty"]),
                "cvar_95": _safe_float(r["cvar_95"]),
                "vol_scale": _safe_float(r["vol_scale"]),
                "ret_5d": _safe_float(r["ret_5d"]),
                "ret_21d": _safe_float(r["ret_21d"]),
                "ret_63d": _safe_float(r["ret_63d"]),
                "barrier_hit": r["barrier_hit"],
                "ic_contribution": _safe_float(r["ic_contribution"]),
            }
            for r in rows
        ]

    except asyncpg.PostgresError as exc:
        logger.error(f"signal_history DB error: {exc}")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")


# ---------------------------------------------------------------------------
# GET /api/v6/performance/decay_curve/{model}
# ---------------------------------------------------------------------------

VALID_MODELS = {"xgb", "lgb", "lstm", "garch", "kalman", "ensemble"}

@router.get("/decay_curve/{model}")
async def decay_curve(
    model: str,
    request: Request,
    current_user: CognitoUser = Depends(get_current_user),
) -> List[Dict]:
    """
    IC over time for a specified model.

    model: one of xgb | lgb | lstm | garch | kalman | ensemble

    Returns list of:
        {"date": "2026-01-15", "ic_21d": 0.063, "icir": 1.7, "n": 42}

    The IC is computed daily as Spearman(model_signal, ret_21d)
    over a rolling 63-signal window for the specified model.
    """
    model = model.lower().strip()
    if model not in VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Valid options: {sorted(VALID_MODELS)}",
        )

    pool = _pool(request)

    # Map model name to the signal column in the signals table
    model_col_map = {
        "xgb": "xgb_signal",
        "lgb": "lgb_signal",
        "lstm": "lstm_pred_21d",
        "garch": "garch_vol_forecast",
        "kalman": "kalman_trend",
        "ensemble": "ensemble_signal",
    }
    signal_col = model_col_map[model]

    try:
        from scipy import stats as scipy_stats
        import numpy as np

        async with pool.acquire() as conn:
            rows = await conn.fetch(
                f"""
                SELECT
                    generated_at::date  AS signal_date,
                    {signal_col}        AS model_signal,
                    ret_21d
                FROM signals
                WHERE {signal_col} IS NOT NULL
                  AND ret_21d IS NOT NULL
                ORDER BY generated_at ASC
                """,
            )

        if len(rows) < 5:
            return []

        dates = [r["signal_date"] for r in rows]
        signals = np.array([float(r["model_signal"]) for r in rows])
        returns = np.array([float(r["ret_21d"]) for r in rows])

        WINDOW = 63
        result = []

        for i in range(WINDOW - 1, len(rows)):
            window_signals = signals[max(0, i - WINDOW + 1): i + 1]
            window_returns = returns[max(0, i - WINDOW + 1): i + 1]
            n = len(window_signals)

            if n < 5:
                continue

            ic, _ = scipy_stats.spearmanr(window_signals, window_returns)
            if math.isnan(ic):
                continue

            # Rolling ICIR: mean IC / std IC over sub-windows inside this window
            # Approximate with the single IC value for now;
            # true ICIR needs IC per signal which requires sub-windowing
            icir: Optional[float] = None
            if n >= 10:
                # Compute per-signal IC contribution: sign match approach
                ic_contributions = np.sign(window_signals) == np.sign(window_returns)
                mean_hit = float(ic_contributions.mean())
                std_hit = float(ic_contributions.std()) if ic_contributions.std() > 0 else 1.0
                icir = round((mean_hit - 0.5) / std_hit * (n ** 0.5), 3)

            result.append({
                "date": str(dates[i]),
                "ic_21d": round(float(ic), 4),
                "icir": icir,
                "n": n,
            })

        return result

    except asyncpg.PostgresError as exc:
        logger.error(f"decay_curve DB error: {exc}")
        raise HTTPException(status_code=500, detail=f"Database error: {exc}")
    except Exception as exc:
        logger.error(f"decay_curve computation error: {exc}")
        raise HTTPException(status_code=500, detail=f"Computation error: {exc}")
