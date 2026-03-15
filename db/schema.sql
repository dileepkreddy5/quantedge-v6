-- QuantEdge v6.0 — PostgreSQL Schema
-- © 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.
--
-- Tables:
--   signals              — every ML prediction with all model outputs
--   performance_daily    — daily IC, ICIR, hit rate aggregates
--   regime_performance   — per-regime performance statistics
--   model_weights_history — historical record of ensemble weight changes
--
-- Applied automatically via docker-compose volume mount:
--   ./db/schema.sql:/docker-entrypoint-initdb.d/schema.sql
--
-- For RDS: psql $DATABASE_URL < db/schema.sql

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- ── signals ───────────────────────────────────────────────────
-- One row per analysis. forward returns (ret_5d, ret_21d, ret_63d)
-- are NULL at insert time and filled later by OutcomeFillerJob.

CREATE TABLE IF NOT EXISTS signals (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    ticker              VARCHAR(10)  NOT NULL,
    generated_at        TIMESTAMPTZ  NOT NULL DEFAULT NOW(),

    -- Regime state (from HMM 5-state model)
    hmm_regime          VARCHAR(30)  NOT NULL,
    hmm_confidence      FLOAT        NOT NULL,

    -- Individual model outputs
    garch_vol_forecast  FLOAT,
    garch_regime        VARCHAR(20),
    hmm_state_probs     JSONB,

    kalman_trend        FLOAT,
    kalman_uncertainty  FLOAT,

    lstm_pred_5d        FLOAT,
    lstm_pred_21d       FLOAT,
    lstm_pred_63d       FLOAT,
    lstm_uncertainty    FLOAT,

    xgb_signal          FLOAT,
    xgb_confidence      FLOAT,
    xgb_shap_values     JSONB,

    lgb_signal          FLOAT,
    lgb_confidence      FLOAT,

    -- Ensemble
    ensemble_signal     FLOAT        NOT NULL,
    ensemble_direction  VARCHAR(10),
    weights_used        JSONB        NOT NULL,

    -- Risk outputs
    cvar_95             FLOAT,
    vol_scale           FLOAT,
    recommended_position FLOAT,

    -- Forward returns (filled by OutcomeFillerJob after 5/21/63 days)
    ret_5d              FLOAT,
    ret_21d             FLOAT,
    ret_63d             FLOAT,
    barrier_hit         VARCHAR(20),   -- 'take_profit' | 'stop_loss' | 'timeout'

    -- IC contribution (filled by compute_daily_ic)
    ic_contribution     FLOAT
);

CREATE INDEX IF NOT EXISTS idx_signals_ticker
    ON signals (ticker);

CREATE INDEX IF NOT EXISTS idx_signals_generated_at
    ON signals (generated_at DESC);

CREATE INDEX IF NOT EXISTS idx_signals_regime
    ON signals (hmm_regime);

CREATE INDEX IF NOT EXISTS idx_signals_ticker_date
    ON signals (ticker, generated_at DESC);

-- For OutcomeFillerJob: efficiently find signals needing outcome fill
CREATE INDEX IF NOT EXISTS idx_signals_ret21d_null
    ON signals (generated_at)
    WHERE ret_21d IS NULL;


-- ── performance_daily ─────────────────────────────────────────
-- One row per calendar day, written by compute_daily_ic().
-- Tracks rolling Spearman IC between ensemble_signal and ret_21d.

CREATE TABLE IF NOT EXISTS performance_daily (
    date        DATE  PRIMARY KEY,
    ic_21d      FLOAT,
    icir_21d    FLOAT,
    hit_rate    FLOAT,
    n_signals   INTEGER,
    model_ics   JSONB   -- {"xgb": 0.065, "lgb": 0.071, "lstm": 0.058, ...}
);


-- ── regime_performance ────────────────────────────────────────
-- IC statistics broken down by HMM regime.
-- Tells us which models work in which regimes.

CREATE TABLE IF NOT EXISTS regime_performance (
    regime          VARCHAR(30)  NOT NULL,
    period_start    DATE         NOT NULL,
    period_end      DATE,
    mean_ic         FLOAT,
    icir            FLOAT,
    hit_rate        FLOAT,
    n_signals       INTEGER,
    PRIMARY KEY (regime, period_start)
);


-- ── model_weights_history ─────────────────────────────────────
-- Records every change to ensemble weights.
-- Enables auditing of how the RegimeConditionalEnsemble evolved.

CREATE TABLE IF NOT EXISTS model_weights_history (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    recorded_at     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    regime          VARCHAR(30)  NOT NULL,
    weights         JSONB        NOT NULL,   -- {"lstm": 0.30, "xgb": 0.25, ...}
    ic_basis        FLOAT,                   -- IC that justified the weight change
    n_signals_used  INTEGER
);

CREATE INDEX IF NOT EXISTS idx_weights_history_regime
    ON model_weights_history (regime, recorded_at DESC);


-- Grant access to application user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO quantedge;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO quantedge;
