"""
QuantEdge v6.0 — Data Pipeline: Fetch & Store
==============================================
© 2026 Dileep Kumar Reddy Kapu. All Rights Reserved.
Proprietary & Confidential.

Fetches OHLCV, fundamentals, macro data from yFinance + FRED
and stores to PostgreSQL. Run daily via cron or Celery Beat.

Usage:
    python fetch_and_store.py --tickers AAPL MSFT NVDA --days 365
    python fetch_and_store.py --universe sp500 --days 90
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime, timedelta
from typing import List, Optional
import pandas as pd
import numpy as np
import yfinance as yf
import sqlalchemy
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from loguru import logger
import time

# ── Config ────────────────────────────────────────────────────
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://quantedge:password@localhost:5432/quantedge"
)
FRED_API_KEY = os.environ.get("FRED_API_KEY", "")

# ── S&P 500 Universe (top 50 by market cap for personal use) ──
SP500_TOP50 = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "LLY", "JPM",
    "AVGO", "XOM", "TSLA", "UNH", "V", "JNJ", "PG", "MA", "COST", "HD",
    "MRK", "ABBV", "CVX", "WMT", "BAC", "AMD", "KO", "PEP", "TMO", "ORCL",
    "ADBE", "CSCO", "CRM", "ACN", "MCD", "NFLX", "ABT", "TXN", "DHR", "WFC",
    "AMGN", "INTC", "VZ", "INTU", "QCOM", "IBM", "RTX", "GE", "NOW", "SPGI"
]


# ── Database Setup ────────────────────────────────────────────
def get_engine():
    engine = create_engine(DB_URL, pool_size=5, max_overflow=10)
    return engine


def create_tables(engine):
    """Create tables if they don't exist."""
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ohlcv (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                open FLOAT,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                adj_close FLOAT,
                volume BIGINT,
                UNIQUE(ticker, date)
            );
            CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date ON ohlcv(ticker, date DESC);

            CREATE TABLE IF NOT EXISTS fundamentals (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) NOT NULL,
                fetched_at TIMESTAMP DEFAULT NOW(),
                market_cap BIGINT,
                pe_ratio FLOAT,
                forward_pe FLOAT,
                pb_ratio FLOAT,
                ps_ratio FLOAT,
                ev_ebitda FLOAT,
                debt_to_equity FLOAT,
                roe FLOAT,
                roa FLOAT,
                profit_margin FLOAT,
                revenue_growth FLOAT,
                earnings_growth FLOAT,
                beta FLOAT,
                dividend_yield FLOAT,
                sector VARCHAR(100),
                industry VARCHAR(200),
                UNIQUE(ticker, fetched_at::date)
            );

            CREATE TABLE IF NOT EXISTS macro_data (
                id SERIAL PRIMARY KEY,
                series_id VARCHAR(50) NOT NULL,
                date DATE NOT NULL,
                value FLOAT,
                UNIQUE(series_id, date)
            );

            CREATE TABLE IF NOT EXISTS model_signals (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(20) NOT NULL,
                date DATE NOT NULL,
                composite_signal FLOAT,
                garch_vol FLOAT,
                hurst_exp FLOAT,
                kelly_fraction FLOAT,
                momentum_1m FLOAT,
                momentum_3m FLOAT,
                rsi_14 FLOAT,
                created_at TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, date)
            );
        """))
        conn.commit()
    logger.info("✅ Tables created/verified")


# ── OHLCV Fetcher ─────────────────────────────────────────────
def fetch_ohlcv(ticker: str, days: int = 365) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data for a ticker."""
    try:
        end = datetime.now()
        start = end - timedelta(days=days)
        t = yf.Ticker(ticker)
        df = t.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))
        if df.empty:
            logger.warning(f"No data for {ticker}")
            return None
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        df['ticker'] = ticker
        # yfinance returns 'close' which is adj close, keep raw too
        if 'close' in df.columns:
            df['adj_close'] = df['close']
        logger.info(f"✅ {ticker}: {len(df)} rows fetched")
        return df
    except Exception as e:
        logger.error(f"❌ Failed to fetch {ticker}: {e}")
        return None


def store_ohlcv(df: pd.DataFrame, engine):
    """Upsert OHLCV data to DB."""
    if df is None or df.empty:
        return 0
    try:
        rows_inserted = 0
        with engine.connect() as conn:
            for _, row in df.iterrows():
                conn.execute(text("""
                    INSERT INTO ohlcv (ticker, date, open, high, low, close, adj_close, volume)
                    VALUES (:ticker, :date, :open, :high, :low, :close, :adj_close, :volume)
                    ON CONFLICT (ticker, date) DO UPDATE SET
                        open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low = EXCLUDED.low,
                        close = EXCLUDED.close,
                        adj_close = EXCLUDED.adj_close,
                        volume = EXCLUDED.volume
                """), {
                    'ticker': row.get('ticker'),
                    'date': row.get('date', row.get('index', '')),
                    'open': float(row.get('open', 0) or 0),
                    'high': float(row.get('high', 0) or 0),
                    'low': float(row.get('low', 0) or 0),
                    'close': float(row.get('close', 0) or 0),
                    'adj_close': float(row.get('adj_close', 0) or 0),
                    'volume': int(row.get('volume', 0) or 0),
                })
                rows_inserted += 1
            conn.commit()
        logger.info(f"✅ Stored {rows_inserted} rows for {df['ticker'].iloc[0]}")
        return rows_inserted
    except Exception as e:
        logger.error(f"❌ DB store failed: {e}")
        return 0


# ── Fundamentals Fetcher ──────────────────────────────────────
def fetch_fundamentals(ticker: str) -> Optional[dict]:
    """Fetch fundamental data via yFinance."""
    try:
        t = yf.Ticker(ticker)
        info = t.info
        if not info:
            return None
        return {
            'ticker': ticker,
            'market_cap': info.get('marketCap'),
            'pe_ratio': info.get('trailingPE'),
            'forward_pe': info.get('forwardPE'),
            'pb_ratio': info.get('priceToBook'),
            'ps_ratio': info.get('priceToSalesTrailing12Months'),
            'ev_ebitda': info.get('enterpriseToEbitda'),
            'debt_to_equity': info.get('debtToEquity'),
            'roe': info.get('returnOnEquity'),
            'roa': info.get('returnOnAssets'),
            'profit_margin': info.get('profitMargins'),
            'revenue_growth': info.get('revenueGrowth'),
            'earnings_growth': info.get('earningsGrowth'),
            'beta': info.get('beta'),
            'dividend_yield': info.get('dividendYield'),
            'sector': info.get('sector', ''),
            'industry': info.get('industry', ''),
        }
    except Exception as e:
        logger.error(f"❌ Fundamentals failed for {ticker}: {e}")
        return None


def store_fundamentals(data: dict, engine):
    """Upsert fundamentals."""
    if not data:
        return
    try:
        with engine.connect() as conn:
            conn.execute(text("""
                INSERT INTO fundamentals (
                    ticker, market_cap, pe_ratio, forward_pe, pb_ratio, ps_ratio,
                    ev_ebitda, debt_to_equity, roe, roa, profit_margin,
                    revenue_growth, earnings_growth, beta, dividend_yield, sector, industry
                )
                VALUES (
                    :ticker, :market_cap, :pe_ratio, :forward_pe, :pb_ratio, :ps_ratio,
                    :ev_ebitda, :debt_to_equity, :roe, :roa, :profit_margin,
                    :revenue_growth, :earnings_growth, :beta, :dividend_yield, :sector, :industry
                )
                ON CONFLICT (ticker, fetched_at::date) DO NOTHING
            """), data)
            conn.commit()
        logger.info(f"✅ Fundamentals stored for {data['ticker']}")
    except Exception as e:
        logger.error(f"❌ Fundamentals DB store failed: {e}")


# ── Macro Data (FRED) ─────────────────────────────────────────
FRED_SERIES = {
    'VIXCLS': 'VIX',
    'DGS10': '10Y_Treasury',
    'DGS2': '2Y_Treasury',
    'T10Y2Y': 'Yield_Curve',
    'CPIAUCSL': 'CPI',
    'UNRATE': 'Unemployment',
    'FEDFUNDS': 'Fed_Funds_Rate',
    'DCOILWTICO': 'WTI_Oil',
    'DTWEXBGS': 'DXY',
}


def fetch_macro_data(days: int = 365) -> pd.DataFrame:
    """Fetch macro series from FRED."""
    if not FRED_API_KEY:
        logger.warning("⚠️ FRED_API_KEY not set — skipping macro data")
        return pd.DataFrame()
    try:
        from fredapi import Fred
        fred = Fred(api_key=FRED_API_KEY)
        end = datetime.now()
        start = end - timedelta(days=days)
        all_series = []
        for series_id in FRED_SERIES:
            try:
                data = fred.get_series(series_id, start, end)
                df = data.reset_index()
                df.columns = ['date', 'value']
                df['series_id'] = series_id
                df = df.dropna()
                all_series.append(df)
                logger.info(f"✅ FRED {series_id}: {len(df)} obs")
                time.sleep(0.1)  # Rate limit
            except Exception as e:
                logger.warning(f"⚠️ FRED {series_id} failed: {e}")
        return pd.concat(all_series) if all_series else pd.DataFrame()
    except ImportError:
        logger.error("fredapi not installed. Run: pip install fredapi")
        return pd.DataFrame()


def store_macro_data(df: pd.DataFrame, engine):
    """Upsert macro data."""
    if df.empty:
        return
    try:
        with engine.connect() as conn:
            for _, row in df.iterrows():
                conn.execute(text("""
                    INSERT INTO macro_data (series_id, date, value)
                    VALUES (:series_id, :date, :value)
                    ON CONFLICT (series_id, date) DO UPDATE SET value = EXCLUDED.value
                """), {
                    'series_id': row['series_id'],
                    'date': row['date'],
                    'value': float(row['value']) if row['value'] is not None else None
                })
            conn.commit()
        logger.info(f"✅ Macro data stored: {len(df)} rows")
    except Exception as e:
        logger.error(f"❌ Macro store failed: {e}")


# ── Main Pipeline ─────────────────────────────────────────────
def run_pipeline(tickers: List[str], days: int = 365, include_macro: bool = True):
    """Run full data pipeline for list of tickers."""
    logger.info(f"🚀 Starting data pipeline for {len(tickers)} tickers, {days} days")
    engine = get_engine()
    create_tables(engine)

    # Fetch OHLCV
    total_rows = 0
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Fetching {ticker}...")
        df = fetch_ohlcv(ticker, days)
        if df is not None:
            rows = store_ohlcv(df, engine)
            total_rows += rows
        # Respectful rate limiting
        time.sleep(0.5)

    logger.info(f"✅ OHLCV complete: {total_rows} total rows stored")

    # Fetch Fundamentals (every ticker, weekly cadence)
    logger.info("📊 Fetching fundamentals...")
    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{len(tickers)}] Fundamentals {ticker}...")
        data = fetch_fundamentals(ticker)
        if data:
            store_fundamentals(data, engine)
        time.sleep(1.0)  # yfinance rate limit for info endpoint

    # Fetch Macro Data
    if include_macro:
        logger.info("🌍 Fetching macro data from FRED...")
        macro_df = fetch_macro_data(days)
        if not macro_df.empty:
            store_macro_data(macro_df, engine)

    logger.info("🎉 Pipeline complete!")
    engine.dispose()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantEdge Data Pipeline")
    parser.add_argument("--tickers", nargs="+", help="List of tickers", default=None)
    parser.add_argument("--universe", choices=["sp500_top50", "custom"], default="custom")
    parser.add_argument("--days", type=int, default=365, help="Days of history to fetch")
    parser.add_argument("--no-macro", action="store_true", help="Skip macro data")
    args = parser.parse_args()

    if args.universe == "sp500_top50":
        tickers = SP500_TOP50
    elif args.tickers:
        tickers = args.tickers
    else:
        logger.error("Provide --tickers or --universe")
        sys.exit(1)

    run_pipeline(tickers, days=args.days, include_macro=not args.no_macro)
