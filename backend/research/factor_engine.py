"""
QuantEdge v6.0 — Factor Engine
===============================
For a single ticker, compute 4 orthogonal factor scores (0-100 each):

  1. QUALITY     — fundamental business quality (ROIC, margins, F-score, Z-score)
  2. MOMENTUM    — risk-adjusted 12-1 / 6m / 3m price momentum
  3. ACCUMULATION— OBV divergence, volume surge, Amihud illiquidity
  4. TREND       — MA alignment, Hurst persistence, vol-adjusted returns

These four factors are deliberately chosen to be orthogonal — each captures
a distinct dimension of predictive signal. Combining them gives alpha without
data mining.

Academic anchors:
  Momentum:    Jegadeesh & Titman (1993), Moskowitz et al (2012)
  Quality:     Asness, Frazzini, Pedersen (2019) "Quality Minus Junk"
  Accumulation: Blume, Easley, O'Hara (1994)
  Trend:       Hurst (1951), Moskowitz et al (2012)
"""

import os
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger

from ml.fundamentals.quality_engine import QualityEngine


POLYGON_BASE = "https://api.polygon.io"


@dataclass
class FactorScores:
    """4-factor scorecard for a single ticker."""
    ticker: str
    quality: float = 50.0
    momentum: float = 50.0
    accumulation: float = 50.0
    trend: float = 50.0

    # Raw metrics (for inspection and debugging)
    metrics: Dict[str, float] = field(default_factory=dict)

    # Data quality
    data_quality: str = "unknown"
    price_history_days: int = 0
    error: Optional[str] = None


# ══════════════════════════════════════════════════════════════
# PRICE DATA FETCHER (Polygon aggregates)
# ══════════════════════════════════════════════════════════════
async def fetch_price_history(
    ticker: str,
    api_key: str,
    session: aiohttp.ClientSession,
    days: int = 400,
) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV for trailing `days` days from Polygon /v2/aggs.
    Returns DataFrame with columns: open, high, low, close, volume
    Index: datetime
    """
    end = datetime.utcnow().date()
    start = end - timedelta(days=days + 60)  # extra buffer for weekends/holidays

    url = (f"{POLYGON_BASE}/v2/aggs/ticker/{ticker.upper()}"
           f"/range/1/day/{start.isoformat()}/{end.isoformat()}")
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": api_key}

    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status != 200:
                return None
            data = await resp.json()
    except Exception:
        return None

    results = data.get("results", [])
    if not results or len(results) < 60:
        return None

    df = pd.DataFrame(results)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
    df = df[["datetime", "open", "high", "low", "close", "volume"]].set_index("datetime")
    df = df.sort_index()
    return df


# ══════════════════════════════════════════════════════════════
# FACTOR 1: MOMENTUM (price-based, risk-adjusted)
# ══════════════════════════════════════════════════════════════
def compute_momentum_raw(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute raw momentum metrics. Does NOT yet normalize to 0-100 score —
    that happens universe-wide after all tickers are scored.

    Returns:
      mom_12_1: 12-month return, skipping last 21 days (classic academic momentum)
      mom_6m:   6-month return
      mom_3m:   3-month return
      mom_1m:   1-month return (often negative correlation — mean reversion)
      vol_3m:   3-month annualized volatility
      sharpe_3m: 3m return / 3m vol (risk-adjusted momentum)
    """
    if df is None or len(df) < 252:
        return {}

    close = df["close"]
    if len(close) < 252:
        return {}

    # Classic 12-1 momentum: 12m return skipping last month (mean reversion noise)
    p_12m = close.iloc[-252]
    p_1m = close.iloc[-21]
    p_6m = close.iloc[-126] if len(close) >= 126 else close.iloc[0]
    p_3m = close.iloc[-63]  if len(close) >= 63  else close.iloc[0]
    p_now = close.iloc[-1]

    mom_12_1 = (p_1m / p_12m) - 1 if p_12m > 0 else 0.0
    mom_6m   = (p_now / p_6m)  - 1 if p_6m  > 0 else 0.0
    mom_3m   = (p_now / p_3m)  - 1 if p_3m  > 0 else 0.0
    mom_1m   = (p_now / p_1m)  - 1 if p_1m  > 0 else 0.0

    returns = close.pct_change().dropna()
    vol_3m  = float(returns.tail(63).std() * np.sqrt(252))
    sharpe_3m = float(mom_3m / vol_3m) if vol_3m > 0 else 0.0

    return {
        "mom_12_1":  float(mom_12_1),
        "mom_6m":    float(mom_6m),
        "mom_3m":    float(mom_3m),
        "mom_1m":    float(mom_1m),
        "vol_3m":    vol_3m,
        "sharpe_3m": sharpe_3m,
    }


# ══════════════════════════════════════════════════════════════
# FACTOR 2: ACCUMULATION (volume-based, institutional footprint proxy)
# ══════════════════════════════════════════════════════════════
def compute_accumulation_raw(df: pd.DataFrame) -> Dict[str, float]:
    """
    Volume/price relationship metrics. The "smart money" footprint proxy.

    Returns:
      obv_slope:          On-Balance Volume slope (last 63 days, normalized)
      price_vol_divergence: Price slope vs OBV slope — divergence is bullish
      volume_surge_ratio: Last 20d vol / trailing 90d vol
      amihud_illiquidity: |return| / $volume — lower = more liquid = more institutional
      ad_line_slope:      Accumulation/Distribution line slope (last 63 days)
    """
    if df is None or len(df) < 90:
        return {}

    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    returns = close.pct_change().fillna(0.0)

    # ── OBV (On-Balance Volume) ──
    # +volume on up days, -volume on down days, cumulative
    sign = np.sign(returns).replace(0, 0)
    obv = (volume * sign).cumsum()
    obv_recent = obv.tail(63)
    if len(obv_recent) >= 20 and obv_recent.std() > 0:
        x = np.arange(len(obv_recent), dtype=np.float64)
        y = obv_recent.values.astype(np.float64)
        obv_slope = float(np.polyfit(x, y, 1)[0]) / float(abs(obv_recent.mean()) + 1e-9)
    else:
        obv_slope = 0.0

    # ── Price vs OBV divergence ──
    # If price is flat but OBV rising → accumulation
    close_recent = close.tail(63)
    if len(close_recent) >= 20 and close_recent.std() > 0:
        x = np.arange(len(close_recent), dtype=np.float64)
        price_slope = float(np.polyfit(x, close_recent.values, 1)[0]) / float(close_recent.mean() + 1e-9)
    else:
        price_slope = 0.0

    # Divergence: OBV rising faster than price → bullish accumulation
    price_vol_divergence = obv_slope - price_slope

    # ── Volume surge ──
    vol_20 = float(volume.tail(20).mean())
    vol_90 = float(volume.tail(90).mean())
    volume_surge = vol_20 / vol_90 if vol_90 > 0 else 1.0

    # ── Amihud illiquidity ──
    # |return| / dollar-volume. Lower = more institutional interest.
    dollar_vol = (close * volume).tail(60)
    abs_ret = returns.abs().tail(60)
    if dollar_vol.mean() > 0:
        amihud = float((abs_ret / (dollar_vol + 1)).mean()) * 1e9
    else:
        amihud = 100.0

    # ── Accumulation/Distribution line ──
    # Close relative to daily range, weighted by volume
    hl_range = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / hl_range  # money flow multiplier
    mfv = (mfm.fillna(0) * volume).cumsum()  # money flow volume
    mfv_recent = mfv.tail(63)
    if len(mfv_recent) >= 20 and mfv_recent.std() > 0:
        x = np.arange(len(mfv_recent), dtype=np.float64)
        y = mfv_recent.values.astype(np.float64)
        ad_slope = float(np.polyfit(x, y, 1)[0]) / float(abs(mfv_recent.mean()) + 1e-9)
    else:
        ad_slope = 0.0

    return {
        "obv_slope":            obv_slope,
        "price_slope":          price_slope,
        "price_vol_divergence": price_vol_divergence,
        "volume_surge_ratio":   volume_surge,
        "amihud_illiquidity":   amihud,
        "ad_line_slope":        ad_slope,
    }


# ══════════════════════════════════════════════════════════════
# FACTOR 3: TREND QUALITY (healthy vs exhausted trend)
# ══════════════════════════════════════════════════════════════
def _hurst_exponent(series: pd.Series, min_lag: int = 2, max_lag: int = 50) -> float:
    """
    Rescaled range analysis for Hurst exponent.
    H > 0.5 = trending, H < 0.5 = mean-reverting, H = 0.5 = random walk.
    """
    if len(series) < max_lag * 2:
        return 0.5
    vals = series.values.astype(np.float64)
    lags = range(min_lag, min(max_lag, len(vals) // 2))
    tau = []
    for lag in lags:
        if lag < 2: continue
        diffs = vals[lag:] - vals[:-lag]
        if len(diffs) < 2: continue
        std = np.std(diffs)
        if std > 0:
            tau.append(std)
    if len(tau) < 3:
        return 0.5
    log_lags = np.log(list(lags)[:len(tau)])
    log_tau = np.log(tau)
    try:
        slope = float(np.polyfit(log_lags, log_tau, 1)[0])
        return float(np.clip(slope, 0.0, 1.0))
    except Exception:
        return 0.5


def compute_trend_raw(df: pd.DataFrame) -> Dict[str, float]:
    """
    Trend quality: is the existing trend healthy and sustainable?

    Returns:
      pct_above_ma50:   % distance from 50-day MA
      pct_above_ma200:  % distance from 200-day MA
      ma_alignment:     1.0 if price > MA50 > MA200 (golden structure), -1.0 if reverse
      hurst:            Hurst exponent — trend persistence
      vol_adj_return:   12m return / annualized vol (Sharpe-like)
    """
    if df is None or len(df) < 200:
        return {}

    close = df["close"]
    returns = close.pct_change().dropna()

    ma50 = float(close.tail(50).mean())
    ma200 = float(close.tail(200).mean())
    price = float(close.iloc[-1])

    pct_above_ma50 = (price / ma50) - 1 if ma50 > 0 else 0.0
    pct_above_ma200 = (price / ma200) - 1 if ma200 > 0 else 0.0

    # Alignment: +1 if price > MA50 > MA200 (bullish structure),
    # -1 if price < MA50 < MA200 (bearish structure), 0 otherwise
    if price > ma50 > ma200:
        alignment = 1.0
    elif price < ma50 < ma200:
        alignment = -1.0
    else:
        alignment = 0.0

    hurst = _hurst_exponent(close.tail(252))

    vol_annual = float(returns.tail(252).std() * np.sqrt(252)) if len(returns) >= 252 else 0.0
    ret_12m = float((close.iloc[-1] / close.iloc[-252]) - 1) if len(close) >= 252 else 0.0
    vol_adj_ret = ret_12m / vol_annual if vol_annual > 0 else 0.0

    return {
        "pct_above_ma50":  pct_above_ma50,
        "pct_above_ma200": pct_above_ma200,
        "ma_alignment":    alignment,
        "hurst":           hurst,
        "vol_adj_return":  vol_adj_ret,
    }


# ══════════════════════════════════════════════════════════════
# SCORING: metric → 0-100 score (ticker-independent calibration)
# ══════════════════════════════════════════════════════════════
# These are ABSOLUTE calibration thresholds used for a first-pass score.
# Final scoring will be universe-relative (percentile rank) — but each ticker
# gets a scored value first, so a ticker can be analyzed standalone.

def _linear_score(value: float, low: float, high: float) -> float:
    """Map value to 0-100 linearly. value<=low → 0, value>=high → 100."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 50.0
    if high == low:
        return 50.0
    frac = (value - low) / (high - low)
    return float(np.clip(frac * 100, 0, 100))


def score_momentum(m: Dict[str, float]) -> float:
    """Composite momentum score. Weights on 12-1, 6m, 3m, sharpe."""
    if not m:
        return 50.0
    s = (
        0.40 * _linear_score(m.get("mom_12_1", 0), low=-0.20, high=0.50) +
        0.25 * _linear_score(m.get("mom_6m", 0),   low=-0.15, high=0.30) +
        0.20 * _linear_score(m.get("mom_3m", 0),   low=-0.10, high=0.20) +
        0.15 * _linear_score(m.get("sharpe_3m", 0), low=-1.0,  high=2.0)
    )
    return float(np.clip(s, 0, 100))


def score_accumulation(m: Dict[str, float]) -> float:
    """Composite accumulation score."""
    if not m:
        return 50.0
    s = (
        0.30 * _linear_score(m.get("obv_slope", 0),         low=-0.02, high=0.02) +
        0.25 * _linear_score(m.get("price_vol_divergence", 0), low=-0.01, high=0.01) +
        0.20 * _linear_score(m.get("volume_surge_ratio", 1),  low=0.7, high=1.8) +
        0.15 * _linear_score(m.get("ad_line_slope", 0),      low=-0.02, high=0.02) +
        # Lower amihud = more liquid = better; invert
        0.10 * (100 - _linear_score(m.get("amihud_illiquidity", 10), low=0.1, high=10.0))
    )
    return float(np.clip(s, 0, 100))


def score_trend(m: Dict[str, float]) -> float:
    """Composite trend quality score."""
    if not m:
        return 50.0
    alignment = m.get("ma_alignment", 0)
    alignment_score = 100.0 if alignment > 0 else 0.0 if alignment < 0 else 50.0
    s = (
        0.25 * _linear_score(m.get("pct_above_ma50", 0),  low=-0.10, high=0.15) +
        0.25 * _linear_score(m.get("pct_above_ma200", 0), low=-0.20, high=0.30) +
        0.20 * alignment_score +
        0.15 * _linear_score(m.get("hurst", 0.5),         low=0.40, high=0.65) +
        0.15 * _linear_score(m.get("vol_adj_return", 0),  low=-0.5, high=2.0)
    )
    return float(np.clip(s, 0, 100))


# ══════════════════════════════════════════════════════════════
# FACTOR ENGINE (main class)
# ══════════════════════════════════════════════════════════════
class FactorEngine:
    """
    Top-level entry. For a ticker, produce a 4-factor scorecard.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        self.quality_engine = QualityEngine(api_key=self.api_key)

    async def score(
        self,
        ticker: str,
        session: Optional[aiohttp.ClientSession] = None,
        skip_quality: bool = False,
    ) -> FactorScores:
        """
        Compute all 4 factor scores for a single ticker.
        If skip_quality=True, quality score remains at default 50.0
        (useful when doing a fast price-only scan).
        """
        close_session = False
        if session is None:
            session = aiohttp.ClientSession()
            close_session = True

        try:
            # Price-based factors require OHLCV history
            df = await fetch_price_history(ticker, self.api_key, session, days=400)
            if df is None or len(df) < 200:
                return FactorScores(
                    ticker=ticker,
                    data_quality="insufficient",
                    error="Insufficient price history (<200 days)",
                )

            mom_raw   = compute_momentum_raw(df)
            acc_raw   = compute_accumulation_raw(df)
            trend_raw = compute_trend_raw(df)

            momentum_score     = score_momentum(mom_raw)
            accumulation_score = score_accumulation(acc_raw)
            trend_score        = score_trend(trend_raw)

            # Quality factor (optional — slow, requires fundamentals fetch)
            quality_score = 50.0
            quality_metrics = {}
            if not skip_quality:
                try:
                    q = await self.quality_engine.analyze(ticker, n_quarters=40)
                    quality_score = q.past_score
                    quality_metrics = {
                        "quality_piotroski": q.piotroski_f_score,
                        "quality_altman_z": q.altman_z_score,
                        "quality_data_q":   q.data_quality,
                    }
                except Exception as e:
                    logger.debug(f"Quality fetch failed for {ticker}: {e}")

            return FactorScores(
                ticker=ticker,
                quality=round(quality_score, 1),
                momentum=round(momentum_score, 1),
                accumulation=round(accumulation_score, 1),
                trend=round(trend_score, 1),
                metrics={
                    # Momentum
                    "mom_12_1":  round(mom_raw.get("mom_12_1", 0) * 100, 2),
                    "mom_6m":    round(mom_raw.get("mom_6m",   0) * 100, 2),
                    "mom_3m":    round(mom_raw.get("mom_3m",   0) * 100, 2),
                    "mom_1m":    round(mom_raw.get("mom_1m",   0) * 100, 2),
                    "sharpe_3m": round(mom_raw.get("sharpe_3m", 0), 3),
                    # Accumulation
                    "obv_slope_norm":    round(acc_raw.get("obv_slope", 0), 6),
                    "volume_surge":      round(acc_raw.get("volume_surge_ratio", 0), 3),
                    "amihud":            round(acc_raw.get("amihud_illiquidity", 0), 3),
                    # Trend
                    "pct_above_ma50":    round(trend_raw.get("pct_above_ma50", 0) * 100, 2),
                    "pct_above_ma200":   round(trend_raw.get("pct_above_ma200", 0) * 100, 2),
                    "ma_alignment":      trend_raw.get("ma_alignment", 0),
                    "hurst":             round(trend_raw.get("hurst", 0.5), 3),
                    "vol_adj_return":    round(trend_raw.get("vol_adj_return", 0), 3),
                    # Quality extras
                    **quality_metrics,
                },
                data_quality="good",
                price_history_days=len(df),
            )
        except Exception as e:
            logger.warning(f"FactorEngine failed for {ticker}: {e}")
            return FactorScores(ticker=ticker, data_quality="error", error=str(e))
        finally:
            if close_session:
                await session.close()


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════
async def _test_single(ticker: str):
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set"); return

    engine = FactorEngine(api_key=api_key)
    print(f"\nScoring {ticker}...")
    fs = await engine.score(ticker)

    print(f"\n{'='*60}")
    print(f"  FACTOR SCORECARD: {fs.ticker}")
    print(f"{'='*60}")
    print(f"  Data quality: {fs.data_quality} ({fs.price_history_days} days)")
    if fs.error:
        print(f"  ERROR: {fs.error}")
        return

    def bar(v: float) -> str:
        return "#" * int(v / 5)

    print(f"\n  Factor scores:")
    print(f"    QUALITY      {fs.quality:>6.1f}  {bar(fs.quality)}")
    print(f"    MOMENTUM     {fs.momentum:>6.1f}  {bar(fs.momentum)}")
    print(f"    ACCUMULATION {fs.accumulation:>6.1f}  {bar(fs.accumulation)}")
    print(f"    TREND        {fs.trend:>6.1f}  {bar(fs.trend)}")

    print(f"\n  Raw metrics:")
    for k, v in fs.metrics.items():
        print(f"    {k:<22} {v}")


async def _test_batch():
    """Test on a small basket to verify calibration across different stocks."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    tickers = ["AAPL", "NVDA", "META", "KO", "F", "INTC", "WBA", "PLTR"]

    engine = FactorEngine(api_key=api_key)
    async with aiohttp.ClientSession() as session:
        tasks = [engine.score(t, session=session) for t in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=False)

    print(f"\n{'='*80}")
    print(f"  BATCH FACTOR COMPARISON")
    print(f"{'='*80}")
    print(f"  {'Ticker':<8} {'Quality':>8} {'Momentum':>9} {'Accum':>8} {'Trend':>7}")
    print(f"  {'-'*8} {'-'*8} {'-'*9} {'-'*8} {'-'*7}")
    for fs in results:
        if fs.error:
            print(f"  {fs.ticker:<8} ERROR: {fs.error}")
        else:
            print(f"  {fs.ticker:<8} {fs.quality:>8.1f} {fs.momentum:>9.1f} "
                  f"{fs.accumulation:>8.1f} {fs.trend:>7.1f}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "batch":
        asyncio.run(_test_batch())
    else:
        ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
        asyncio.run(_test_single(ticker))
