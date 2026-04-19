"""
QuantEdge v6.0 — Historical Data Fetcher & Point-in-Time Cache
==============================================================
Fetches 5 years of daily OHLCV for the universe, caches to local
parquet files, and provides point-in-time query methods.

CRITICAL: All queries must respect the as-of date to avoid look-ahead
bias. `get_prices(ticker, end_date)` returns data ONLY up to and
including end_date — never future data.
"""

import os
import asyncio
import aiohttp
import pickle
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from loguru import logger


POLYGON_BASE = "https://api.polygon.io"

# Cache directory — survives across runs. Persists as pickled pandas DataFrames.
CACHE_DIR = Path.home() / ".quantedge_cache" / "historical"


@dataclass
class CacheStats:
    total_tickers: int
    cached_tickers: int
    fresh_fetched: int
    failed: int
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None


def _cache_path(ticker: str) -> Path:
    """Return cache file path for a ticker."""
    return CACHE_DIR / f"{ticker.upper()}.pkl"


def _ensure_cache_dir():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_is_fresh(path: Path, max_age_days: int = 1) -> bool:
    """Check if cache file is fresh enough to skip re-fetch."""
    if not path.exists():
        return False
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    age = datetime.now() - mtime
    return age.days < max_age_days


async def _fetch_one_ticker(
    ticker: str,
    api_key: str,
    session: aiohttp.ClientSession,
    start: date,
    end: date,
) -> Optional[pd.DataFrame]:
    """Fetch raw aggregates for one ticker."""
    url = (f"{POLYGON_BASE}/v2/aggs/ticker/{ticker.upper()}"
           f"/range/1/day/{start.isoformat()}/{end.isoformat()}")
    params = {
        "adjusted": "true",
        "sort": "asc",
        "limit": 50000,
        "apiKey": api_key,
    }

    for attempt in range(3):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status == 429:
                    # Rate limited — exponential backoff
                    wait = 2 ** attempt
                    logger.debug(f"{ticker}: rate-limited (429), waiting {wait}s before retry {attempt+1}")
                    await asyncio.sleep(wait)
                    continue
                if resp.status != 200:
                    logger.warning(f"{ticker}: HTTP {resp.status} — {await resp.text()}")
                    return None
                data = await resp.json()
                break
        except asyncio.TimeoutError:
            logger.warning(f"{ticker}: timeout on attempt {attempt+1}")
            if attempt == 2:
                return None
            await asyncio.sleep(1)
        except Exception as e:
            logger.warning(f"{ticker}: {type(e).__name__}: {e}")
            return None
    else:
        return None

    results = data.get("results", [])
    if len(results) < 100:
        return None

    df = pd.DataFrame(results)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms")
    df = df.rename(columns={
        "o": "open", "h": "high", "l": "low",
        "c": "close", "v": "volume", "vw": "vwap", "n": "trades",
    })
    keep = ["datetime", "open", "high", "low", "close", "volume"]
    if "vwap" in df.columns: keep.append("vwap")
    df = df[keep].set_index("datetime").sort_index()
    return df


class HistoricalDataStore:
    """
    Point-in-time historical price store.

    Usage:
        store = HistoricalDataStore()
        await store.populate(tickers, years=5)    # one-time
        df = store.get_prices("AAPL", end_date=date(2023, 6, 1))
        # df is AAPL OHLCV from April 2021 through June 1, 2023 — no future data
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        _ensure_cache_dir()
        # Lazy in-memory cache: load from disk on first access per ticker
        self._memory: Dict[str, pd.DataFrame] = {}

    def _load_from_disk(self, ticker: str) -> Optional[pd.DataFrame]:
        path = _cache_path(ticker)
        if not path.exists():
            return None
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Cache load failed for {ticker}: {e}")
            return None

    def _save_to_disk(self, ticker: str, df: pd.DataFrame):
        path = _cache_path(ticker)
        try:
            with open(path, "wb") as f:
                pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            logger.warning(f"Cache save failed for {ticker}: {e}")

    async def populate(
        self,
        tickers: List[str],
        years: int = 5,
        concurrency: int = 3,
        force_refresh: bool = False,
    ) -> CacheStats:
        """
        Download historical data for all tickers. Cached on disk.
        Skips tickers already in cache unless force_refresh=True.
        """
        if not self.api_key:
            raise RuntimeError("POLYGON_API_KEY not set")

        end = date.today()
        start = end - timedelta(days=years * 365 + 30)

        stats = CacheStats(total_tickers=len(tickers), cached_tickers=0, fresh_fetched=0, failed=0)
        to_fetch: List[str] = []

        for t in tickers:
            path = _cache_path(t)
            if not force_refresh and _cache_is_fresh(path):
                stats.cached_tickers += 1
            else:
                to_fetch.append(t)

        logger.info(f"Historical data: {stats.cached_tickers} cached, {len(to_fetch)} to fetch")

        if not to_fetch:
            stats.date_range_start = start
            stats.date_range_end = end
            return stats

        sem = asyncio.Semaphore(concurrency)
        connector = aiohttp.TCPConnector(limit=10, force_close=False, enable_cleanup_closed=True)

        async def _fetch_and_cache(ticker: str, session: aiohttp.ClientSession) -> Tuple[str, bool]:
            async with sem:
                df = await _fetch_one_ticker(ticker, self.api_key, session, start, end)
                if df is None or len(df) < 100:
                    return ticker, False
                self._save_to_disk(ticker, df)
                return ticker, True

        # CRITICAL: ONE session shared across all tasks. Creating a
        # session per task causes SSL state corruption under concurrency.
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [_fetch_and_cache(t, session) for t in to_fetch]
            results = await asyncio.gather(*tasks, return_exceptions=False)

        for t, ok in results:
            if ok:
                stats.fresh_fetched += 1
            else:
                stats.failed += 1

        stats.date_range_start = start
        stats.date_range_end = end
        logger.info(
            f"Populate done: {stats.fresh_fetched} fetched, "
            f"{stats.failed} failed, {stats.cached_tickers} already cached"
        )
        return stats

    def get_prices(
        self,
        ticker: str,
        end_date: Optional[date] = None,
        start_date: Optional[date] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Point-in-time price query.
        Returns OHLCV for ticker through end_date (inclusive).
        NEVER returns data after end_date — this is the look-ahead safety.
        """
        if ticker not in self._memory:
            df = self._load_from_disk(ticker)
            if df is None:
                return None
            self._memory[ticker] = df

        df = self._memory[ticker]

        # Apply end_date cut (critical for no look-ahead)
        if end_date is not None:
            end_ts = pd.Timestamp(end_date) + pd.Timedelta(hours=23, minutes=59)
            df = df.loc[df.index <= end_ts]

        if start_date is not None:
            start_ts = pd.Timestamp(start_date)
            df = df.loc[df.index >= start_ts]

        return df.copy() if len(df) > 0 else None

    def available_tickers(self) -> List[str]:
        """Return list of tickers currently cached on disk."""
        if not CACHE_DIR.exists():
            return []
        return sorted([p.stem.upper() for p in CACHE_DIR.glob("*.pkl")])

    def cache_size_mb(self) -> float:
        if not CACHE_DIR.exists():
            return 0.0
        total = sum(p.stat().st_size for p in CACHE_DIR.glob("*.pkl"))
        return total / (1024 * 1024)

    def get_price_at(self, ticker: str, as_of: date, column: str = "close") -> Optional[float]:
        """Get the close (or other column) for the most recent trading day on/before as_of."""
        df = self.get_prices(ticker, end_date=as_of)
        if df is None or len(df) == 0:
            return None
        return float(df[column].iloc[-1])

    def get_returns(
        self,
        ticker: str,
        end_date: Optional[date] = None,
        lookback_days: int = 252,
    ) -> Optional[pd.Series]:
        """Daily returns series up to end_date."""
        df = self.get_prices(ticker, end_date=end_date)
        if df is None or len(df) < lookback_days // 2:
            return None
        returns = df["close"].pct_change().dropna().tail(lookback_days)
        return returns


# ══════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════
async def _test_populate():
    """Quick populate test with 10 tickers to verify the pipeline."""
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set"); return

    test_tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMZN", "JPM", "V", "SPY"]

    store = HistoricalDataStore(api_key=api_key)
    print(f"\nPopulating {len(test_tickers)} tickers (5 years)...")
    stats = await store.populate(test_tickers, years=5)

    print(f"\n{'='*60}")
    print(f"  HISTORICAL DATA POPULATE")
    print(f"{'='*60}")
    print(f"  Total tickers:    {stats.total_tickers}")
    print(f"  Already cached:   {stats.cached_tickers}")
    print(f"  Fresh fetched:    {stats.fresh_fetched}")
    print(f"  Failed:           {stats.failed}")
    print(f"  Date range:       {stats.date_range_start} → {stats.date_range_end}")
    print(f"  Cache size:       {store.cache_size_mb():.2f} MB")

    # Verify point-in-time behavior
    print(f"\n  Point-in-time test (AAPL):")
    full = store.get_prices("AAPL")
    if full is not None:
        print(f"    Full history: {full.index[0].date()} → {full.index[-1].date()} ({len(full)} days)")

    past_cut = date(2023, 1, 15)
    partial = store.get_prices("AAPL", end_date=past_cut)
    if partial is not None:
        print(f"    As of {past_cut}: {partial.index[0].date()} → {partial.index[-1].date()} "
              f"({len(partial)} days)")
        assert partial.index[-1].date() <= past_cut, "LOOK-AHEAD BUG"
        print(f"    ✅ No look-ahead: last date {partial.index[-1].date()} ≤ {past_cut}")


async def _test_full_universe():
    """Populate the full 507-ticker universe."""
    from research.universe import UniverseBuilder
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY not set"); return

    builder = UniverseBuilder(api_key=api_key)
    universe = await builder.build()
    tickers = [u.ticker for u in universe]

    print(f"\nPopulating {len(tickers)} tickers (5 years)...")
    print("This takes 1-2 minutes on first run.\n")

    store = HistoricalDataStore(api_key=api_key)
    stats = await store.populate(tickers, years=5, concurrency=3)

    print(f"\n{'='*60}")
    print(f"  FULL UNIVERSE HISTORICAL POPULATE")
    print(f"{'='*60}")
    print(f"  Tickers total:    {stats.total_tickers}")
    print(f"  Cached already:   {stats.cached_tickers}")
    print(f"  Fresh fetched:    {stats.fresh_fetched}")
    print(f"  Failed:           {stats.failed}")
    print(f"  Cache on disk:    {store.cache_size_mb():.1f} MB")
    print(f"  Success rate:     {(stats.fresh_fetched + stats.cached_tickers) / stats.total_tickers * 100:.1f}%")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        asyncio.run(_test_full_universe())
    else:
        asyncio.run(_test_populate())
