"""Universe price store — local SQLite of daily close+volume for EVERY US stock.

Why this exists (step-12): REBOUND must ask "how far is every stock below its
3-year high?" — a question about 5,000 tickers x 750 days. Per-ticker API
calls cannot answer it nightly; a local store answers it in milliseconds.
One Polygon grouped-daily call returns the ENTIRE market for one day, so:
  backfill  = ~756 calls, once (gentle pacing, resumable)
  nightly   = 1 call, appended by the scan job
The same store feeds the PIT backtest (step 21), stage detection (step 19),
volume confirmation (step 15) and the cross-sectional model (stage F).

Size: ~12k tickers x 756 days ~= 9M rows ~= 400MB SQLite. Disk is at 62%.

PIT note: bars are immutable history keyed by calendar date — a query
`series(t, start, end)` with end <= as_of can never see the future.
"""
from __future__ import annotations
import os, sqlite3, time, json, urllib.request
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

DEFAULT_PATH = os.environ.get(
    "PRICE_STORE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "price_store.db"),
)


def _poly_key() -> str:
    k = os.environ.get("POLYGON_API_KEY") or os.environ.get("POLYGON")
    if not k:
        raise RuntimeError("No POLYGON key in environment")
    return k


class PriceStore:
    def __init__(self, path: str = DEFAULT_PATH):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.path = path
        self._con = sqlite3.connect(path)
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS bars ("
            " d TEXT NOT NULL, t TEXT NOT NULL, c REAL NOT NULL, v REAL,"
            " PRIMARY KEY (d, t)) WITHOUT ROWID"
        )
        self._con.execute("CREATE INDEX IF NOT EXISTS idx_bars_t ON bars(t, d)")
        # days we asked Polygon about — including holidays that returned empty,
        # so a resume never refetches them
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS fetched_days (d TEXT PRIMARY KEY, n INTEGER)"
        )
        self._con.commit()

    # ── ingest ────────────────────────────────────────────────
    def _fetch_grouped(self, day: date) -> List[Tuple[str, float, float]]:
        url = (
            "https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/"
            f"{day.isoformat()}?adjusted=true&apiKey={_poly_key()}"
        )
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        return [
            (row["T"], float(row["c"]), float(row.get("v") or 0.0))
            for row in d.get("results", [])
            if row.get("c")
        ]

    def ingest_day(self, day: date) -> int:
        """Fetch one grouped-daily bar set and store it. Returns rows stored."""
        rows = self._fetch_grouped(day)
        ds = day.isoformat()
        if rows:
            self._con.executemany(
                "INSERT OR REPLACE INTO bars (d, t, c, v) VALUES (?,?,?,?)",
                [(ds, t, c, v) for t, c, v in rows],
            )
        self._con.execute(
            "INSERT OR REPLACE INTO fetched_days (d, n) VALUES (?,?)", (ds, len(rows))
        )
        self._con.commit()
        return len(rows)

    def backfill(self, years: int = 3, pause: float = 0.35, verbose: bool = True) -> int:
        """Fill every weekday of the last `years` years not yet fetched.
        Resumable: rerunning skips completed days. ~756 calls for 3 years."""
        done = {r[0] for r in self._con.execute("SELECT d FROM fetched_days")}
        day = date.today() - timedelta(days=1)
        start = day - timedelta(days=int(years * 365.25))
        total = 0
        d = start
        while d <= day:
            if d.weekday() < 5 and d.isoformat() not in done:
                try:
                    n = self.ingest_day(d)
                    total += n
                    if verbose and n:
                        print(f"{d} {n} tickers", flush=True)
                except urllib.error.HTTPError as e:
                    if e.code == 403:
                        # plan-entitlement floor: permanent for this day —
                        # record it so future backfills never re-crawl it
                        self._con.execute(
                            "INSERT OR REPLACE INTO fetched_days (d, n) VALUES (?, 0)",
                            (d.isoformat(),))
                        self._con.commit()
                        if verbose:
                            print(f"{d} 403 plan floor — marked, won't retry", flush=True)
                    else:
                        print(f"{d} HTTP {e.code} — will retry on next run", flush=True)
                except Exception as e:
                    print(f"{d} ERROR {e} — will retry on next run", flush=True)
                time.sleep(pause)
            d += timedelta(days=1)
        return total

    def append_latest(self, lookback_days: int = 7) -> int:
        """Nightly top-up: fetch any recent weekdays not yet stored."""
        done = {r[0] for r in self._con.execute("SELECT d FROM fetched_days")}
        total = 0
        for back in range(lookback_days, 0, -1):
            d = date.today() - timedelta(days=back)
            if d.weekday() < 5 and d.isoformat() not in done:
                try:
                    total += self.ingest_day(d)
                    time.sleep(0.35)
                except Exception:
                    pass
        return total

    # ── read ──────────────────────────────────────────────────
    def series(
        self, ticker: str, start: date, end: date
    ) -> List[Tuple[date, float, float]]:
        """[(date, close, volume)] ascending, start<=d<=end. Pass end<=as_of
        and the result is point-in-time by construction."""
        cur = self._con.execute(
            "SELECT d, c, v FROM bars WHERE t=? AND d>=? AND d<=? ORDER BY d",
            (ticker.upper(), start.isoformat(), end.isoformat()),
        )
        return [(date.fromisoformat(d), c, v or 0.0) for d, c, v in cur]

    def closes_on(self, day: date) -> Dict[str, float]:
        cur = self._con.execute("SELECT t, c FROM bars WHERE d=?", (day.isoformat(),))
        return dict(cur.fetchall())

    def last_day(self) -> Optional[date]:
        row = self._con.execute("SELECT MAX(d) FROM bars").fetchone()
        return date.fromisoformat(row[0]) if row and row[0] else None

    def coverage(self) -> dict:
        n_days = self._con.execute(
            "SELECT COUNT(*) FROM fetched_days WHERE n > 0"
        ).fetchone()[0]
        n_rows = self._con.execute("SELECT COUNT(*) FROM bars").fetchone()[0]
        lo, hi = self._con.execute("SELECT MIN(d), MAX(d) FROM bars").fetchone()
        return {"trading_days": n_days, "rows": n_rows, "from": lo, "to": hi}

    def close(self):
        self._con.close()
