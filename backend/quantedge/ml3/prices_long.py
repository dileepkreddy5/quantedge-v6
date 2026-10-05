"""ML v3, phase A — 25 years of daily total-return prices (split- and dividend-adjusted) straight from Yahoo's
chart endpoint (the yfinance version installed returns empty data). Gentle (3 at a time, back-off on 429) and
resumable (saved every 150 tickers). Caveat: companies delisted before today aren't included (survivorship bias)."""
from __future__ import annotations
import asyncio, glob, json, os, time
import pandas as pd
from loguru import logger

OUT = "/app/models/panel_v3"
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"}


async def download_long(pool, start=946684800):
    import httpx
    os.makedirs(f"{OUT}/parts", exist_ok=True)
    rows = await pool.fetch("""SELECT ticker FROM company_facts WHERE as_of=(SELECT max(as_of) FROM company_facts)
                               AND primary_listing AND NOT is_spac AND tier IN ('large','mid','small')""")
    tks = sorted(r["ticker"] for r in rows) + ["SPY"]
    done = set()
    for f in glob.glob(f"{OUT}/parts/*.parquet"): done |= set(pd.read_parquet(f, columns=["ticker"])["ticker"].unique())
    todo = [t for t in tks if t not in done]; failed = []; buf = []; strikes = 0
    logger.info(f"[ml3] {len(tks)} tickers, {len(done)} already downloaded, {len(todo)} to go (serial, ~2s apart)")
    async with httpx.AsyncClient(timeout=30, headers={"User-Agent": "Mozilla/5.0"}) as cx:
        for n, t in enumerate(todo, 1):
            df = None
            for a in range(3):
                try:
                    r = await cx.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{t.replace('.', '-')}",
                                     params={"period1": start, "period2": int(time.time()), "interval": "1d"})
                except Exception:
                    r = None
                if r is not None and r.status_code == 429:
                    strikes += 1; logger.warning(f"[ml3] rate-limited at {t} (strike {strikes}) — pausing 10 min"); await asyncio.sleep(600); continue
                strikes = 0
                if r is None or r.status_code != 200: break
                try:
                    res = r.json()["chart"]["result"][0]; ind = res["indicators"]
                    d = pd.to_datetime(res["timestamp"], unit="s", utc=True).tz_convert("America/New_York").date
                    df = pd.DataFrame({"ticker": t, "d": d, "c": ind["adjclose"][0]["adjclose"], "cs": ind["quote"][0]["close"], "v": ind["quote"][0]["volume"]}).dropna(subset=["c"])
                    if len(df) < 60: df = None
                except Exception:
                    df = None
                break
            if strikes >= 6:
                logger.error("[ml3] still rate-limited after an hour of pauses — stopping; rerun later (it resumes)"); break
            if df is None: failed.append(t)
            else: buf.append(df)
            if len(buf) >= 100 or (n == len(todo) and buf):
                pd.concat(buf, ignore_index=True).astype({"c": "float32", "cs": "float32"}).to_parquet(f"{OUT}/parts/p{int(time.time())}.parquet", index=False); buf = []
                logger.info(f"[ml3] prices {len(done) + n}/{len(tks)} · failed {len(failed)}")
            await asyncio.sleep(2.0)
    if buf: pd.concat(buf, ignore_index=True).astype({"c": "float32", "cs": "float32"}).to_parquet(f"{OUT}/parts/p{int(time.time())}.parquet", index=False)
    P = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{OUT}/parts/*.parquet"))], ignore_index=True)
    P.to_parquet(f"{OUT}/prices_long.parquet", index=False)
    first = P.groupby("ticker")["d"].min()
    summ = {"tickers": int(P["ticker"].nunique()), "failed": len(failed), "rows": len(P),
            "with_data_before_2005": int((first < pd.Timestamp("2005-01-01").date()).sum()),
            "with_data_before_2012": int((first < pd.Timestamp("2012-01-01").date()).sum()), "failed_sample": failed[:20]}
    json.dump(summ, open(f"{OUT}/prices_long_summary.json", "w"), indent=1)
    logger.info(f"[ml3] long price history: {summ}")
    return summ
