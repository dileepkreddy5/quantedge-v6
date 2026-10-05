"""ML v3 — SEC event history for two research factors.
1) Results dates: every 8-K with item 2.02 since 2004, from each company's submissions index (no documents
   fetched), with the acceptance time in New York so after-close results map to the next day's reaction.
2) Insider trades: the SEC's quarterly Insider Transactions Data Sets (2006+), open-market purchases (P) and
   sales (S) in our universe, with role and the pre-planned (10b5-1) flag where reported."""
from __future__ import annotations
import asyncio, io, json, os, zipfile, datetime as dt
import pandas as pd
from loguru import logger

OUT = "/app/models/panel_v3"
UA = {"User-Agent": "QuantEdge Research dileepkreddy5@gmail.com"}


async def build_results_dates(pool, since="2004-01-01"):
    import httpx
    rows = await pool.fetch("""SELECT f.ticker, u.cik FROM company_facts f JOIN universe u USING (ticker)
        WHERE f.as_of=(SELECT max(as_of) FROM company_facts) AND f.primary_listing AND NOT f.is_spac AND f.tier IN ('large','mid','small') AND u.cik IS NOT NULL""")
    out, sem, failed = [], asyncio.Semaphore(4), 0
    async with httpx.AsyncClient(timeout=30, headers=UA) as cx:
        async def get(url):
            for a in range(4):
                async with sem:
                    try: r = await cx.get(url)
                    except Exception: r = None
                    await asyncio.sleep(0.15)
                if r is not None and r.status_code == 200: return r.json()
                await asyncio.sleep(5 * (a + 1))
            return None
        def take(blk, tk, cik):
            n = len(blk.get("form", []))
            for i in range(n):
                if blk["form"][i] != "8-K" or "2.02" not in (blk.get("items") or [""] * n)[i] or blk["filingDate"][i] < since: continue
                acc = blk.get("acceptanceDateTime", [""] * n)[i]
                out.append({"ticker": tk, "cik": int(cik), "filed": blk["filingDate"][i], "accepted_utc": acc})
        async def one(r):
            nonlocal failed
            c10 = str(int(r["cik"])).zfill(10); d = await get(f"https://data.sec.gov/submissions/CIK{c10}.json")
            if not d: failed += 1; return
            take(d["filings"]["recent"], r["ticker"], r["cik"])
            for p in d["filings"].get("files", []):
                if p.get("filingTo", "") >= since:
                    pg = await get(f"https://data.sec.gov/submissions/{p['name']}")
                    if pg: take(pg, r["ticker"], r["cik"])
        for i in range(0, len(rows), 100):
            await asyncio.gather(*[one(r) for r in rows[i:i + 100]])
            logger.info(f"[ml3 sec] results dates {min(i + 100, len(rows))}/{len(rows)} · {len(out):,} found · failed {failed}")
    R = pd.DataFrame(out).drop_duplicates(["ticker", "filed"])
    acc = pd.to_datetime(R["accepted_utc"], utc=True, errors="coerce").dt.tz_convert("America/New_York")
    R["accepted_et"] = acc.dt.strftime("%Y-%m-%d %H:%M"); R["after_close"] = acc.dt.hour >= 16
    R.to_parquet(f"{OUT}/results_dates.parquet", index=False)
    s = {"events": len(R), "companies": int(R["ticker"].nunique()), "failed": failed, "earliest": R["filed"].min(), "after_close_share": float(R["after_close"].mean())}
    logger.info(f"[ml3 sec] results dates: {s}"); return s


def _tsv(z, name, cols):
    with z.open(name) as f:
        df = pd.read_csv(f, sep="\t", dtype=str, quoting=3, on_bad_lines="skip", encoding_errors="replace")
    miss = [c for c in cols if c not in df.columns]
    if miss: raise KeyError(f"{name} missing {miss}; has {list(df.columns)[:20]}")
    return df[cols]


def _date(s):
    d = pd.to_datetime(s, format="%d-%b-%Y", errors="coerce")
    return d.fillna(pd.to_datetime(s, errors="coerce"))


async def build_insiders(pool, first_year=2006):
    import httpx
    ciks = {int(r["cik"]) for r in await pool.fetch("""SELECT u.cik FROM company_facts f JOIN universe u USING (ticker)
        WHERE f.as_of=(SELECT max(as_of) FROM company_facts) AND f.tier IN ('large','mid','small') AND u.cik IS NOT NULL""")}
    os.makedirs(f"{OUT}/insider_parts", exist_ok=True); today = dt.date.today(); got = skipped = 0
    async with httpx.AsyncClient(timeout=180, headers=UA, follow_redirects=True) as cx:
        for y in range(first_year, today.year + 1):
            for q in range(1, 5):
                if dt.date(y, 3 * q, 1) > today: break
                part = f"{OUT}/insider_parts/{y}q{q}.parquet"
                if os.path.exists(part): got += 1; continue
                r = None
                for a in range(3):
                    try: r = await cx.get(f"https://www.sec.gov/files/structureddata/data/insider-transactions-data-sets/{y}q{q}_form345.zip")
                    except Exception: r = None
                    if r is not None and r.status_code == 200: break
                    await asyncio.sleep(10)
                if r is None or r.status_code != 200: logger.info(f"[ml3 sec] insiders {y}q{q}: not available"); skipped += 1; continue
                try:
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    S = _tsv(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"] + (["AFF10B5ONE"] if True else []))
                except KeyError:
                    S = _tsv(z, "SUBMISSION.tsv", ["ACCESSION_NUMBER", "FILING_DATE", "DOCUMENT_TYPE", "ISSUERCIK", "ISSUERTRADINGSYMBOL"]); S["AFF10B5ONE"] = None
                O = _tsv(z, "REPORTINGOWNER.tsv", ["ACCESSION_NUMBER", "RPTOWNERCIK", "RPTOWNER_RELATIONSHIP", "RPTOWNER_TITLE"]).drop_duplicates("ACCESSION_NUMBER")
                T = _tsv(z, "NONDERIV_TRANS.tsv", ["ACCESSION_NUMBER", "TRANS_DATE", "TRANS_CODE", "TRANS_SHARES", "TRANS_PRICEPERSHARE", "TRANS_ACQUIRED_DISP_CD"])
                T = T[T["TRANS_CODE"].isin(["P", "S"])]
                S = S[S["DOCUMENT_TYPE"].astype(str).str.startswith("4")]
                S = S[pd.to_numeric(S["ISSUERCIK"], errors="coerce").isin(ciks)]
                D = T.merge(S, on="ACCESSION_NUMBER").merge(O, on="ACCESSION_NUMBER", how="left")
                D = pd.DataFrame({"issuer_cik": pd.to_numeric(D["ISSUERCIK"], errors="coerce").astype("Int64"), "symbol": D["ISSUERTRADINGSYMBOL"],
                                  "owner_cik": pd.to_numeric(D["RPTOWNERCIK"], errors="coerce").astype("Int64"), "role": D["RPTOWNER_RELATIONSHIP"],
                                  "title": D["RPTOWNER_TITLE"], "filed": _date(D["FILING_DATE"]), "trans_date": _date(D["TRANS_DATE"]), "code": D["TRANS_CODE"],
                                  "shares": pd.to_numeric(D["TRANS_SHARES"], errors="coerce"), "price": pd.to_numeric(D["TRANS_PRICEPERSHARE"], errors="coerce"),
                                  "planned_10b5_1": D["AFF10B5ONE"]})
                D["value"] = D["shares"] * D["price"]
                D.to_parquet(part, index=False); got += 1
                logger.info(f"[ml3 sec] insiders {y}q{q}: {len(D):,} open-market trades in our universe ({(D['code'] == 'P').sum():,} purchases)")
                await asyncio.sleep(1)
    import glob
    I = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{OUT}/insider_parts/*.parquet"))], ignore_index=True)
    I.to_parquet(f"{OUT}/insiders.parquet", index=False)
    s = {"quarters": got, "skipped": skipped, "trades": len(I), "purchases": int((I["code"] == "P").sum()), "issuers": int(I["issuer_cik"].nunique()),
         "earliest": str(I["trans_date"].min())[:10]}
    logger.info(f"[ml3 sec] insider history: {s}"); return s
