"""SEC balance-sheet history: total assets and shareholders' equity at each quarter end, dated by the FIRST
filing that reported them (later comparatives don't count), so a model never sees a figure early."""
from __future__ import annotations
import json
import pandas as pd
from loguru import logger
from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk

V2 = "/app/models/panel_v2"
TAGS = (("assets", ("Assets",)), ("equity", ("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")))


def balance(cik):
    g = ((company_facts_from_bulk(str(cik)) or {}).get("facts") or {}).get("us-gaap") or {}
    res = {}
    for key, tags in TAGS:
        for tag in tags:
            items = [u for u in (g.get(tag) or {}).get("units", {}).get("USD", [])
                     if "start" not in u and u.get("val") is not None and u.get("filed") and str(u.get("form", "")).startswith("10-")]
            if not items: continue
            for u in items:
                r = res.setdefault(u["end"], {})
                if key not in r or u["filed"] < r[key + "_f"]: r[key] = float(u["val"]); r[key + "_f"] = u["filed"]
            break
    return [{"end": e, "filed": r.get("assets_f") or r.get("equity_f"), "assets": r.get("assets"), "equity": r.get("equity")} for e, r in sorted(res.items())]


async def build_balance(pool):
    rows = await pool.fetch("""SELECT f.ticker, u.cik FROM company_facts f JOIN universe u USING (ticker)
        WHERE f.as_of=(SELECT max(as_of) FROM company_facts) AND f.primary_listing AND NOT f.is_spac AND f.tier IN ('large','mid','small') AND u.cik IS NOT NULL""")
    out = []
    for i, r in enumerate(rows):
        try: out += [{"ticker": r["ticker"], **x} for x in balance(r["cik"])]
        except Exception: pass
        if i % 500 == 0: logger.info(f"[ml3] balance sheets {i}/{len(rows)}")
    B = pd.DataFrame(out); B.to_parquet(f"{V2}/fund_balance.parquet", index=False)
    s = {"rows": len(B), "companies": int(B["ticker"].nunique()), "with_equity": int(B["equity"].notna().sum())}
    logger.info(f"[ml3] balance history: {s}"); return s
