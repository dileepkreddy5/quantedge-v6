"""Async EDGAR fetcher — resolves ticker->CIK, pulls every concept with multi-tag
fallback, polite rate-limiting, in-process caching. Rolling-window relative to today.
"""
from __future__ import annotations
import asyncio, datetime as dt
from typing import Dict, List, Optional
import httpx
from quantedge.scoring.edgar_xbrl import (
    SEC_BASE, UA, CONCEPTS, parse_flow_concept, parse_stock_concept)

_CIK_CACHE: Dict[str, int] = {}
_CONCEPT_CACHE: Dict[str, dict] = {}
FLOW = {"capex","dividends_paid","buybacks","depreciation_amortization","sbc",
        "interest_expense","sga","rd"}
STOCK = {"receivables","goodwill","intangibles","operating_lease_liab",
         "short_term_debt","inventory","accounts_payable"}

async def _ticker_to_cik(ticker: str, client: httpx.AsyncClient) -> Optional[int]:
    t = ticker.upper().strip()
    if t in _CIK_CACHE:
        return _CIK_CACHE[t]
    r = await client.get("https://www.sec.gov/files/company_tickers.json",
                         headers={"User-Agent": UA}, timeout=30)
    if r.status_code != 200:
        return None
    for row in r.json().values():
        _CIK_CACHE[row["ticker"].upper()] = int(row["cik_str"])
    return _CIK_CACHE.get(t)

async def _fetch_concept(cik: int, tags: List[str], client: httpx.AsyncClient) -> List[dict]:
    for tag in tags:
        ck = f"{cik}:{tag}"
        if ck in _CONCEPT_CACHE:
            return _CONCEPT_CACHE[ck].get("units", {}).get("USD", [])
        url = SEC_BASE.format(cik=cik, concept=tag)
        try:
            r = await client.get(url, headers={"User-Agent": UA}, timeout=30)
        except httpx.HTTPError:
            continue
        await asyncio.sleep(0.12)
        if r.status_code == 200:
            data = r.json()
            _CONCEPT_CACHE[ck] = data
            units = data.get("units", {}).get("USD", [])
            if units:
                return units
    return []

async def fetch_edgar_supplement(ticker: str, years_back: int = 6) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    async with httpx.AsyncClient() as client:
        cik = await _ticker_to_cik(ticker, client)
        if cik is None:
            return out
        for metric, tags in CONCEPTS.items():
            units = await _fetch_concept(cik, tags, client)
            if not units:
                out[metric] = []
                continue
            if metric in FLOW:
                out[metric] = parse_flow_concept(units, years_back)
            else:
                out[metric] = parse_stock_concept(units, years_back)
    return out
