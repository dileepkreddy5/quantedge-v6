"""Async EDGAR fetcher — resolves ticker->CIK, pulls every concept with multi-tag
fallback, polite rate-limiting, in-process caching. Rolling-window relative to today.
"""
from __future__ import annotations
import asyncio, datetime as dt
from typing import Dict, List, Optional
import httpx
from loguru import logger
from quantedge.scoring.edgar_xbrl import (
    SEC_BASE, UA, CONCEPTS, parse_flow_concept, parse_stock_concept)

_CIK_CACHE: Dict[str, int] = {}
_CONCEPT_CACHE: Dict[str, dict] = {}
FLOW = {"capex","dividends_paid","buybacks","depreciation_amortization",
        "depreciation","amortization","sbc","interest_expense","interest_expense_full","sga","rd"}
STOCK = {"receivables","goodwill","intangibles","operating_lease_liab",
         "short_term_debt","inventory","accounts_payable","deferred_revenue",
         "retained_earnings","short_term_debt2","operating_lease_total","cash",
         "long_term_debt_edgar"}

async def _ticker_to_cik(ticker: str, client: httpx.AsyncClient) -> Optional[int]:
    """Resolve a ticker to its SEC CIK.

    This map is a static ~10k-entry file that changes weekly, but it was fetched
    on every cache miss. Under load the SEC returns 429 here, _ticker_to_cik
    returns None, and every concept call downstream fails — so EVERY
    EDGAR-sourced field on EVERY tab goes blank at once, with no error surfaced.
    Cache it on disk and log the throttle loudly rather than returning None into
    a pipeline that treats missing data as an ordinary absence.
    """
    t = ticker.upper().strip()
    if t in _CIK_CACHE:
        return _CIK_CACHE[t]
    import json as _json, os as _os, time as _time
    _cache_path = "/app/data/sec_ticker_map.json"
    try:
        if _os.path.exists(_cache_path) and _time.time() - _os.path.getmtime(_cache_path) < 7*86400:
            with open(_cache_path) as fh:
                for k, v in _json.load(fh).items():
                    _CIK_CACHE[k] = v
            if t in _CIK_CACHE:
                return _CIK_CACHE[t]
    except Exception as e:
        logger.warning(f"sec ticker map cache unreadable: {e}")
    r = await client.get("https://www.sec.gov/files/company_tickers.json",
                         headers={"User-Agent": UA}, timeout=30)
    if r.status_code != 200:
        logger.error(
            f"SEC ticker map returned HTTP {r.status_code} — CIK lookup for {t} "
            f"failed, so every EDGAR-sourced field will be empty for this request")
        return None
    _rows = list(r.json().values())
    for row in _rows:
        _CIK_CACHE[row["ticker"].upper()] = int(row["cik_str"])
    try:
        _os.makedirs(_os.path.dirname(_cache_path), exist_ok=True)
        with open(_cache_path, "w") as fh:
            _json.dump(_CIK_CACHE, fh)
        # Keep the titles too: when a ticker maps to an entity with no filings,
        # _resolve_filing_cik searches these for the company that does file.
        with open("/app/data/sec_ticker_map_full.json", "w") as fh:
            _json.dump(_rows, fh)
    except Exception as e:
        logger.warning(f"could not persist sec ticker map: {e}")
    return _CIK_CACHE.get(t)

_BULK_FACTS: Dict[int, dict] = {}

def _bulk_units(cik: int, tag: str) -> List[dict]:
    """Read one tag's USD series from the nightly companyfacts.zip.

    The per-company concept API rate-limits hard enough that fundamentals were
    unavailable for whole sessions — every EDGAR-sourced signal on every tab
    blank at once. The bulk archive is the same data, complete, already on the
    persistent volume and refreshed nightly by the Multibagger job, and its
    points carry the identical {end, val, fy, fp, form, frame} shape the
    parsers expect. No network, no throttling.
    """
    facts = _BULK_FACTS.get(cik)
    if facts is None:
        try:
            from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
            facts = company_facts_from_bulk(f"{cik:010d}") or {}
        except Exception as e:
            logger.warning(f"bulk facts unavailable for CIK {cik}: {e}")
            facts = {}
        _BULK_FACTS[cik] = facts
    node = ((facts.get("facts") or {}).get("us-gaap") or {}).get(tag)
    if not node:
        return []
    return (node.get("units") or {}).get("USD", []) or []



_CIK_REDIRECT: Dict[int, int] = {}

def _resolve_filing_cik(cik: int) -> int:
    """Follow a ticker's CIK to the entity that actually files.

    The SEC's company_tickers.json can point a ticker at a successor entity
    before that entity has filed anything: XOM maps to ExxonMobil Holdings Corp
    (CIK 2115436, zero tags in the bulk archive) rather than Exxon Mobil Corp
    (34088, 438 tags with revenue to 2026-03-31). Every EDGAR-sourced field for
    Exxon came back empty, which the valuation tab reported as "the filings are
    missing operating cash flow" — a data-quality message for what was really a
    lookup pointing at an empty shell. When the mapped CIK has no facts, find
    the entity whose name matches and does.
    """
    # Some restructurings move a ticker to a new registrant and drop the old one
    # from company_tickers.json entirely, so there is no path from the symbol
    # back to the entity holding the filing history. XOM points at ExxonMobil
    # Holdings Corp (2115436, zero tags) and Exxon Mobil Corp (34088, 438 tags)
    # is no longer listed under any ticker. Searching the bulk archive by name
    # would mean scanning 1.4GB per miss, so known cases are listed here.
    _KNOWN = {2115436: 34088}   # ExxonMobil Holdings -> Exxon Mobil Corp
    if cik in _KNOWN:
        return _KNOWN[cik]
    if cik in _CIK_REDIRECT:
        return _CIK_REDIRECT[cik]
    try:
        from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
        facts = company_facts_from_bulk(f"{cik:010d}") or {}
        if ((facts.get("facts") or {}).get("us-gaap") or {}):
            _CIK_REDIRECT[cik] = cik
            return cik
        name = (facts.get("entityName") or "").strip()
        if not name:
            _CIK_REDIRECT[cik] = cik
            return cik
        # "ExxonMobil Holdings Corp" -> "EXXONMOBIL"; match the distinctive stem
        # against the SEC's own ticker file, preferring the oldest registrant.
        import json as _json, os as _os, re as _re
        stem = _re.sub(r"[^A-Z]", "", name.upper())[:8]
        path = "/app/data/sec_ticker_map_full.json"
        if _os.path.exists(path):
            with open(path) as fh:
                rows = _json.load(fh)
            cands = [int(r["cik_str"]) for r in rows
                     if _re.sub(r"[^A-Z]", "", (r.get("title") or "").upper()).startswith(stem)
                     and int(r["cik_str"]) != cik]
            for c in sorted(cands):
                fx = company_facts_from_bulk(f"{c:010d}") or {}
                if ((fx.get("facts") or {}).get("us-gaap") or {}):
                    logger.info(f"CIK {cik} ({name}) has no facts; using {c} instead")
                    _CIK_REDIRECT[cik] = c
                    return c
    except Exception as e:
        logger.debug(f"CIK redirect check failed for {cik}: {e}")
    _CIK_REDIRECT[cik] = cik
    return cik


async def _fetch_concept(cik: int, tags: List[str], client: httpx.AsyncClient) -> List[dict]:
    """Return the mapped tag whose data runs closest to the present.

    This previously returned the first tag with any points at all. Filers change
    tags: NVIDIA stopped filing PaymentsToAcquirePropertyPlantAndEquipment in
    2020 and moved to PaymentsToAcquireProductiveAssets, and Coca-Cola left
    LongTermDebtNoncurrent in 2024 for LongTermDebtAndCapitalLeaseObligations.
    Both newer tags were already in the map, but sat behind a stale one that
    still returned data — so capital intensity and leverage trend were computed
    from series that ended years ago, or not at all. Nothing raised, because a
    stale series is a valid series.
    """
    cik = _resolve_filing_cik(cik)
    best: List[dict] = []
    best_end = ""
    for tag in tags:
        units = _bulk_units(cik, tag)
        if units:
            last = max((u.get("end") or "") for u in units)
            if last > best_end:
                best, best_end = units, last
    if best:
        return best
    # Nothing in the archive for any mapped tag — fall back to the concept API.
    for tag in tags:
        ck = f"{cik}:{tag}"
        if ck in _CONCEPT_CACHE:
            units = _CONCEPT_CACHE[ck].get("units", {}).get("USD", [])
        else:
            url = SEC_BASE.format(cik=cik, concept=tag)
            try:
                r = await client.get(url, headers={"User-Agent": UA}, timeout=30)
            except httpx.HTTPError:
                continue
            await asyncio.sleep(0.12)
            if r.status_code != 200:
                continue
            data = r.json()
            _CONCEPT_CACHE[ck] = data
            units = data.get("units", {}).get("USD", [])
        if not units:
            continue
        last = max((u.get("end") or "") for u in units)
        if last > best_end:
            best, best_end = units, last
    return best

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
