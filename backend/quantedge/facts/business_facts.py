"""Nightly facts sheet, part 2: business facts from the SEC bulk file.

Per company, the last 8 quarters of sales, gross profit, operating income, net
income and operating cash flow, each dated to its FIRST filing (point-in-time).
Income-statement quarters come from SEC-framed 3-month periods (Q4 = annual - 9M).
Operating cash flow is reported year-to-date in 10-Qs, so each quarter is the
difference between consecutive year-to-date totals within the same fiscal year.

Derived: sales growth vs a year ago (latest and previous quarter -> accelerating?),
margin trends, profit trend, cash backing profits, and a plain health check.
Stored in company_facts.fundamentals for the latest as_of.
"""
from __future__ import annotations
import json
from datetime import date, timedelta
from loguru import logger
from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk

REV_TAGS = ["Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"]


def _q_income(facts, tag):
    """end-date -> {val, filed}. Periods classified by their actual dates, not SEC's
    calendar frames: frames only line up for December fiscal years, so companies with
    other year-ends (Estee Lauder, Nike, Micron, Walmart...) lost their fiscal Q4.
    Quarter = 80-100 days; year = 350-380 days; a missing fiscal Q4 = year minus the
    three quarters inside it, dated to the year's end and its 10-K filing."""
    try: units = facts["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError: return {}
    first = {}
    for r in units:
        if r.get("form") not in ("10-Q", "10-K") or r.get("val") is None or not r.get("start"): continue
        k = (r["start"], r["end"])
        if k not in first or r["filed"] < first[k]["filed"]: first[k] = {"val": float(r["val"]), "filed": r["filed"]}
    q, years = {}, []
    for (st, en), v in first.items():
        span = (date.fromisoformat(en) - date.fromisoformat(st)).days
        if 80 <= span <= 100: q[en] = {**v, "start": st}
        elif 350 <= span <= 380: years.append((st, en, v))
    for st, en, v in years:
        if en in q: continue
        inside = [x for e, x in q.items() if x["start"] >= st and e < en]
        if len(inside) == 3:
            q[en] = {"val": v["val"] - sum(x["val"] for x in inside), "filed": v["filed"], "start": st, "derived": True}
    return {e: {"val": x["val"], "filed": x["filed"]} for e, x in q.items()}


def _q_cashflow(facts, tag="NetCashProvidedByUsedInOperatingActivities"):
    try: units = facts["facts"]["us-gaap"][tag]["units"]["USD"]
    except KeyError: return {}
    first = {}
    for r in units:
        if r.get("form") not in ("10-Q", "10-K") or r.get("val") is None or not r.get("start"): continue
        k = (r["start"], r["end"])
        if k not in first or r["filed"] < first[k]["filed"]: first[k] = {"val": float(r["val"]), "filed": r["filed"]}
    by_start = {}
    for (st, en), v in first.items(): by_start.setdefault(st, []).append((en, v))
    out = {}
    for st, lst in by_start.items():
        lst.sort(); prev = 0.0
        for en, v in lst:
            span = (date.fromisoformat(en) - date.fromisoformat(st)).days
            if span > 380: break
            out[en] = {"val": v["val"] - prev, "filed": v["filed"]}; prev = v["val"]
    return out


def _yoy(series, end, key="val"):
    e = date.fromisoformat(end)
    for other, v in series.items():
        if abs((e - date.fromisoformat(other)).days - 365) <= 20 and series[end][key] and v[key]:
            return series[end][key] / v[key] - 1 if v[key] > 0 else None
    return None


def business_facts(cik: str) -> dict | None:
    facts = company_facts_from_bulk(str(cik))
    if not facts: return None
    revs = [(t, _q_income(facts, t)) for t in REV_TAGS]
    revs = [(t, s) for t, s in revs if s]
    if not revs: return {"available": False, "reason": "no quarterly sales in SEC filings"}
    rev_tag, rev = max(revs, key=lambda ts: max(ts[1]))            # the tag with the most recent quarter
    gp, op, ni = _q_income(facts, "GrossProfit"), _q_income(facts, "OperatingIncomeLoss"), _q_income(facts, "NetIncomeLoss")
    ocf = _q_cashflow(facts)
    ends = sorted(rev)[-8:]
    qs = []
    for e in ends:
        r = rev[e]["val"]
        qs.append({"end": e, "filed": rev[e]["filed"], "sales": r,
                   "gross_margin": gp[e]["val"] / r if e in gp and r else None,
                   "op_margin": op[e]["val"] / r if e in op and r else None,
                   "net_income": ni.get(e, {}).get("val"), "op_cash_flow": ocf.get(e, {}).get("val"),
                   "sales_yoy": _yoy(rev, e)})
    if not qs: return {"available": False, "reason": "no recent quarters"}
    last, prev = qs[-1], (qs[-2] if len(qs) > 1 else None)
    streak = 0
    for i in range(len(qs) - 1, 0, -1):
        a, b = qs[i]["sales_yoy"], qs[i - 1]["sales_yoy"]
        if a is not None and b is not None and a > b: streak += 1
        else: break
    gm_now, gm_then = last["gross_margin"], (qs[-5]["gross_margin"] if len(qs) >= 5 else None)
    om_now, om_then = last["op_margin"], (qs[-5]["op_margin"] if len(qs) >= 5 else None)
    ni_ttm = sum(q["net_income"] for q in qs[-4:] if q["net_income"] is not None) if len(qs) >= 4 else None
    ocf_ttm = sum(q["op_cash_flow"] for q in qs[-4:] if q["op_cash_flow"] is not None) if len(qs) >= 4 else None
    healthy = bool((last["sales_yoy"] is None or last["sales_yoy"] > -0.10) and (om_now is None or om_now > -0.05 or (om_then is not None and om_now > om_then)))
    stale = (date.today() - date.fromisoformat(last["end"])).days > 200
    return {"available": True, "stale": stale, "rev_tag": rev_tag, "quarters": qs,
            "sales_yoy": last["sales_yoy"], "sales_yoy_prev": prev["sales_yoy"] if prev else None,
            "accelerating": bool(prev and last["sales_yoy"] is not None and prev["sales_yoy"] is not None and last["sales_yoy"] > prev["sales_yoy"]),
            "acceleration_streak": streak,
            "gross_margin": gm_now, "gross_margin_change_1y": (gm_now - gm_then) if gm_now is not None and gm_then is not None else None,
            "op_margin": om_now, "op_margin_change_1y": (om_now - om_then) if om_now is not None and om_then is not None else None,
            "net_income_ttm": ni_ttm, "op_cash_flow_ttm": ocf_ttm,
            "cash_backed": bool(ni_ttm is not None and ocf_ttm is not None and ni_ttm > 0 and ocf_ttm >= 0.8 * ni_ttm),
            "healthy": healthy and not stale}


async def build_business_facts(pool) -> dict:
    async with pool.acquire() as con:
        as_of = await con.fetchval("SELECT max(as_of) FROM company_facts")
        rows = await con.fetch("""SELECT f.ticker, u.cik FROM company_facts f JOIN universe u USING (ticker)
                                  WHERE f.as_of=$1 AND f.tier IN ('large','mid','small') AND NOT f.is_spac AND u.cik IS NOT NULL""", as_of)
    ok = none = 0
    for i, r in enumerate(rows):
        try: bf = business_facts(r["cik"])
        except Exception as e: bf = {"available": False, "reason": f"parse error: {type(e).__name__}"}
        if bf and bf.get("available"): ok += 1
        else: none += 1
        async with pool.acquire() as con:
            await con.execute("UPDATE company_facts SET fundamentals=$3::jsonb WHERE ticker=$1 AND as_of=$2",
                              r["ticker"], as_of, json.dumps(bf or {"available": False, "reason": "not in SEC bulk file"}, default=str))
        if (i + 1) % 500 == 0: logger.info(f"[facts] business facts {i + 1}/{len(rows)}")
    logger.info(f"[facts] business facts as of {as_of}: {ok} with quarters, {none} without")
    return {"as_of": str(as_of), "with_quarters": ok, "without": none}
