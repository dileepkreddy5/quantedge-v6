"""ML v2, stage 2b — point-in-time fundamentals history from the SEC bulk file.

Every quarter each company reported (not just the last 8), each with the date it was FILED, so a
model on any past date only sees what had been published by then. Uses the same extraction as the
nightly business facts (sales tag choice, 4 net-income tags, year-to-date cash flow split).
Also the cover-page share count each filing reports (dei:EntityCommonStockSharesOutstanding)."""
from __future__ import annotations
import os, json
import pandas as pd
from loguru import logger
from quantedge.facts.business_facts import REV_TAGS, _q_income, _q_cashflow
from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk

OUT = "/app/models/panel_v2"


def history(cik):
    facts = company_facts_from_bulk(str(cik))
    if not facts: return [], []
    revs = [s for s in (_q_income(facts, t) for t in REV_TAGS) if s]
    if not revs: return [], []
    rev = max(revs, key=lambda s: max(s))
    gp, op = _q_income(facts, "GrossProfit"), _q_income(facts, "OperatingIncomeLoss")
    cost = {}
    for _t in ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold", "CostOfServices"):
        for _e, _v in (_q_income(facts, _t) or {}).items(): cost.setdefault(_e, _v)
    ni = {}
    for t in ("NetIncomeLoss", "ProfitLoss", "NetIncomeLossAvailableToCommonStockholdersBasic", "NetIncomeLossAvailableToCommonStockholdersDiluted"):
        for e, v in (_q_income(facts, t) or {}).items(): ni.setdefault(e, v)
    ocf = _q_cashflow(facts)
    q = [{"end": e, "filed": rev[e]["filed"], "sales": rev[e]["val"], "gross_profit": (gp[e]["val"] if e in gp else ((rev[e]["val"] - cost[e]["val"]) if e in cost else None)),
          "op_income": (op.get(e) or {}).get("val"), "net_income": (ni.get(e) or {}).get("val"), "op_cash_flow": (ocf.get(e) or {}).get("val")}
         for e in sorted(rev)]
    sh = []
    for u in (((facts.get("facts") or {}).get("dei") or {}).get("EntityCommonStockSharesOutstanding") or {}).get("units", {}).get("shares", []):
        if u.get("val") and u.get("end") and u.get("filed"): sh.append({"asof": u["end"], "filed": u["filed"], "shares": float(u["val"])})
    return q, sh


async def build_fund_history(pool):
    os.makedirs(OUT, exist_ok=True)
    rows = await pool.fetch("""SELECT f.ticker, u.cik FROM company_facts f JOIN universe u USING (ticker)
        WHERE f.as_of = (SELECT max(as_of) FROM company_facts) AND f.primary_listing AND NOT f.is_spac
          AND f.tier IN ('large','mid','small') AND u.cik IS NOT NULL""")
    Q, S, ok = [], [], 0
    for i, r in enumerate(rows):
        try: q, sh = history(r["cik"])
        except Exception: q, sh = [], []
        if q: ok += 1
        Q += [{"ticker": r["ticker"], **x} for x in q]; S += [{"ticker": r["ticker"], **x} for x in sh]
        if i % 500 == 0: logger.info(f"[ml2] fundamentals history {i}/{len(rows)}")
    qd, sd = pd.DataFrame(Q), pd.DataFrame(S).drop_duplicates()
    qd.to_parquet(f"{OUT}/fund_quarters.parquet", index=False); sd.to_parquet(f"{OUT}/fund_shares.parquet", index=False)
    per = qd.groupby("ticker").size()
    summ = {"companies": len(rows), "with_quarters": ok, "quarter_rows": len(qd), "share_rows": len(sd),
            "quarters_per_company_median": int(per.median()) if len(per) else 0,
            "earliest_filed": str(qd["filed"].min()) if len(qd) else None,
            "with_share_counts": int(sd["ticker"].nunique()) if len(sd) else 0}
    json.dump(summ, open(f"{OUT}/fund_summary.json", "w"), indent=1)
    logger.info(f"[ml2] fundamentals history: {summ}")
    return summ
