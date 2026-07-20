"""Fetch + parse SEC 13G/13D major-holder filings from EDGAR. Returns institutional
ownership signals: major holders, concentration, stake sizes."""
import re, asyncio, httpx
from datetime import datetime, timezone, timedelta

_UA={"User-Agent":"QuantEdge research contact@quantedge.io"}

def _parse_13g(t):
    def g(tag):
        m=re.search(rf'<{tag}[^>]*>([^<]+)</{tag}>',t,re.I)
        return m.group(1).strip() if m else None
    name=g("reportingPersonName") or g("filingPersonName") or g("filerName")
    pct=g("classPercent")
    amt=g("amountBeneficiallyOwned")
    try: pct=float(pct) if pct else None
    except: pct=None
    try: amt=float(amt) if amt else None
    except: amt=None
    return {"holder":name,"percent":pct,"shares":amt}

async def fetch_ownership(cik, days_back=730, max_filings=30):
    cik=str(cik).zfill(10)
    out={"available":False}
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            sub=await c.get(f"https://data.sec.gov/submissions/CIK{cik}.json",headers=_UA)
            if sub.status_code!=200: return out
            rec=(sub.json() or {}).get("filings",{}).get("recent",{})
            forms=rec.get("form",[]); dates=rec.get("filingDate",[])
            accs=rec.get("accessionNumber",[]); docs=rec.get("primaryDocument",[])
            cutoff=(datetime.now(timezone.utc)-timedelta(days=days_back)).date().isoformat()
            idxs=[i for i,f in enumerate(forms) if "13G" in f and dates[i]>=cutoff][:max_filings]
            if not idxs: return out
            cik_num=cik.lstrip("0")
            sem=asyncio.Semaphore(6)
            async def fetch_one(i):
                acc=accs[i].replace("-",""); doc=docs[i].split("/")[-1]
                url=f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc}/{doc}"
                async with sem:
                    try:
                        r=await c.get(url,headers=_UA)
                        if r.status_code==200:
                            p=_parse_13g(r.text); p["date"]=dates[i]; return p
                    except Exception: return None
                return None
            results=[r for r in await asyncio.gather(*[fetch_one(i) for i in idxs]) if r and r.get("holder")]
            if not results: return out
            # dedupe by holder, keep latest
            latest={}
            for r in results:
                h=r["holder"]
                if h not in latest or r["date"]>latest[h]["date"]: latest[h]=r
            holders=list(latest.values())
            pcts=[h["percent"] for h in holders if h["percent"] is not None]
            out.update({
              "available":True,
              "major_holders":len(holders),
              "total_disclosed_pct":sum(pcts) if pcts else None,
              "top_holder_pct":max(pcts) if pcts else None,
              "avg_holder_pct":sum(pcts)/len(pcts) if pcts else None,
              "concentration":sum(sorted(pcts,reverse=True)[:3]) if len(pcts)>=3 else (sum(pcts) if pcts else None),
              "holders":sorted(holders,key=lambda x:x.get("percent") or 0,reverse=True)[:10],
            })
            return out
    except Exception:
        return out
