"""Fetch + parse SEC Form 4 insider transactions from EDGAR. Returns aggregated insider
activity (buy/sell counts, net shares, officer vs director) over recent filings."""
import re, asyncio, httpx
from datetime import datetime, timezone, timedelta

_UA={"User-Agent":"QuantEdge research contact@quantedge.io"}

def _parse_form4(xml):
    def g(tag,s):
        m=re.search(rf'<{tag}>\s*<value>([^<]+)</value>',s) or re.search(rf'<{tag}>([^<]+)</{tag}>',s)
        return m.group(1).strip() if m else None
    name=g('rptOwnerName',xml)
    is_officer=('<isOfficer>1' in xml) or ('<isOfficer>true' in xml.lower())
    is_director=('<isDirector>1' in xml) or ('<isDirector>true' in xml.lower())
    is_ten=('<isTenPercentOwner>1' in xml) or ('<isTenPercentOwner>true' in xml.lower())
    txns=[]
    for block in re.findall(r'<nonDerivativeTransaction>.*?</nonDerivativeTransaction>',xml,re.DOTALL):
        sh=g('transactionShares',block); px=g('transactionPricePerShare',block); cd=g('transactionAcquiredDisposedCode',block)
        code=g('transactionCode',block)
        if sh:
            try: txns.append({"shares":float(sh),"price":float(px) if px else 0.0,"ad":cd,"code":code})
            except: pass
    return {"name":name,"is_officer":is_officer,"is_director":is_director,"is_ten":is_ten,"txns":txns}

async def fetch_insider_activity(cik, days_back=365, max_filings=60):
    """Aggregate insider Form 4 activity over the recent window."""
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
            idxs=[i for i,f in enumerate(forms) if f=="4" and dates[i]>=cutoff][:max_filings]
            if not idxs: return out
            cik_num=cik.lstrip("0")
            buy_txn=0; sell_txn=0; buy_val=0.0; sell_val=0.0; buy_sh=0.0; sell_sh=0.0
            officer_buys=0; officer_sells=0; director_buys=0; director_sells=0
            n_parsed=0; buyers=set(); sellers=set()
            async def fetch_one(i):
                acc=accs[i].replace("-",""); doc=docs[i].split("/")[-1]
                url=f"https://www.sec.gov/Archives/edgar/data/{cik_num}/{acc}/{doc}"
                try:
                    r=await c.get(url,headers=_UA)
                    if r.status_code==200: return _parse_form4(r.text)
                except Exception: return None
                return None
            sem=asyncio.Semaphore(6)
            async def guarded(i):
                async with sem: return await fetch_one(i)
            results=await asyncio.gather(*[guarded(i) for i in idxs])
            for p in results:
                if not p or not p["txns"]: continue
                n_parsed+=1
                # only count open-market buys (P) and sells (S); skip grants/options (A/M/F/G)
                for t in p["txns"]:
                    code=t.get("code") or ""
                    val=t["shares"]*(t["price"] or 0)
                    if code=="P" or (t["ad"]=="A" and code not in ("A","M","G","F")):  # purchase
                        buy_txn+=1; buy_val+=val; buy_sh+=t["shares"]
                        if p["name"]: buyers.add(p["name"])
                        if p["is_officer"]: officer_buys+=1
                        if p["is_director"]: director_buys+=1
                    elif code=="S" or t["ad"]=="D":  # sale/disposal
                        sell_txn+=1; sell_val+=val; sell_sh+=t["shares"]
                        if p["name"]: sellers.add(p["name"])
                        if p["is_officer"]: officer_sells+=1
                        if p["is_director"]: director_sells+=1
            if n_parsed==0: return out
            total=buy_txn+sell_txn
            out.update({
              "available":True,"filings_parsed":n_parsed,
              "buy_txns":buy_txn,"sell_txns":sell_txn,
              "buy_value":buy_val,"sell_value":sell_val,
              "buy_sell_txn_ratio":(buy_txn/total) if total>0 else None,
              "net_insider_value":buy_val-sell_val,
              "buy_value_ratio":(buy_val/(buy_val+sell_val)) if (buy_val+sell_val)>0 else None,
              "unique_buyers":len(buyers),"unique_sellers":len(sellers),
              "officer_net":officer_buys-officer_sells,"director_net":director_buys-director_sells,
              "cluster_buying":1.0 if len(buyers)>=3 else 0.0,
              "any_insider_buying":1.0 if buy_txn>0 else 0.0,
            })
            return out
    except Exception:
        return out
