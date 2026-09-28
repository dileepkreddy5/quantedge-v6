"""Where the money comes from: revenue by product/service, business segment and region,
read from the XBRL data file of the company's latest 10-K. Cached per filing (changes yearly).

Companies often tag both a total and its parts on the same axis (Apple: "Products" plus
iPhone/Mac/iPad/Wearables). Lines are reconciled against total revenue and an overlapping
total is dropped, so shares never double-count."""
from __future__ import annotations
import datetime as dt, json, re
from xml.etree import ElementTree as ET
import httpx

REV = ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "RevenueFromContractWithCustomerIncludingAssessedTax",
       "SalesRevenueNet", "RevenuesNetOfInterestExpense")
AXES = [("ProductOrServiceAxis", "By product or service"), ("StatementBusinessSegmentsAxis", "By business segment"),
        ("StatementGeographicalAxis", "By region")]


def _ln(tag): return tag.split("}", 1)[-1]
def _days(a, b): return (dt.date.fromisoformat(b) - dt.date.fromisoformat(a)).days


FIX = [("Three Six Five", "365"), ("Linked In", "LinkedIn"), ("Non Us", "Outside the US"), ("Non US", "Outside the US"),
       ("Service Other", "Services and other"), ("XBOX", "Xbox"), ("I Phone", "iPhone"), ("I Pad", "iPad")]


def _nice(qname, labels):
    key = qname.replace(":", "_")
    if key in labels: t = re.sub(r"\s*\[Member\]$", "", labels[key]).strip()
    else:
        base = re.sub(r"Member$", "", qname.split(":")[-1])
        t = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", base).strip()
        if t in ("US", "U S"): t = "United States"
    for a, b in FIX: t = t.replace(a, b)
    if t.isupper() and len(t) > 4: t = t.title()                      # "UNITED STATES" -> "United States"
    return t


def _labels(xml):
    if not xml: return {}
    root = ET.fromstring(xml); X = "{http://www.w3.org/1999/xlink}"
    loc, lab, arc = {}, {}, []
    for e in root.iter():
        n = _ln(e.tag)
        if n == "loc": loc[e.get(X + "label")] = (e.get(X + "href") or "").split("#")[-1]
        elif n == "label":
            role = e.get(X + "role") or ""
            if role.endswith("/label") or role.endswith("terseLabel"): lab.setdefault(e.get(X + "label"), (e.text or "").strip())
        elif n == "labelArc": arc.append((e.get(X + "from"), e.get(X + "to")))
    return {loc[f]: lab[t] for f, t in arc if f in loc and t in lab}


def parse(instance_xml, labels_xml=None):
    root = ET.fromstring(instance_xml); labels = _labels(labels_xml)
    ctx = {}
    for c in root.iter():
        if _ln(c.tag) != "context": continue
        mems, s, e = [], None, None
        for x in c.iter():
            n = _ln(x.tag)
            if n == "explicitMember": mems.append((x.get("dimension") or "", (x.text or "").strip()))
            elif n == "startDate": s = (x.text or "").strip()
            elif n == "endDate": e = (x.text or "").strip()
        if s and e: ctx[c.get("id")] = (mems, s, e)
    facts = []
    for x in root:
        n = _ln(x.tag)
        if n in REV and x.get("contextRef") in ctx and (x.text or "").strip():
            try: v = float(x.text.strip())
            except ValueError: continue
            m, s, e = ctx[x.get("contextRef")]
            if 350 <= _days(s, e) <= 380: facts.append((n, m, s, e, v))
    if not facts: return None
    # Pick the revenue label per breakdown (Google tags products and segments under different
    # labels); the total comes from whichever label reports it for that year.
    ends_dim = [f[3] for f in facts if f[1]]
    end = max(ends_dim) if ends_dim else max(f[3] for f in facts)
    nodim = [f for f in facts if not f[1] and f[3] == end]
    total = max((f[4] for f in nodim), default=None)
    out = []
    for axis, title in AXES:
        tags = [t for t in REV if any(f[0] == t and len(f[1]) == 1 and f[1][0][0].endswith(axis) and f[3] == end for f in facts)]
        if not tags: continue
        tag = max(tags, key=lambda t: sum(1 for f in facts if f[0] == t and len(f[1]) == 1 and f[1][0][0].endswith(axis)))
        F = [f for f in facts if f[0] == tag]
        cur, prev = {}, {}
        for n, m, s, e, v in F:
            if len(m) != 1 or not m[0][0].endswith(axis): continue
            mem = m[0][1]
            if e == end: cur.setdefault(mem, v)
            elif 340 <= _days(e, end) <= 390: prev.setdefault(mem, v)
        if len(cur) < 2: continue
        items = dict(cur)
        if total and sum(items.values()) > total * 1.03:
            # Several complete splits can share one axis (Microsoft: Product + Service AND a detailed
            # product list). Keep the MOST DETAILED set of lines that adds up to total revenue.
            from itertools import combinations
            keys = sorted(items, key=lambda k: -items[k])[:16]; best = None
            for r_ in range(len(keys), 1, -1):
                for combo in combinations(keys, r_):
                    if abs(sum(items[k] for k in combo) - total) <= total * 0.02: best = combo; break
                if best: break
            if best: items = {k: items[k] for k in best}
        base = total or sum(items.values())
        rows = sorted(({"label": _nice(k, labels), "revenue": v, "share": v / base if base else None,
                        "growth": (v / prev[k] - 1) if prev.get(k) else None} for k, v in items.items()), key=lambda r: -r["revenue"])
        out.append({"axis": title, "items": rows})
    return {"fiscal_year_end": end, "total_revenue": total, "breakdowns": out, "labels_loaded": len(labels)} if out else None


async def get_segments(pool, cik: str, ticker: str, ua: str):
    await pool.execute("""CREATE TABLE IF NOT EXISTS company_segments (accession TEXT PRIMARY KEY, cik TEXT, ticker TEXT,
                          data JSONB, created_at TIMESTAMPTZ DEFAULT NOW())""")
    async with httpx.AsyncClient(timeout=40, headers={"User-Agent": ua}) as cx:
        sub = (await cx.get(f"https://data.sec.gov/submissions/CIK{int(cik):010d}.json")).json()
        rec = sub.get("filings", {}).get("recent", {})
        acc = next((rec["accessionNumber"][i] for i, fm in enumerate(rec.get("form", [])) if fm == "10-K"), None)
        if not acc: return None
        hit = await pool.fetchval("SELECT data FROM company_segments WHERE accession=$1", acc)
        if hit: return json.loads(hit) if isinstance(hit, str) else hit
        base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc.replace('-', '')}"
        names = [i["name"] for i in (await cx.get(base + "/index.json")).json().get("directory", {}).get("item", [])]
        inst = next((n for n in names if n.endswith("_htm.xml")), None) or next((n for n in names if n.endswith(".xml")
                    and not re.search(r"(_cal|_def|_lab|_pre)\.xml$|FilingSummary", n)), None)
        lab = next((n for n in names if n.endswith("_lab.xml")), None)
        if not inst: return None
        ixml = (await cx.get(f"{base}/{inst}")).content
        lxml = (await cx.get(f"{base}/{lab}")).content if lab else None
    data = parse(ixml, lxml)
    if data:
        data.update({"accession": acc, "url": f"{base}/", "source": "the company's latest 10-K (SEC XBRL data)"})
        await pool.execute("INSERT INTO company_segments (accession, cik, ticker, data) VALUES ($1,$2,$3,$4::jsonb) ON CONFLICT DO NOTHING",
                           acc, cik, ticker, json.dumps(data))
    return data
