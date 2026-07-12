"""REBOUND Layer 4b — insider open-market purchases from Form 4 (step-17).

Mechanism (Cohen-Malloy-Pomorski 2012; Lakonishok-Lee 2001): insider OPEN-
MARKET purchases — transaction code "P" — predict returns; grants, awards and
option exercises do not. Clusters (multiple distinct insiders buying within a
window) are the strongest form: several people who know the business best,
independently deciding the market price is wrong, with their own money.

Design: we do NOT ingest every Form 4 in America. Insider history is fetched
PER CANDIDATE via SEC's submissions API — only for stocks that already passed
the discount/health/knife layers — and cached in SQLite by accession number so
nothing is fetched twice. Form 4s carry their own filed date, so historical
cluster detection for the backtest is PIT-exact.

Honest scope note: full CMP "routine vs opportunistic" classification needs
3y of per-insider calendar habits; v1 ships pure code-P clusters (code P
already excludes grants/awards/exercises). Whether clusters add predictive
lift on OUR data is exactly what the step-21 backtest measures — the gate
decides, not the paper.

SEC etiquette: descriptive User-Agent, <=8 req/s pacing, cache-first.
"""
from __future__ import annotations
import json, os, re, sqlite3, time, urllib.request
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from typing import Dict, List, Optional

UA = os.environ.get("SEC_USER_AGENT", "Dileep Kapu dileepkreddy5@gmail.com")
_PACE = 0.13  # ~8 req/s max per SEC guidance

DEFAULT_DB = os.environ.get(
    "INSIDER_STORE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "insider_store.db"),
)


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    time.sleep(_PACE)
    return data


# ── Form 4 XML parsing (pure — fully unit-testable) ───────────

def _txt(node, path) -> Optional[str]:
    el = node.find(path)
    return el.text.strip() if el is not None and el.text else None


def parse_form4_xml(xml_text: str) -> List[Dict]:
    """Extract OPEN-MARKET PURCHASES (code P, acquired) from one Form 4.

    Returns one dict per P-transaction:
      {owner_name, owner_cik, is_officer, is_director, officer_title,
       trans_date, shares, price, value}
    Sales (S), grants (A), exercises (M/F/...) and derivative tables are
    ignored — they are not the CMP signal."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    owners = []
    for ro in root.findall(".//reportingOwner"):
        owners.append({
            "owner_name": _txt(ro, ".//rptOwnerName") or "UNKNOWN",
            "owner_cik": _txt(ro, ".//rptOwnerCik") or "",
            "is_officer": (_txt(ro, ".//isOfficer") or "0").strip() in ("1", "true"),
            "is_director": (_txt(ro, ".//isDirector") or "0").strip() in ("1", "true"),
            "officer_title": _txt(ro, ".//officerTitle") or "",
        })
    if not owners:
        return []
    owner = owners[0]  # co-filers share the transactions; attribute to first

    out = []
    for tr in root.findall(".//nonDerivativeTransaction"):
        code = _txt(tr, ".//transactionCoding/transactionCode")
        acq = _txt(tr, ".//transactionAcquiredDisposedCode/value")
        if code != "P" or (acq and acq != "A"):
            continue
        try:
            shares = float(_txt(tr, ".//transactionShares/value") or 0)
            price = float(_txt(tr, ".//transactionPricePerShare/value") or 0)
        except ValueError:
            continue
        tdate = _txt(tr, ".//transactionDate/value")
        if shares <= 0 or not tdate:
            continue
        out.append({
            **owner,
            "trans_date": tdate,
            "shares": shares,
            "price": price,
            "value": round(shares * price, 2),
        })
    return out


# ── SQLite cache + issuer collection ──────────────────────────

class InsiderStore:
    def __init__(self, path: str = DEFAULT_DB):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._con = sqlite3.connect(path)
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS buys ("
            " accession TEXT, issuer_cik TEXT, filed TEXT, trans_date TEXT,"
            " owner_cik TEXT, owner_name TEXT, is_officer INTEGER,"
            " is_director INTEGER, officer_title TEXT,"
            " shares REAL, price REAL, value REAL,"
            " PRIMARY KEY (accession, owner_cik, trans_date, shares))"
        )
        self._con.execute(
            "CREATE TABLE IF NOT EXISTS scanned ("
            " accession TEXT PRIMARY KEY, issuer_cik TEXT, filed TEXT, n_buys INTEGER)"
        )
        self._con.execute("CREATE INDEX IF NOT EXISTS idx_buys_issuer ON buys(issuer_cik, filed)")
        self._con.commit()

    def _seen(self, accession: str) -> bool:
        return self._con.execute(
            "SELECT 1 FROM scanned WHERE accession=?", (accession,)).fetchone() is not None

    def collect_for_issuer(self, cik: str, since: date, until: date) -> int:
        """Fetch + parse all Form 4s FILED in [since, until] for one issuer.
        Cached by accession — reruns cost zero network. Returns new buys stored."""
        cik10 = str(cik).zfill(10)
        subs = json.loads(_get(f"https://data.sec.gov/submissions/CIK{cik10}.json"))
        recent = subs.get("filings", {}).get("recent", {})
        rows = list(zip(recent.get("form", []), recent.get("accessionNumber", []),
                        recent.get("filingDate", []), recent.get("primaryDocument", [])))
        # older filings live in paginated files; pull any page overlapping the range
        for extra in subs.get("filings", {}).get("files", []):
            if extra.get("filingTo", "9999") >= since.isoformat():
                try:
                    older = json.loads(_get(
                        "https://data.sec.gov/submissions/" + extra["name"]))
                    rows += list(zip(older.get("form", []), older.get("accessionNumber", []),
                                     older.get("filingDate", []), older.get("primaryDocument", [])))
                except Exception:
                    continue

        new_buys = 0
        for form, acc, filed, doc in rows:
            if form not in ("4", "4/A"):
                continue
            if not (since.isoformat() <= filed <= until.isoformat()):
                continue
            if self._seen(acc):
                continue
            acc_nodash = acc.replace("-", "")
            try:
                # the primary document is the form4 XML (or an index we resolve)
                url = (f"https://www.sec.gov/Archives/edgar/data/{int(cik10)}/"
                       f"{acc_nodash}/{doc}")
                raw = _get(url).decode("utf-8", "ignore")
                if "<ownershipDocument" not in raw:
                    m = re.search(r'href="([^"]+\.xml)"', raw)
                    if m:
                        raw = _get("https://www.sec.gov" + m.group(1)).decode("utf-8", "ignore")
                buys = parse_form4_xml(raw)
            except Exception:
                buys = []
            for b in buys:
                self._con.execute(
                    "INSERT OR IGNORE INTO buys VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (acc, cik10, filed, b["trans_date"], b["owner_cik"],
                     b["owner_name"], int(b["is_officer"]), int(b["is_director"]),
                     b["officer_title"], b["shares"], b["price"], b["value"]))
                new_buys += 1
            self._con.execute(
                "INSERT OR REPLACE INTO scanned VALUES (?,?,?,?)",
                (acc, cik10, filed, len(buys)))
        self._con.commit()
        return new_buys

    def buys_for(self, cik: str, since: date, as_of: date) -> List[Dict]:
        """Stored buys FILED in [since, as_of] — filed-date filter = PIT."""
        cur = self._con.execute(
            "SELECT filed, trans_date, owner_cik, owner_name, is_officer,"
            " is_director, officer_title, shares, price, value FROM buys"
            " WHERE issuer_cik=? AND filed>=? AND filed<=?",
            (str(cik).zfill(10), since.isoformat(), as_of.isoformat()))
        cols = ["filed", "trans_date", "owner_cik", "owner_name", "is_officer",
                "is_director", "officer_title", "shares", "price", "value"]
        return [dict(zip(cols, r)) for r in cur]

    def close(self):
        self._con.close()


# ── cluster detection (pure — fully unit-testable) ────────────

def opportunistic_cluster(buys: List[Dict], as_of: date, params: dict) -> Dict:
    """Cluster verdict from a list of P-purchases (already PIT-filtered by
    filed date). Cluster = >= min distinct buyers within the window."""
    p = params["rebound"]["confirm"]
    window_start = as_of - timedelta(days=p["insider_cluster_window_days"])
    inwin = [b for b in buys
             if window_start.isoformat() <= b["filed"] <= as_of.isoformat()]
    buyers: Dict[str, Dict] = {}
    for b in inwin:
        key = b["owner_cik"] or b["owner_name"]
        agg = buyers.setdefault(key, {"name": b["owner_name"], "value": 0.0,
                                      "officer": False, "director": False,
                                      "title": b.get("officer_title", "")})
        agg["value"] += b["value"]
        agg["officer"] |= bool(b["is_officer"])
        agg["director"] |= bool(b["is_director"])

    n = len(buyers)
    total = round(sum(v["value"] for v in buyers.values()), 2)
    cluster = n >= p["insider_cluster_min_buyers"]
    result = {
        "ok": True,
        "n_buyers": n,
        "n_purchases": len(inwin),
        "total_value": total,
        "n_officers": sum(1 for v in buyers.values() if v["officer"]),
        "n_directors": sum(1 for v in buyers.values() if v["director"]),
        "cluster": bool(cluster),
        "window_days": p["insider_cluster_window_days"],
    }
    if cluster:
        who = []
        if result["n_officers"]:
            titles = [v["title"] for v in buyers.values() if v["officer"] and v["title"]]
            who.append(titles[0] if titles else f"{result['n_officers']} officer(s)")
        if result["n_directors"]:
            who.append(f"{result['n_directors']} director(s)")
        result["reason"] = (f"{n} insiders bought ${total:,.0f} open-market"
                            + (f" ({' + '.join(who)})" if who else ""))
    elif n == 1:
        result["reason"] = f"1 insider bought ${total:,.0f} open-market (no cluster)"
    else:
        result["reason"] = "no open-market insider buys in window"
    return result
