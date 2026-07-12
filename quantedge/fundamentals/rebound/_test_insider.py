"""Tests for the insider Form 4 layer (parser + clusters). Network fetch path
is smoke-tested on the VPS; here we prove parsing and cluster math with
realistic ownershipDocument fixtures. Run: PYTHONPATH=. python this_file."""
from datetime import date
import yaml, os

from quantedge.fundamentals.rebound.insider_form4 import (
    parse_form4_xml, opportunistic_cluster,
)

PARAMS = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "..", "params.yaml")))

AS_OF = date(2026, 7, 10)


def form4(owner, cik, officer, director, title, txns):
    tx = ""
    for code, acq, shares, price, d in txns:
        tx += f"""
    <nonDerivativeTransaction>
      <transactionDate><value>{d}</value></transactionDate>
      <transactionCoding><transactionCode>{code}</transactionCode></transactionCoding>
      <transactionAmounts>
        <transactionShares><value>{shares}</value></transactionShares>
        <transactionPricePerShare><value>{price}</value></transactionPricePerShare>
        <transactionAcquiredDisposedCode><value>{acq}</value></transactionAcquiredDisposedCode>
      </transactionAmounts>
    </nonDerivativeTransaction>"""
    return f"""<?xml version="1.0"?>
<ownershipDocument>
  <reportingOwner>
    <reportingOwnerId>
      <rptOwnerCik>{cik}</rptOwnerCik>
      <rptOwnerName>{owner}</rptOwnerName>
    </reportingOwnerId>
    <reportingOwnerRelationship>
      <isDirector>{int(director)}</isDirector>
      <isOfficer>{int(officer)}</isOfficer>
      <officerTitle>{title}</officerTitle>
    </reportingOwnerRelationship>
  </reportingOwner>
  <nonDerivativeTable>{tx}
  </nonDerivativeTable>
</ownershipDocument>"""


def test_parser_purchases_only():
    xml = form4("DOE JANE", "0001111111", True, False, "Chief Financial Officer",
                [("P", "A", 10000, 12.50, "2026-06-20"),     # open-market buy
                 ("S", "D", 5000, 13.10, "2026-06-25"),      # sale — ignored
                 ("A", "A", 20000, 0.0, "2026-06-25")])      # grant — ignored
    buys = parse_form4_xml(xml)
    assert len(buys) == 1, buys
    b = buys[0]
    assert b["value"] == 125000.0 and b["is_officer"] and not b["is_director"]
    assert b["officer_title"] == "Chief Financial Officer"
    print("  1 purchase extracted ($125,000); sale + grant ignored")


def test_parser_malformed():
    assert parse_form4_xml("<not-xml") == []
    assert parse_form4_xml("<ownershipDocument></ownershipDocument>") == []
    print("  malformed / ownerless documents return empty, never raise")


def _buy(owner, cik, officer, director, title, value, filed):
    return {"filed": filed, "trans_date": filed, "owner_cik": cik,
            "owner_name": owner, "is_officer": officer, "is_director": director,
            "officer_title": title, "shares": value / 10.0, "price": 10.0,
            "value": value}


def test_cluster_detection():
    buys = [
        _buy("DOE JANE", "01", True, False, "CFO", 900_000, "2026-06-20"),
        _buy("SMITH ROBERT", "02", False, True, "", 500_000, "2026-05-30"),
        _buy("LEE KAREN", "03", False, True, "", 400_000, "2026-06-28"),
        _buy("DOE JANE", "01", True, False, "CFO", 250_000, "2026-07-01"),  # same buyer
    ]
    c = opportunistic_cluster(buys, AS_OF, PARAMS)
    assert c["cluster"] is True and c["n_buyers"] == 3, c
    assert c["total_value"] == 2_050_000.0
    assert c["n_officers"] == 1 and c["n_directors"] == 2
    print("  cluster:", c["reason"])


def test_single_buyer_not_cluster():
    buys = [_buy("DOE JANE", "01", True, False, "CFO", 900_000, "2026-06-20")]
    c = opportunistic_cluster(buys, AS_OF, PARAMS)
    assert c["cluster"] is False and c["n_buyers"] == 1
    print("  single buyer correctly not a cluster")


def test_window_pit():
    # a big cluster that all happened 6 months ago must NOT count today,
    # and a filing dated AFTER as_of must be invisible
    buys = [
        _buy("A", "01", True, False, "CEO", 1e6, "2026-01-05"),
        _buy("B", "02", False, True, "", 1e6, "2026-01-08"),
        _buy("C", "03", False, True, "", 1e6, "2026-01-12"),
        _buy("D", "04", True, False, "CFO", 1e6, "2026-07-20"),  # future filing
    ]
    c = opportunistic_cluster(buys, AS_OF, PARAMS)
    assert c["cluster"] is False and c["n_buyers"] == 0, c
    print("  stale cluster + future filing both excluded (window/PIT)")


if __name__ == "__main__":
    for t in (test_parser_purchases_only, test_parser_malformed,
              test_cluster_detection, test_single_buyer_not_cluster,
              test_window_pit):
        print(f"── {t.__name__}")
        t()
    print("ALL INSIDER TESTS PASSED")
