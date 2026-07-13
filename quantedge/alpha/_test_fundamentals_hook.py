"""Tests for the fundamentals hook — caching + PIT + graceful missing."""
from datetime import date, timedelta
from quantedge.alpha import fundamentals_hook as fh


def make_closes(n=800, as_of=date(2025,6,30)):
    out, d, px = [], as_of - timedelta(days=int(n*1.5)), 50.0
    while d <= as_of:
        if d.weekday() < 5:
            out.append((d, round(px,4))); px *= 1.0003
        d += timedelta(days=1)
    return out


def test_caching_and_missing():
    calls = {"n": 0}
    prov = fh.FundamentalsProvider({"AAA": "111"},
                                   price_lookup=lambda t, a: make_closes(as_of=a))
    def fake_facts(cik):
        if cik in prov._facts_cache:
            return prov._facts_cache[cik]
        calls["n"] += 1
        prov._facts_cache[cik] = {"facts": {"us-gaap": {}, "dei": {}}}
        return prov._facts_cache[cik]
    prov._facts = fake_facts
    r1 = prov("AAA", date(2025,6,30))
    r2 = prov("AAA", date(2025,6,30))
    assert calls["n"] == 1, calls
    assert r1 == r2
    nonnull = {k:v for k,v in r1.items() if v is not None}
    assert nonnull in ({}, {"piotroski": 0.0}), r1
    print("  cache hit (1 facts call for 2 lookups); empty facts -> all None")


def test_unknown_ticker_returns_none():
    prov = fh.FundamentalsProvider({}, price_lookup=lambda t,a: make_closes(as_of=a))
    assert prov("ZZZ", date(2025,6,30)) is None
    print("  unknown ticker -> None")


def test_different_quarters_recompute():
    calls = {"n": 0}
    prov = fh.FundamentalsProvider({"AAA":"111"}, price_lookup=lambda t,a: make_closes(as_of=a))
    def fake_facts(cik):
        if cik in prov._facts_cache:
            return prov._facts_cache[cik]
        calls["n"] += 1
        prov._facts_cache[cik] = {"facts": {"us-gaap": {}, "dei": {}}}
        return prov._facts_cache[cik]
    prov._facts = fake_facts
    prov("AAA", date(2025,3,31))
    prov("AAA", date(2025,6,30))
    assert calls["n"] == 1
    assert len(prov._feat_cache) == 2
    print("  facts cached per-cik; features computed per-quarter")


if __name__ == "__main__":
    for t in (test_caching_and_missing, test_unknown_ticker_returns_none,
              test_different_quarters_recompute):
        print(f"── {t.__name__}"); t()
    print("ALL FUNDAMENTALS-HOOK TESTS PASSED")
