"""Tests for the panel builder. PYTHONPATH=. python this."""
from datetime import date, timedelta
from quantedge.alpha.panel import (
    forward_return, month_ends, cross_sectional_rank, build_panel_for_date,
)


def test_forward_return_real_future():
    bars = [(date(2024,1,1)+timedelta(days=i), 100*(1.01**i), 1e6) for i in range(100)]
    r = forward_return(bars, date(2024,1,10), 20)
    assert r is not None and abs(r - (1.01**20 - 1)) < 1e-6, r
    assert forward_return(bars, date(2024,4,5), 20) is None
    print("  forward return real + unmatured=None:", round(r,4))


def test_month_ends():
    me = month_ends(date(2024,1,15), date(2024,4,15))
    assert me == [date(2024,1,31), date(2024,2,29), date(2024,3,31)], me
    print("  month ends:", [d.isoformat() for d in me])


def test_cross_sectional_rank():
    vals = [0.5, None, 0.1, 0.9, 0.3]
    r = cross_sectional_rank(vals)
    assert r[1] is None
    assert r[2] == 0.0 and r[3] == 1.0
    assert abs(r[0]-2/3)<1e-9
    print("  cross-sectional rank [0,1], None preserved")


class FakeStore:
    def __init__(self):
        self.data = {}
        base = date(2023,1,1)
        for t, drift in (("AAA",0.001),("BBB",0.0),("CCC",-0.0005)):
            d, px, n = base, 50.0, 0
            while n < 700:
                if d.weekday()<5:
                    self.data.setdefault(d,{})[t]=(px,1e7); px*=(1+drift); n+=1
                d+=timedelta(days=1)
    def closes_on(self, d):
        return {t:p for t,(p,_) in self.data.get(d,{}).items()}
    def series(self, t, start, end):
        return [(d, self.data[d][t][0], self.data[d][t][1])
                for d in sorted(self.data) if start<=d<=end and t in self.data[d]]


def test_build_panel_slice():
    s = FakeStore()
    cikmap = {"AAA":"1","BBB":"2","CCC":"3"}
    as_of = sorted(s.data)[400]
    rows = build_panel_for_date(s, cikmap, as_of, fwd_nbars=63, min_dollar_adv=1e5)
    assert len(rows) == 3, len(rows)
    by = {r["ticker"]: r for r in rows}
    assert by["AAA"]["r_3m"] > by["CCC"]["r_3m"]
    assert all(r["fwd_3m_rank"] is None or 0<=r["fwd_3m_rank"]<=1 for r in rows)
    print("  panel slice: 3 tickers, AAA r_3m", round(by["AAA"]["r_3m"],3),
          "> CCC", round(by["CCC"]["r_3m"],3))


if __name__ == "__main__":
    for t in (test_forward_return_real_future, test_month_ends,
              test_cross_sectional_rank, test_build_panel_slice):
        print(f"── {t.__name__}")
        t()
    print("ALL PANEL TESTS PASSED")
