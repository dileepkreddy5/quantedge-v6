import os, tempfile
from datetime import date, timedelta
from quantedge.alpha.panel_cache import load_or_build, _cache_key


class FakeStore:
    def __init__(self, to="2026-07-10"):
        self._to = to
        self.data = {}
        base = date(2021,7,13)
        for t, drift in (("AAA",0.001),("BBB",0.0),("CCC",-0.0005)):
            d, px, n = base, 50.0, 0
            while n < 1250:
                if d.weekday()<5:
                    self.data.setdefault(d,{})[t]=(px,1e7); px*=(1+drift); n+=1
                d+=timedelta(days=1)
    def coverage(self):
        return {"from":"2021-07-13","to":self._to,"rows":len(self.data)*3}
    def closes_on(self, d):
        return {t:p for t,(p,_) in self.data.get(d,{}).items()}
    def series(self, t, start, end):
        return [(d, self.data[d][t][0], self.data[d][t][1])
                for d in sorted(self.data) if start<=d<=end and t in self.data[d]]


CFG = {"fwd_nbars": 63, "min_dollar_adv": 1e5}


def test_build_then_hit():
    s = FakeStore(); cikmap={"AAA":"1","BBB":"2","CCC":"3"}
    with tempfile.TemporaryDirectory() as tmp:
        cp = os.path.join(tmp, "panels.json")
        p1 = load_or_build(s, cikmap, CFG, cp, verbose=False)
        assert os.path.exists(cp)
        p2 = load_or_build(s, cikmap, CFG, cp, verbose=False)
        assert list(p1.keys()) == list(p2.keys()) and len(p1) > 0
        print(f"  built {len(p1)} months, cache hit returns same keys")


def test_staleness_on_store_change():
    cikmap={"AAA":"1","BBB":"2","CCC":"3"}
    with tempfile.TemporaryDirectory() as tmp:
        cp = os.path.join(tmp, "panels.json")
        load_or_build(FakeStore(to="2026-07-10"), cikmap, CFG, cp, verbose=False)
        k1 = _cache_key(FakeStore(to="2026-07-10").coverage(), CFG, 3)
        k2 = _cache_key(FakeStore(to="2026-08-10").coverage(), CFG, 3)
        assert k1 != k2
        print("  store-range change invalidates cache key")


if __name__ == "__main__":
    for t in (test_build_then_hit, test_staleness_on_store_change):
        print(f"── {t.__name__}"); t()
    print("ALL PANEL-CACHE TESTS PASSED")
