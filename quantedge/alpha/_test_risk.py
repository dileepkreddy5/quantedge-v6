from quantedge.alpha.risk import (
    inverse_vol_weights, vol_target_leverage, drawdown_governor, size_book)


def test_inverse_vol_weights():
    w = inverse_vol_weights({"A": 0.20, "B": 0.40})
    assert abs(w["A"] - 2/3) < 1e-9 and abs(w["B"] - 1/3) < 1e-9
    assert abs(sum(w.values()) - 1.0) < 1e-9
    print("  inverse-vol: lower vol gets more weight, sums to 1")


def test_vol_target_leverage():
    w = {"A": 0.5, "B": 0.5}; v = {"A": 0.20, "B": 0.20}
    lev = vol_target_leverage(w, v, target_ann_vol=0.40, max_leverage=2.0)
    assert lev > 1.0, lev
    lev2 = vol_target_leverage(w, v, target_ann_vol=0.05, max_leverage=2.0)
    assert lev2 < 1.0, lev2
    print(f"  vol-target: lever up {round(lev,2)}, lever down {round(lev2,2)}")


def test_drawdown_governor():
    assert drawdown_governor(-0.02) == 1.0
    assert drawdown_governor(-0.25) == 0.3
    mid = drawdown_governor(-0.14)
    assert 0.3 < mid < 1.0
    print(f"  governor: flat->cut->floor, mid={round(mid,2)}")


def test_size_book_integration():
    longs = ["A", "B", "C"]
    vols = {"A": 0.20, "B": 0.30, "C": 0.25}
    healthy = size_book(longs, vols, target_ann_vol=0.15, current_drawdown=-0.02)
    stressed = size_book(longs, vols, target_ann_vol=0.15, current_drawdown=-0.25)
    assert sum(healthy.values()) > sum(stressed.values())
    assert all(w >= 0 for w in healthy.values())
    print(f"  sized book: healthy gross {round(sum(healthy.values()),3)} > "
          f"stressed {round(sum(stressed.values()),3)}")


if __name__ == "__main__":
    for t in (test_inverse_vol_weights, test_vol_target_leverage,
              test_drawdown_governor, test_size_book_integration):
        print(f"── {t.__name__}"); t()
    print("ALL RISK TESTS PASSED")
