"""Tests for the cross-sectional model + walk-forward. PYTHONPATH=. python this."""
import random
from quantedge.alpha.model import CrossSectionalModel, walk_forward, _spearman
from quantedge.alpha.features import FEATURE_NAMES

random.seed(11)


def make_month(as_of, n, signal_strength):
    rows = []
    for k in range(n):
        r3m = random.gauss(0, 0.1)
        feats = {f: random.gauss(0, 1) for f in FEATURE_NAMES}
        feats["r_3m"] = r3m
        fwd = signal_strength * r3m + random.gauss(0, 0.05)
        row = {"as_of": as_of, "ticker": f"T{k}"}
        row.update(feats)
        row["_fwd_3m_raw"] = fwd
        rows.append(row)
    order = sorted(range(n), key=lambda i: rows[i]["_fwd_3m_raw"])
    for rank, i in enumerate(order):
        rows[i]["fwd_3m_rank"] = rank/(n-1)
    return rows


def test_spearman():
    assert abs(_spearman([1,2,3,4,5],[1,2,3,4,5]) - 1.0) < 1e-9
    assert abs(_spearman([1,2,3,4,5],[5,4,3,2,1]) + 1.0) < 1e-9
    print("  spearman +1/-1 correct")


def test_recovers_planted_signal():
    panel = {f"2024-{m:02d}-28": make_month(f"2024-{m:02d}-28", 120, 1.5) for m in range(1,13)}
    panel.update({f"2025-{m:02d}-28": make_month(f"2025-{m:02d}-28", 120, 1.5) for m in range(1,7)})
    res = walk_forward(panel, "fwd_3m_rank", min_train_months=6)
    assert res["mean_rank_ic"] > 0.15, res["mean_rank_ic"]
    assert res["ic_t_stat"] > 3, res["ic_t_stat"]
    assert list(res["feature_importance"].keys())[0] == "r_3m"
    print(f"  planted signal recovered: IC {res['mean_rank_ic']}, t {res['ic_t_stat']}, "
          f"top '{list(res['feature_importance'].keys())[0]}' ({res['backend']})")


def test_rejects_pure_noise():
    panel = {f"2024-{m:02d}-28": make_month(f"2024-{m:02d}-28", 120, 0.0) for m in range(1,13)}
    panel.update({f"2025-{m:02d}-28": make_month(f"2025-{m:02d}-28", 120, 0.0) for m in range(1,7)})
    res = walk_forward(panel, "fwd_3m_rank", min_train_months=6)
    assert res["mean_rank_ic"] is None or abs(res["mean_rank_ic"]) < 0.05, res["mean_rank_ic"]
    print(f"  pure noise: IC {res['mean_rank_ic']} (no false signal)")


def test_no_lookahead_in_walk_forward():
    panel = {}
    for m in range(1,13): panel[f"2024-{m:02d}-28"] = make_month(f"2024-{m:02d}-28",120,1.5)
    for m in range(1,7):  panel[f"2025-{m:02d}-28"] = make_month(f"2025-{m:02d}-28",120,-1.5)
    res = walk_forward(panel, "fwd_3m_rank", min_train_months=6)
    post = [ic for d, ic in res["monthly_ic"] if d.startswith("2025")]
    assert any(ic < 0 for ic in post), post
    print(f"  no-lookahead confirmed: post-flip negatives {[round(x,2) for x in post[:4]]}")


if __name__ == "__main__":
    for t in (test_spearman, test_recovers_planted_signal,
              test_rejects_pure_noise, test_no_lookahead_in_walk_forward):
        print(f"── {t.__name__}")
        t()
    print("ALL MODEL TESTS PASSED")
