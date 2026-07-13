"""Pre-registered SECOND test (step-22) — stage-restricted REBOUND.

Reads the events already computed by backtest.py (rebound_backtest.json).
Re-scores under the frozen `rebound_stage_restricted` hypothesis: passers
count ONLY if their stage is in eligible_stages (TURNING, RECOVERING). The
control group is unchanged (the beaten-down base rate). Same stats, same
cost model. Judged against the SEPARATE, raised, pre-committed threshold.

No recomputation, no new data, no tuning. One shot.
"""
from __future__ import annotations
import argparse, importlib.util, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
import yaml

from quantedge.harness.costs import round_trip_cost_bps

_spec = importlib.util.spec_from_file_location(
    "rb_stats", os.path.join(os.path.dirname(__file__), "stats.py"))
st = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(st)

PARAMS = yaml.safe_load(open(os.path.join(
    os.path.dirname(__file__), "..", "..", "params.yaml")))
# tolerate either top-level or (legacy) nested-under-rebound placement
CFG = PARAMS.get("rebound_stage_restricted") or PARAMS["rebound"]["rebound_stage_restricted"]
KILL = CFG["kill_threshold"]
ELIGIBLE = set(CFG["eligible_stages"])


def restrict(events):
    """Keep control events as-is; keep passer events ONLY if eligible stage.
    Passers of ineligible stages are DROPPED (not moved to control) — the
    hypothesis is about a strategy that never buys them at all."""
    out = []
    for e in events:
        if e["passed"]:
            if e.get("stage") in ELIGIBLE:
                out.append(e)
        else:
            out.append(e)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest-json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    bt = json.load(open(args.backtest_json))
    events = restrict(bt["events"])
    regimes = bt["regimes"]
    rt = round_trip_cost_bps(PARAMS) / 1e4

    s6 = st.summarize(events, "6m", rt)
    s12 = st.summarize(events, "12m", rt)
    primary = s12 if (s12.get("n_dates") or 0) >= 3 else s6

    checks = {
        f"lift>={KILL['min_oos_lift_over_base_rate']}":
            (primary.get("lift") or 0) >= KILL["min_oos_lift_over_base_rate"],
        f"t>={KILL['min_t_stat']}":
            (primary.get("t_stat") or 0) >= KILL["min_t_stat"],
        f"regimes>={KILL['min_distinct_regimes']}":
            len(set(r for r in regimes if r != "UNKNOWN")) >= KILL["min_distinct_regimes"],
        f"dates>={KILL['min_dates']}":
            (primary.get("n_dates") or 0) >= KILL["min_dates"],
    }
    verdict = "PASS" if all(checks.values()) else "FAIL"

    out = {"test": "rebound_stage_restricted", "eligible_stages": sorted(ELIGIBLE),
           "regimes": regimes, "summary_6m": s6, "summary_12m": s12,
           "primary_horizon": primary.get("horizon"),
           "kill_threshold": KILL, "checks": checks, "verdict": verdict}
    json.dump(out, open(args.out, "w"), indent=1)

    print("=== PRE-REGISTERED STAGE-RESTRICTED TEST ===")
    print("eligible stages:", sorted(ELIGIBLE))
    for nm, s in (("6m", s6), ("12m", s12)):
        print(f"{nm}: dates={s['n_dates']} passers={s['n_pass_events']} "
              f"spread={s['mean_spread_per_date']} t={s['t_stat']} lift={s['lift']} "
              f"win={s['passer_win_rate']} vs {s['control_win_rate']}")
    print("checks:", checks)
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
