"""Multibagger-trait scorer (Manual §4 — RU-001 fundamentals).

Three signals from the founding papers, each fully transparent:
  - Piotroski F-Score (0-9): durable improver vs value trap (Piotroski 2000)
  - Gross profitability: gross_profit / assets (Novy-Marx 2013)
  - Growth acceleration: is revenue growth itself rising? (the inflection)

Every check shows its inputs. A score you can't audit is decoration.
"""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ScoreResult:
    ticker: str
    piotroski: int
    piotroski_checks: dict           # each check -> bool, fully visible
    gross_profitability: float | None
    rev_growth_recent: float | None
    rev_growth_prior: float | None
    growth_accelerating: bool | None
    notes: list = field(default_factory=list)


def _yoy(series: dict, y: int):
    """Year-over-year change helper; returns (curr, prev) or (None, None)."""
    if y in series and (y - 1) in series:
        return series[y], series[y - 1]
    return None, None


def score(ticker: str, f: dict) -> ScoreResult:
    """f = {metric: {fiscal_year: value}} from edgar_facts.fetch_annual_facts."""
    years = sorted(set().union(*[set(v.keys()) for v in f.values() if v]))
    if len(years) < 2:
        return ScoreResult(ticker, 0, {}, None, None, None, None, ["insufficient history"])
    y = years[-1]            # most recent fiscal year
    checks = {}
    notes = []

    ni  = f["net_income"].get(y);   ni_p  = f["net_income"].get(y-1)
    ocf = f["op_cash_flow"].get(y)
    a   = f["assets"].get(y);       a_p   = f["assets"].get(y-1)
    li  = f["liabilities"].get(y);  li_p  = f["liabilities"].get(y-1)
    rev = f["revenue"].get(y);      rev_p = f["revenue"].get(y-1)
    gp  = f["gross_profit"].get(y); gp_p  = f["gross_profit"].get(y-1)
    ca  = f["cur_assets"].get(y);   ca_p  = f["cur_assets"].get(y-1)
    cl  = f["cur_liab"].get(y);     cl_p  = f["cur_liab"].get(y-1)
    sh  = f["shares"].get(y);       sh_p  = f["shares"].get(y-1)

    # --- Piotroski 9 checks ---
    checks["1_positive_net_income"]   = (ni is not None and ni > 0)
    checks["2_positive_op_cash_flow"] = (ocf is not None and ocf > 0)
    # 3. Rising ROA (net income / assets)
    if None not in (ni, a, ni_p, a_p) and a and a_p:
        checks["3_rising_roa"] = (ni/a) > (ni_p/a_p)
    else: checks["3_rising_roa"] = False
    # 4. Quality of earnings: OCF > net income
    checks["4_ocf_gt_net_income"] = (ocf is not None and ni is not None and ocf > ni)
    # 5. Falling leverage (liabilities/assets)
    if None not in (li, a, li_p, a_p) and a and a_p:
        checks["5_falling_leverage"] = (li/a) < (li_p/a_p)
    else: checks["5_falling_leverage"] = False
    # 6. Rising current ratio
    if None not in (ca, cl, ca_p, cl_p) and cl and cl_p:
        checks["6_rising_current_ratio"] = (ca/cl) > (ca_p/cl_p)
    else: checks["6_rising_current_ratio"] = False
    # 7. No dilution (shares not up materially) — guard against stock splits
    if sh is not None and sh_p is not None and sh_p:
        ratio = sh / sh_p
        if ratio > 1.5:            # likely a split, not dilution
            checks["7_no_dilution"] = True
            notes.append(f"share jump {ratio:.1f}x treated as split, not dilution")
        else:
            checks["7_no_dilution"] = sh <= sh_p * 1.02   # allow 2% noise
    else: checks["7_no_dilution"] = False
    # 8. Rising gross margin
    if None not in (gp, rev, gp_p, rev_p) and rev and rev_p:
        checks["8_rising_gross_margin"] = (gp/rev) > (gp_p/rev_p)
    else: checks["8_rising_gross_margin"] = False
    # 9. Rising asset turnover (revenue/assets)
    if None not in (rev, a, rev_p, a_p) and a and a_p:
        checks["9_rising_asset_turnover"] = (rev/a) > (rev_p/a_p)
    else: checks["9_rising_asset_turnover"] = False

    piotroski = sum(1 for v in checks.values() if v)

    # --- Novy-Marx gross profitability ---
    gross_prof = (gp / a) if (gp is not None and a) else None

    # --- Growth acceleration: rev growth recent vs prior ---
    g_recent = g_prior = None
    accel = None
    if rev and rev_p:
        g_recent = rev/rev_p - 1
    if rev_p and f["revenue"].get(y-2):
        g_prior = rev_p / f["revenue"][y-2] - 1
    if g_recent is not None and g_prior is not None:
        accel = g_recent > g_prior

    return ScoreResult(ticker, piotroski, checks, gross_prof,
                       g_recent, g_prior, accel, notes)
