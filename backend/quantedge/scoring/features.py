"""Compute the feature dict the catalog scores against — real values only.
A metric that can't be computed is absent (never faked) so rollup lowers
confidence rather than scoring a lie.
"""
from __future__ import annotations
import math
from typing import List, Optional, Dict, Any


def _f(v) -> Optional[float]:
    try:
        v = float(v)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _cov_stability(series: List[float]) -> Optional[float]:
    vals = [x for x in (_f(s) for s in series) if x is not None]
    if len(vals) < 3:
        return None
    m = sum(vals) / len(vals)
    if m == 0:
        return None
    var = sum((x - m) ** 2 for x in vals) / len(vals)
    return max(0.0, 1.0 - (var ** 0.5) / abs(m))


def _cagr(first, last, years) -> Optional[float]:
    first, last = _f(first), _f(last)
    if first is None or last is None or first <= 0 or last <= 0 or years <= 0:
        return None
    return (last / first) ** (1.0 / years) - 1.0


def _slope(series: List[float]) -> Optional[float]:
    vals = [x for x in (_f(s) for s in series) if x is not None]
    n = len(vals)
    if n < 3:
        return None
    xs = list(range(n)); mx = sum(xs)/n; my = sum(vals)/n
    denom = sum((x-mx)**2 for x in xs)
    if denom == 0:
        return None
    return sum((xs[i]-mx)*(vals[i]-my) for i in range(n)) / denom


def _ttm(quarters, attr) -> Optional[float]:
    vals = [_f(getattr(q, attr, None)) for q in quarters[-4:]]
    vals = [v for v in vals if v is not None]
    return sum(vals) if len(vals) == 4 else None


def _annual_series(quarters, attr) -> List[float]:
    out = []
    for i in range(3, len(quarters)):
        vals = [_f(getattr(q, attr, None)) for q in quarters[i-3:i+1]]
        vals = [v for v in vals if v is not None]
        if len(vals) == 4:
            out.append(sum(vals))
    return out


def compute_features(quarters: List[Any], market_cap: Optional[float] = None,
                     beta: Optional[float] = None, wacc: Optional[float] = None) -> Dict[str, float]:
    f: Dict[str, float] = {}
    if not quarters:
        return f
    q = sorted(quarters, key=lambda x: (getattr(x,"fiscal_year",0), getattr(x,"fiscal_period","")))
    latest = q[-1]
    rev_ttm=_ttm(q,"revenue"); ni_ttm=_ttm(q,"net_income"); ocf_ttm=_ttm(q,"operating_cash_flow")
    oi_ttm=_ttm(q,"operating_income"); gp_ttm=_ttm(q,"gross_profit"); capex_ttm=_ttm(q,"capex")
    rev_annual=_annual_series(q,"revenue"); ni_annual=_annual_series(q,"net_income")

    if len(rev_annual)>=2 and rev_annual[-2]>0: f["rev_growth_yoy"]=rev_annual[-1]/rev_annual[-2]-1.0
    if len(rev_annual)>=4:
        c=_cagr(rev_annual[-4],rev_annual[-1],3)
        if c is not None: f["rev_cagr_3y"]=c
    if len(rev_annual)>=6:
        c=_cagr(rev_annual[-6],rev_annual[-1],5)
        if c is not None: f["rev_cagr_5y"]=c
    rq=[_f(getattr(x,"revenue",None)) for x in q]
    if len(rq)>=2 and rq[-2]: f["rev_growth_qoq"]=rq[-1]/rq[-2]-1.0
    growths=[rev_annual[i]/rev_annual[i-1]-1.0 for i in range(1,len(rev_annual)) if rev_annual[i-1]>0]
    if growths:
        s=_slope(growths)
        if s is not None: f["rev_growth_persistence"]=s
        if len(growths)>=3:
            m=sum(growths)/len(growths); f["rev_volatility"]=(sum((x-m)**2 for x in growths)/len(growths))**0.5

    if rev_ttm:
        if gp_ttm is not None: f["gross_margin"]=gp_ttm/rev_ttm
        if oi_ttm is not None: f["operating_margin"]=oi_ttm/rev_ttm; f["ebitda_margin"]=oi_ttm/rev_ttm
        if ni_ttm is not None: f["net_margin"]=ni_ttm/rev_ttm
        if capex_ttm is not None: f["capex_intensity"]=abs(capex_ttm)/rev_ttm
        if ocf_ttm is not None and capex_ttm is not None: f["fcf_margin"]=(ocf_ttm-abs(capex_ttm))/rev_ttm

    gm=[(_f(getattr(x,"gross_profit",None)) or 0)/(_f(getattr(x,"revenue",None)) or 1) for x in q if _f(getattr(x,"revenue",None))]
    st=_cov_stability(gm);  f["gross_margin_stability"]=st if st is not None else f.get("gross_margin_stability")
    om=[(_f(getattr(x,"operating_income",None)) or 0)/(_f(getattr(x,"revenue",None)) or 1) for x in q if _f(getattr(x,"revenue",None))]
    st=_cov_stability(om);  f["operating_margin_stability"]=st if st is not None else f.get("operating_margin_stability")
    s=_slope(gm)
    if s is not None: f["gross_margin_trend"]=s
    s=_slope(om)
    if s is not None: f["operating_margin_trend"]=s

    def _roic(x):
        oi=_f(getattr(x,"operating_income",None)); eq=_f(getattr(x,"total_equity",None)) or 0
        debt=_f(getattr(x,"long_term_debt",None)) or 0; cash=_f(getattr(x,"cash",None)) or 0
        te=_f(getattr(x,"tax_expense",None)); pt=_f(getattr(x,"pretax_income",None))
        tr=(min(max(te/pt,0),0.35) if te is not None and pt and pt>0 else 0.21)
        ic=eq+debt-cash
        return (oi*(1-tr))/ic if (oi is not None and ic>0) else None
    roics=[r for r in (_roic(x) for x in q) if r is not None]
    if roics:
        f["roic"]=sum(roics[-4:])/len(roics[-4:])
        if wacc is not None: f["roic_wacc_spread"]=f["roic"]-wacc
        st=_cov_stability(roics)
        if st is not None: f["roic_stability"]=st
        s=_slope(roics)
        if s is not None: f["roic_trend"]=s
    if ni_ttm is not None:
        eq=_f(getattr(latest,"total_equity",None)); ta=_f(getattr(latest,"total_assets",None))
        if eq and eq>0: f["roe"]=ni_ttm/eq
        if ta and ta>0: f["roa"]=ni_ttm/ta

    if ni_ttm and ocf_ttm is not None:
        f["ocf_to_net_income"]=ocf_ttm/ni_ttm; f["earnings_quality_ratio"]=ocf_ttm/ni_ttm
        if capex_ttm is not None and ni_ttm!=0: f["fcf_conversion"]=(ocf_ttm-abs(capex_ttm))/ni_ttm
    ocf_annual=_annual_series(q,"operating_cash_flow")
    if len(ocf_annual)>=2 and ocf_annual[-2]: f["ocf_growth"]=ocf_annual[-1]/ocf_annual[-2]-1.0
    fcf_series=[]
    for x in q:
        o=_f(getattr(x,"operating_cash_flow",None)); c=_f(getattr(x,"capex",None))
        if o is not None and c is not None: fcf_series.append(o-abs(c))
    if len(fcf_series)>=4:
        st=_cov_stability([sum(fcf_series[i-3:i+1]) for i in range(3,len(fcf_series))])
        if st is not None: f["fcf_stability"]=st
        f["cash_gen_consistency"]=sum(1 for v in fcf_series if v>0)/len(fcf_series)

    if len(ni_annual)>=2 and ni_annual[-2]: f["net_income_growth"]=ni_annual[-1]/ni_annual[-2]-1.0
    oi_annual=_annual_series(q,"operating_income")
    if len(oi_annual)>=2 and oi_annual[-2]: f["ebit_growth"]=oi_annual[-1]/oi_annual[-2]-1.0
    gp_annual=_annual_series(q,"gross_profit")
    if len(gp_annual)>=2 and gp_annual[-2]: f["gross_profit_growth"]=gp_annual[-1]/gp_annual[-2]-1.0
    eps=[_f(getattr(x,"eps_diluted",None)) for x in q]; eps=[e for e in eps if e is not None]
    eps_annual=[sum(eps[i-3:i+1]) for i in range(3,len(eps))] if len(eps)>=4 else []
    if len(eps_annual)>=2 and eps_annual[-2]: f["eps_growth"]=eps_annual[-1]/eps_annual[-2]-1.0
    if len(eps_annual)>=4 and eps_annual[-4]>0 and eps_annual[-1]>0:
        c=_cagr(eps_annual[-4],eps_annual[-1],3)
        if c is not None: f["eps_cagr_3y"]=c
    st=_cov_stability(eps_annual)
    if st is not None: f["eps_stability"]=st

    ca=_f(getattr(latest,"current_assets",None)); cl=_f(getattr(latest,"current_liabilities",None))
    cash=_f(getattr(latest,"cash",None)); debt=_f(getattr(latest,"long_term_debt",None))
    eq=_f(getattr(latest,"total_equity",None)); ta=_f(getattr(latest,"total_assets",None))
    if ca and cl and cl>0:
        f["current_ratio"]=ca/cl; f["quick_ratio"]=ca/cl
        if cash is not None: f["cash_ratio"]=cash/cl
        if ocf_ttm is not None: f["operating_cash_ratio"]=ocf_ttm/cl
    if debt is not None and eq and eq>0: f["debt_to_equity"]=debt/eq
    if oi_ttm and oi_ttm>0 and debt is not None:
        f["debt_to_ebitda"]=debt/oi_ttm
        if cash is not None: f["net_debt_to_ebitda"]=(debt-cash)/oi_ttm
    if ta and ta>0 and rev_ttm: f["asset_turnover"]=rev_ttm/ta
    tas=[_f(getattr(x,"total_assets",None)) for x in q]; tas=[a for a in tas if a is not None]
    if len(tas)>=5 and tas[-5]: f["asset_growth"]=tas[-1]/tas[-5]-1.0

    if ni_ttm is not None and ocf_ttm is not None and ta and ta>0:
        f["accruals_ratio"]=(ni_ttm-ocf_ttm)/ta; f["sloan_ratio"]=(ni_ttm-ocf_ttm)/ta
        f["cash_earnings_gap"]=abs(ni_ttm-ocf_ttm)/(abs(ni_ttm) if ni_ttm else 1)
    return f
