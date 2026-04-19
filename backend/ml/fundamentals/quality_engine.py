"""
QuantEdge v6.0 — Fundamentals Quality Engine
=============================================
Computes the "Past Score" (0-100) — is this a good business?

Uses Polygon /vX/reference/financials endpoint to pull 10+ years of
quarterly financials, then computes institutional-grade quality metrics:

  1. ROIC 10yr trend (Buffett's #1 metric)
  2. Gross margin stability (pricing power)
  3. FCF conversion (earnings quality)
  4. Revenue growth consistency
  5. Debt trajectory
  6. Piotroski F-Score (9-point fundamental screen)
  7. Altman Z-Score (bankruptcy risk)
  8. Sloan accrual ratio (earnings manipulation detector)

Reference: Greenblatt (2005), Piotroski (2000), Altman (1968), Sloan (1996)
"""

import os
import asyncio
import aiohttp
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from loguru import logger


POLYGON_BASE = "https://api.polygon.io"


@dataclass
class QuarterlyFinancials:
    fiscal_period: str
    fiscal_year: int
    filing_date: Optional[str]
    period_end: Optional[str]
    revenue: Optional[float] = None
    gross_profit: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    eps_diluted: Optional[float] = None
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    total_liabilities: Optional[float] = None
    total_equity: Optional[float] = None
    long_term_debt: Optional[float] = None
    cash: Optional[float] = None
    operating_cash_flow: Optional[float] = None
    capex: Optional[float] = None

    @property
    def free_cash_flow(self) -> Optional[float]:
        if self.operating_cash_flow is not None and self.capex is not None:
            return self.operating_cash_flow + self.capex
        return None

    @property
    def gross_margin(self) -> Optional[float]:
        if self.revenue and self.gross_profit and self.revenue > 0:
            return self.gross_profit / self.revenue
        return None

    @property
    def operating_margin(self) -> Optional[float]:
        if self.revenue and self.operating_income and self.revenue > 0:
            return self.operating_income / self.revenue
        return None

    @property
    def net_margin(self) -> Optional[float]:
        if self.revenue and self.net_income and self.revenue > 0:
            return self.net_income / self.revenue
        return None

    @property
    def current_ratio(self) -> Optional[float]:
        if self.current_assets and self.current_liabilities and self.current_liabilities > 0:
            return self.current_assets / self.current_liabilities
        return None

    @property
    def debt_to_equity(self) -> Optional[float]:
        if self.long_term_debt is not None and self.total_equity and self.total_equity > 0:
            return self.long_term_debt / self.total_equity
        return None

    @property
    def roic(self) -> Optional[float]:
        if self.operating_income is None:
            return None
        invested = (self.total_equity or 0) + (self.long_term_debt or 0)
        if invested <= 0:
            return None
        return self.operating_income / invested

    @property
    def roe(self) -> Optional[float]:
        if self.net_income is not None and self.total_equity and self.total_equity > 0:
            return self.net_income / self.total_equity
        return None

    @property
    def roa(self) -> Optional[float]:
        if self.net_income is not None and self.total_assets and self.total_assets > 0:
            return self.net_income / self.total_assets
        return None


@dataclass
class QualityScorecard:
    ticker: str
    past_score: float
    sub_scores: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    trends: Dict[str, List[float]] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    piotroski_f_score: Optional[int] = None
    altman_z_score: Optional[float] = None
    n_quarters_used: int = 0
    data_quality: str = "unknown"


def _safe(d: dict, key: str) -> Optional[float]:
    if not d or key not in d:
        return None
    v = d[key]
    if isinstance(v, dict):
        v = v.get("value")
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


async def fetch_quarterly_financials(
    ticker: str,
    api_key: str,
    limit: int = 40,
    session: Optional[aiohttp.ClientSession] = None,
) -> List[QuarterlyFinancials]:
    url = f"{POLYGON_BASE}/vX/reference/financials"
    params = {
        "ticker": ticker.upper(),
        "timeframe": "quarterly",
        "order": "desc",
        "limit": limit,
        "sort": "period_of_report_date",
        "apiKey": api_key,
    }
    close_session = False
    if session is None:
        session = aiohttp.ClientSession()
        close_session = True
    try:
        async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=20)) as resp:
            if resp.status != 200:
                logger.warning(f"Polygon financials returned {resp.status} for {ticker}")
                return []
            data = await resp.json()
    except Exception as e:
        logger.warning(f"Polygon financials fetch failed for {ticker}: {e}")
        return []
    finally:
        if close_session:
            await session.close()

    results = data.get("results", [])
    if not results:
        return []

    quarters: List[QuarterlyFinancials] = []
    for r in results:
        fin = r.get("financials", {}) or {}
        income = fin.get("income_statement", {}) or {}
        balance = fin.get("balance_sheet", {}) or {}
        cashflow = fin.get("cash_flow_statement", {}) or {}
        q = QuarterlyFinancials(
            fiscal_period=r.get("fiscal_period", ""),
            fiscal_year=int(r.get("fiscal_year", 0) or 0),
            filing_date=r.get("filing_date"),
            period_end=r.get("end_date") or r.get("period_of_report_date"),
            revenue=_safe(income, "revenues"),
            gross_profit=_safe(income, "gross_profit"),
            operating_income=_safe(income, "operating_income_loss"),
            net_income=_safe(income, "net_income_loss"),
            eps_diluted=_safe(income, "diluted_earnings_per_share"),
            total_assets=_safe(balance, "assets"),
            current_assets=_safe(balance, "current_assets"),
            current_liabilities=_safe(balance, "current_liabilities"),
            total_liabilities=_safe(balance, "liabilities"),
            total_equity=_safe(balance, "equity"),
            long_term_debt=_safe(balance, "long_term_debt"),
            cash=_safe(balance, "cash"),
            operating_cash_flow=_safe(cashflow, "net_cash_flow_from_operating_activities"),
            capex=_safe(cashflow, "net_cash_flow_from_investing_activities"),
        )
        quarters.append(q)
    quarters.reverse()
    return quarters


def _safe_series(values: List[Optional[float]]) -> pd.Series:
    return pd.Series(
        [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))],
        dtype=np.float64,
    )


def compute_roic_trend(quarters: List[QuarterlyFinancials]) -> Dict[str, float]:
    roics = _safe_series([q.roic for q in quarters])
    if len(roics) < 4:
        return {"mean": np.nan, "std": np.nan, "latest": np.nan, "slope": np.nan, "n": len(roics)}
    mean = float(roics.tail(20).mean())
    std = float(roics.tail(20).std())
    latest = float(roics.iloc[-1])
    if len(roics) >= 8:
        x = np.arange(len(roics), dtype=np.float64)
        slope = float(np.polyfit(x, roics.values, 1)[0])
    else:
        slope = 0.0
    return {"mean": mean, "std": std, "latest": latest, "slope": slope, "n": len(roics)}


def compute_margin_stability(quarters: List[QuarterlyFinancials]) -> Dict[str, float]:
    gms = _safe_series([q.gross_margin for q in quarters])
    if len(gms) < 8:
        return {"mean": np.nan, "cov": np.nan, "n": len(gms)}
    mean = float(gms.tail(20).mean())
    std = float(gms.tail(20).std())
    cov = std / mean if mean > 0 else np.nan
    return {"mean": mean, "cov": cov, "n": len(gms)}


def compute_fcf_conversion(quarters: List[QuarterlyFinancials]) -> Dict[str, float]:
    ratios = []
    for q in quarters:
        fcf = q.free_cash_flow
        ni = q.net_income
        if fcf is not None and ni is not None and ni > 0:
            ratios.append(fcf / ni)
    s = _safe_series(ratios)
    if len(s) < 4:
        return {"mean": np.nan, "n": len(s)}
    return {"mean": float(s.tail(20).mean()), "n": len(s)}


def compute_revenue_growth_consistency(quarters: List[QuarterlyFinancials]) -> Dict[str, float]:
    growths = []
    for i in range(4, len(quarters)):
        curr = quarters[i].revenue
        prior = quarters[i - 4].revenue
        if curr and prior and prior > 0:
            growths.append((curr - prior) / prior)
    s = _safe_series(growths)
    if len(s) < 4:
        return {"mean": np.nan, "std": np.nan, "n": len(s)}
    return {"mean": float(s.mean()), "std": float(s.std()), "n": len(s)}


def compute_debt_trend(quarters: List[QuarterlyFinancials]) -> Dict[str, float]:
    d2es = _safe_series([q.debt_to_equity for q in quarters])
    if len(d2es) < 4:
        return {"latest": np.nan, "slope": np.nan, "n": len(d2es)}
    latest = float(d2es.iloc[-1])
    if len(d2es) >= 8:
        x = np.arange(len(d2es), dtype=np.float64)
        slope = float(np.polyfit(x, d2es.values, 1)[0])
    else:
        slope = 0.0
    return {"latest": latest, "slope": slope, "n": len(d2es)}


def compute_piotroski_f_score(quarters: List[QuarterlyFinancials]) -> Optional[int]:
    if len(quarters) < 8:
        return None

    def ttm(qs, fg):
        vals = [fg(q) for q in qs[-4:]]
        vals = [v for v in vals if v is not None]
        if len(vals) < 4:
            return None
        return sum(vals)

    def latest(qs, fg):
        for q in reversed(qs):
            v = fg(q)
            if v is not None:
                return v
        return None

    curr = quarters[-4:]
    prior = quarters[-8:-4]
    if len(prior) < 4:
        return None

    ni_curr = ttm(curr, lambda q: q.net_income)
    ni_prior = ttm(prior, lambda q: q.net_income)
    ocf_curr = ttm(curr, lambda q: q.operating_cash_flow)
    assets_curr = latest(curr, lambda q: q.total_assets)
    assets_prior = latest(prior, lambda q: q.total_assets)
    ltd_curr = latest(curr, lambda q: q.long_term_debt)
    ltd_prior = latest(prior, lambda q: q.long_term_debt)
    cr_curr = latest(curr, lambda q: q.current_ratio)
    cr_prior = latest(prior, lambda q: q.current_ratio)
    rev_curr = ttm(curr, lambda q: q.revenue)
    rev_prior = ttm(prior, lambda q: q.revenue)
    gp_curr = ttm(curr, lambda q: q.gross_profit)
    gp_prior = ttm(prior, lambda q: q.gross_profit)

    score = 0
    if ni_curr is not None and ni_curr > 0:
        score += 1
    if ocf_curr is not None and ocf_curr > 0:
        score += 1
    if ni_curr is not None and assets_curr and ni_prior is not None and assets_prior:
        if (ni_curr / assets_curr) > (ni_prior / assets_prior):
            score += 1
    if ocf_curr is not None and ni_curr is not None and ocf_curr > ni_curr:
        score += 1
    if ltd_curr is not None and ltd_prior is not None and ltd_curr < ltd_prior:
        score += 1
    if cr_curr is not None and cr_prior is not None and cr_curr > cr_prior:
        score += 1
    if gp_curr and rev_curr and gp_prior and rev_prior and rev_curr > 0 and rev_prior > 0:
        if (gp_curr / rev_curr) > (gp_prior / rev_prior):
            score += 1
    if rev_curr and assets_curr and rev_prior and assets_prior:
        if (rev_curr / assets_curr) > (rev_prior / assets_prior):
            score += 1
    return score


def compute_altman_z_score(
    quarters: List[QuarterlyFinancials], market_cap: Optional[float] = None
) -> Optional[float]:
    if not quarters or len(quarters) < 4:
        return None
    q = quarters[-1]
    ta = q.total_assets
    if not ta or ta <= 0:
        return None
    wc = (q.current_assets or 0) - (q.current_liabilities or 0)
    A = wc / ta
    ttm_ni = sum(x.net_income for x in quarters[-4:] if x.net_income is not None) if len(quarters) >= 4 else 0
    B = ttm_ni / ta if ta > 0 else 0
    ttm_oi = sum(x.operating_income for x in quarters[-4:] if x.operating_income is not None)
    C = ttm_oi / ta if ta > 0 else 0
    tl = q.total_liabilities or 1
    if market_cap and market_cap > 0:
        D = market_cap / tl
    elif q.total_equity:
        D = q.total_equity / tl
    else:
        D = 0
    ttm_rev = sum(x.revenue for x in quarters[-4:] if x.revenue is not None)
    E = ttm_rev / ta if ta > 0 else 0
    z = 1.2 * A + 1.4 * B + 3.3 * C + 0.6 * D + 1.0 * E
    return float(z)


def compute_accrual_ratio(quarters: List[QuarterlyFinancials]) -> Optional[float]:
    if not quarters:
        return None
    recent = quarters[-4:]
    if len(recent) < 4:
        return None
    ni = sum(q.net_income for q in recent if q.net_income is not None)
    ocf = sum(q.operating_cash_flow for q in recent if q.operating_cash_flow is not None)
    ta = recent[-1].total_assets
    if not ta or ta <= 0:
        return None
    return (ni - ocf) / ta


def score_from_percentile(
    value: float, good: float, great: float, higher_is_better: bool = True
) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 50.0
    if higher_is_better:
        if value >= great:
            return 100.0
        if value <= good - (great - good):
            return 0.0
        frac = (value - good) / (great - good) if great != good else 0.0
        return float(np.clip(50 + 50 * frac, 0, 100))
    else:
        if value <= great:
            return 100.0
        if value >= good + (good - great):
            return 0.0
        frac = (good - value) / (good - great) if good != great else 0.0
        return float(np.clip(50 + 50 * frac, 0, 100))


def build_scorecard(
    ticker: str,
    quarters: List[QuarterlyFinancials],
    market_cap: Optional[float] = None,
) -> QualityScorecard:
    if not quarters:
        return QualityScorecard(
            ticker=ticker, past_score=50.0,
            weaknesses=["No financial data available"],
            data_quality="insufficient", n_quarters_used=0,
        )

    n = len(quarters)
    roic = compute_roic_trend(quarters)
    margin = compute_margin_stability(quarters)
    fcf = compute_fcf_conversion(quarters)
    rev_growth = compute_revenue_growth_consistency(quarters)
    debt = compute_debt_trend(quarters)
    f_score = compute_piotroski_f_score(quarters)
    z_score = compute_altman_z_score(quarters, market_cap)
    accruals = compute_accrual_ratio(quarters)

    sub_scores = {
        "roic": score_from_percentile(roic.get("mean", np.nan), good=0.08, great=0.20, higher_is_better=True),
        "margin_stability": score_from_percentile(margin.get("cov", np.nan), good=0.15, great=0.05, higher_is_better=False),
        "fcf_conversion": score_from_percentile(fcf.get("mean", np.nan), good=0.8, great=1.2, higher_is_better=True),
        "revenue_growth": score_from_percentile(rev_growth.get("mean", np.nan), good=0.05, great=0.15, higher_is_better=True),
        "debt_trajectory": score_from_percentile(debt.get("latest", np.nan), good=1.0, great=0.3, higher_is_better=False),
        "piotroski": (f_score / 9.0 * 100.0) if f_score is not None else 50.0,
        "altman_z": score_from_percentile(z_score if z_score is not None else np.nan, good=1.8, great=3.0, higher_is_better=True),
        "earnings_quality": score_from_percentile(accruals if accruals is not None else np.nan, good=0.05, great=-0.05, higher_is_better=False),
    }

    weights = {
        "roic": 0.20, "margin_stability": 0.10, "fcf_conversion": 0.15,
        "revenue_growth": 0.10, "debt_trajectory": 0.10, "piotroski": 0.15,
        "altman_z": 0.10, "earnings_quality": 0.10,
    }
    past_score = sum(sub_scores[k] * w for k, w in weights.items())
    past_score = float(np.clip(past_score, 0, 100))

    strengths, weaknesses = [], []
    for k, v in sub_scores.items():
        if v >= 75:
            strengths.append(k)
        elif v <= 35:
            weaknesses.append(k)

    data_quality = (
        "excellent" if n >= 30 else
        "good" if n >= 16 else
        "partial" if n >= 8 else
        "insufficient"
    )

    return QualityScorecard(
        ticker=ticker,
        past_score=round(past_score, 1),
        sub_scores={k: round(v, 1) for k, v in sub_scores.items()},
        metrics={
            "roic_mean_5y": round(roic.get("mean", 0) or 0, 4),
            "roic_trend": round(roic.get("slope", 0) or 0, 6),
            "gross_margin_mean": round(margin.get("mean", 0) or 0, 4),
            "gross_margin_cov": round(margin.get("cov", 0) or 0, 4),
            "fcf_conversion": round(fcf.get("mean", 0) or 0, 4),
            "revenue_growth_mean": round(rev_growth.get("mean", 0) or 0, 4),
            "revenue_growth_std": round(rev_growth.get("std", 0) or 0, 4),
            "debt_to_equity_latest": round(debt.get("latest", 0) or 0, 4),
            "debt_trend": round(debt.get("slope", 0) or 0, 6),
            "accrual_ratio": round(accruals or 0, 4),
        },
        piotroski_f_score=f_score,
        altman_z_score=round(z_score, 2) if z_score is not None else None,
        strengths=strengths,
        weaknesses=weaknesses,
        n_quarters_used=n,
        data_quality=data_quality,
    )


class QualityEngine:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("POLYGON_API_KEY", "")
        if not self.api_key:
            logger.warning("QualityEngine: no POLYGON_API_KEY set")

    async def analyze(
        self,
        ticker: str,
        market_cap: Optional[float] = None,
        n_quarters: int = 40,
    ) -> QualityScorecard:
        if not self.api_key:
            return QualityScorecard(
                ticker=ticker, past_score=50.0,
                weaknesses=["API key missing"],
                data_quality="insufficient",
            )
        quarters = await fetch_quarterly_financials(ticker, self.api_key, limit=n_quarters)
        return build_scorecard(ticker, quarters, market_cap)


async def _test(ticker: str = "AAPL"):
    import sys
    api_key = os.getenv("POLYGON_API_KEY", "")
    if not api_key:
        print("ERROR: POLYGON_API_KEY env var not set")
        sys.exit(1)

    engine = QualityEngine(api_key=api_key)
    print(f"Analyzing {ticker}...")
    sc = await engine.analyze(ticker, n_quarters=40)

    print(f"\n{'='*60}")
    print(f"  QUALITY SCORECARD: {sc.ticker}")
    print(f"{'='*60}")
    print(f"  PAST SCORE (0-100):     {sc.past_score}")
    print(f"  Data quality:           {sc.data_quality} ({sc.n_quarters_used} quarters)")
    print(f"  Piotroski F-Score:      {sc.piotroski_f_score} / 9")
    print(f"  Altman Z-Score:         {sc.altman_z_score}")
    print(f"\n  Sub-scores:")
    for k, v in sc.sub_scores.items():
        bar = "#" * int(v / 5)
        print(f"    {k:<22} {v:>6.1f}  {bar}")
    print(f"\n  Key metrics:")
    for k, v in sc.metrics.items():
        print(f"    {k:<28} {v}")
    if sc.strengths:
        print(f"\n  Strengths:   {', '.join(sc.strengths)}")
    if sc.weaknesses:
        print(f"  Weaknesses:  {', '.join(sc.weaknesses)}")


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    asyncio.run(_test(ticker))
