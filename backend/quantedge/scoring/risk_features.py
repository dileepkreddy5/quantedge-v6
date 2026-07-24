"""Risk Intelligence — credit models (Altman-Z, Ohlson-O, Beneish M-score), leverage,
liquidity, earnings-quality, structural tail risk, business-model, valuation, governance,
macro sensitivity. From price history + financials. No fakes."""
import math, statistics as st

def _f(v):
    try: v=float(v); return v if math.isfinite(v) else None
    except: return None
def _sd(a,b):
    a=_f(a); b=_f(b)
    return a/b if (a is not None and b not in (None,0)) else None

def compute_risk_features(merged, price_closes=None, market_cap=None, beta=None):
    f={}
    if not merged or len(merged)<4: return {"available":False}
    # q = latest quarter with core data; qy = ~4 quarters before that
    _real=[x for x in merged if _f(x.get("assets")) is not None]
    q=_real[-1] if _real else merged[-1]
    qy=_real[-5] if len(_real)>=5 else (_real[0] if _real else merged[0])
    def cur(k):
        # point-in-time: latest quarter that actually has this field (skip stub quarters)
        for x in reversed(merged):
            v=_f(x.get(k))
            if v is not None: return v
        return None
    def ttm(k):
        # sum the most recent 4 quarters that HAVE data for this field (robust to stub latest quarter)
        vals=[_f(x.get(k)) for x in merged]
        vals=[v for v in vals if v is not None]
        return sum(vals[-4:]) if len(vals)>=4 else (sum(vals) if vals else None)

    def ttm_prior(k):
        """The four quarters before the current TTM window.

        A fixed [-8:-4] slice breaks whenever a null lands inside it — Coca-Cola's
        latest quarter is a stub and NVIDIA has a gap four quarters back — so the
        year-ago comparison silently vanished for exactly the companies where the
        window is ragged. Drop nulls first, as ttm() does, then step back four.
        """
        vals=[_f(x.get(k)) for x in merged]
        vals=[v for v in vals if v is not None]
        return sum(vals[-8:-4]) if len(vals)>=8 else None
    assets=cur("assets"); liab=cur("liabilities"); eq=cur("equity")
    ca=cur("current_assets"); cl=cur("current_liabilities")
    ni_ttm=ttm("net_income"); rev_ttm=ttm("revenue"); ebit_ttm=ttm("operating_income")
    ocf_ttm=ttm("operating_cash_flow"); re=cur("retained_earnings")
    ltd_raw=cur("long_term_debt"); std_raw=cur("short_term_debt")
    if ltd_raw is None and std_raw is None:
        total_debt=None  # genuinely missing - do NOT treat as zero-debt (that would score as "safe")
    else:
        total_debt=(ltd_raw or 0)+(std_raw or 0)
    cash=cur("cash"); wc=(ca-cl) if (ca is not None and cl is not None) else None
    da_ttm=ttm("depreciation_amortization"); ebitda_ttm=(ebit_ttm+da_ttm) if (ebit_ttm and da_ttm) else None

    if all(x is not None for x in [wc,assets,re,ebit_ttm,liab,rev_ttm]) and assets>0 and liab>0:
        mve=market_cap or eq
        z=(1.2*wc/assets+1.4*re/assets+3.3*ebit_ttm/assets+0.6*(mve/liab if mve else 0)+1.0*rev_ttm/assets)
        f["altman_z"]=round(z,2)
    if all(x is not None for x in [assets,liab,wc,ni_ttm,ocf_ttm]) and assets>0:
        tl_ta=liab/assets; wc_ta=wc/assets; ni_ta=ni_ttm/assets; ocf_tl=(ocf_ttm/liab if liab>0 else 0)
        o=-1.32-0.407*math.log(max(assets,1)/1e6)+6.03*tl_ta-1.43*wc_ta-2.37*ni_ta-1.83*ocf_tl
        f["ohlson_o"]=round(o,2); f["bankruptcy_prob"]=round(1/(1+math.exp(-o)),3)
    fscore=0; checks=0
    if ni_ttm is not None: fscore+=1 if ni_ttm>0 else 0; checks+=1
    if ocf_ttm is not None: fscore+=1 if ocf_ttm>0 else 0; checks+=1
    if ocf_ttm is not None and ni_ttm is not None: fscore+=1 if ocf_ttm>ni_ttm else 0; checks+=1
    if checks>0: f["piotroski_partial"]=fscore
    f["ebit_to_debt"]=_sd(ebit_ttm,total_debt) if (total_debt is not None and total_debt>0) else None
    f["debt_to_assets"]=_sd(total_debt,assets)

    f["debt_to_equity"]=_sd(total_debt,eq)
    f["net_debt_to_ebitda"]=_sd((total_debt-(cash or 0)),ebitda_ttm) if total_debt is not None else None
    f["liabilities_to_equity"]=_sd(liab,eq)
    f["debt_to_ebitda"]=_sd(total_debt,ebitda_ttm)
    f["equity_ratio"]=_sd(eq,assets)
    # Compare like with like. This previously measured current TOTAL debt
    # against year-ago LONG-TERM debt, so any company carrying short-term
    # borrowings showed a leverage increase that was purely definitional.
    _ltd_y=_f(qy.get("long_term_debt")) or 0.0
    _std_y=_f(qy.get("short_term_debt")) or 0.0
    total_debt_y=(_ltd_y+_std_y) if (qy.get("long_term_debt") is not None
                                     or qy.get("short_term_debt") is not None) else None
    assets_y=_f(qy.get("assets"))
    if total_debt_y is not None and assets_y and assets and total_debt is not None:
        f["leverage_trend"]=(total_debt/assets)-(total_debt_y/assets_y)

    f["current_ratio"]=_sd(ca,cl)
    f["quick_ratio"]=_sd((ca-_f(q.get("inventory") or 0)) if ca else None, cl)
    f["cash_ratio"]=_sd(cash,cl)
    f["cash_to_debt"]=_sd(cash,total_debt) if (total_debt is not None and total_debt>0) else None
    if ocf_ttm is not None and cash is not None:
        q_burn=-ocf_ttm/4 if ocf_ttm<0 else None
        f["cash_runway_qtrs"]=round(cash/q_burn,1) if q_burn and q_burn>0 else None
    f["working_capital_ratio"]=_sd(wc,assets)
    opex_ttm=ttm("operating_expenses") or (rev_ttm-ebit_ttm if (rev_ttm and ebit_ttm) else None)
    if ca is not None and opex_ttm and opex_ttm>0:
        f["defensive_interval_days"]=round(ca/(opex_ttm/365),0)

    if ni_ttm is not None and ocf_ttm is not None and assets:
        f["sloan_accruals"]=(ni_ttm-ocf_ttm)/assets
    f["cash_conversion"]=_sd(ocf_ttm,ni_ttm)
    nis=[_f(x.get("net_income")) for x in merged[-8:]]; nis=[x for x in nis if x is not None]
    if len(nis)>=4 and st.mean(nis)!=0:
        f["earnings_volatility"]=abs(st.pstdev(nis)/st.mean(nis))
    # DSRI compares receivables-to-sales now against a year ago, so both sides
    # must use the same span. This divided current receivables by TTM revenue and
    # year-ago receivables by a SINGLE quarter's revenue, building a factor of
    # about four into every reading — Coca-Cola came out at 0.181 where the index
    # centres on 1.0, and since the scale treats anything under 1.0 as pristine,
    # the error read as perfect accounting quality.
    rec = cur("receivables")
    rev_ttm_y = ttm_prior("revenue")
    rec_y = _f(qy.get("receivables"))
    if rec and rev_ttm and rec_y and rev_ttm_y and rev_ttm_y > 0:
        dsri = (rec / rev_ttm) / (rec_y / rev_ttm_y)
        f["beneish_dsri"] = round(dsri, 3) if dsri else None
    # GMI has the same span mismatch DSRI had: gross profit over a single
    # year-ago quarter against gross profit over TTM. Both sides now use TTM.
    # cur() returns the latest QUARTER while rev_ttm covers four, so this divided
    # one quarter's gross profit by a year of revenue and reported a margin a
    # quarter of its true size — inverted, that gave Coca-Cola a GMI of 3.6 where
    # the index centres on 1.0. Both sides now TTM.
    gp_ttm = ttm("gross_profit")
    gp_ttm_y = ttm_prior("gross_profit")
    if gp_ttm and rev_ttm and gp_ttm_y and rev_ttm_y:
        gmi = (gp_ttm_y / rev_ttm_y) / (gp_ttm / rev_ttm)
        f["beneish_gmi"] = round(gmi, 3) if gmi else None
    # Same span mismatch a third time: TTM revenue against a single year-ago
    # quarter reports roughly 300% growth for a flat business, and the same for
    # operating cash flow, so the divergence between them was noise.
    ocf_ttm_y = ttm_prior("operating_cash_flow")
    if rev_ttm and rev_ttm_y and ocf_ttm is not None and ocf_ttm_y:
        rev_g = rev_ttm / rev_ttm_y - 1
        ocf_g = (ocf_ttm / ocf_ttm_y - 1) if ocf_ttm_y > 0 else 0
        f["revenue_ocf_divergence"] = rev_g - ocf_g

    if price_closes and len(price_closes)>=120:
        rets=[(price_closes[i]/price_closes[i-1]-1) for i in range(1,len(price_closes))]
        srt=sorted(rets); n=len(srt)
        f["cvar_5pct"]=round(st.mean(srt[:max(1,int(0.05*n))]),4)
        f["worst_day"]=round(min(rets),4)
        peak=price_closes[0]; maxdd=0; maxdur=0; cur_start=0
        for i,p in enumerate(price_closes):
            if p>peak: peak=p; cur_start=i
            dd=(p-peak)/peak
            if dd<maxdd: maxdd=dd
            if p<peak: maxdur=max(maxdur,i-cur_start)
        f["max_drawdown"]=round(maxdd,4); f["drawdown_duration_days"]=maxdur
        if len(price_closes)>=21:
            f["worst_month"]=round(min((price_closes[i]/price_closes[i-21]-1) for i in range(21,len(price_closes))),4)
        f["annualized_vol"]=round(st.pstdev(rets)*math.sqrt(252),4)
        downs=[r for r in rets if r<0]
        f["downside_deviation"]=round(st.pstdev(downs)*math.sqrt(252),4) if len(downs)>2 else None
        sq=[r*r for r in rets]
        if len(sq)>10:
            m=st.mean(sq); num=sum((sq[i]-m)*(sq[i-1]-m) for i in range(1,len(sq))); den=sum((x-m)**2 for x in sq)
            f["vol_clustering"]=round(num/den,3) if den>0 else None
        m=st.mean(rets); f["semivariance"]=round(sum((r-m)**2 for r in rets if r<m)/len(rets)*252,4)
    if beta is not None:
        f["beta"]=beta
        # Score on distance from zero, not on the signed value. With hib=False
        # a raw beta scored -0.25 as safer than +0.80, and would score -2.0 —
        # violently anticorrelated, genuinely dangerous — as the safest of all.
        # What carries systematic risk is magnitude; sign only says direction.
        f["abs_beta"]=abs(beta)
        # market_sensitivity duplicated abs_beta, which the beta signal already
        # scores. Two signals, one number, in two categories.

    f["operating_margin"]=_sd(ebit_ttm,rev_ttm)
    # margin_cushion repeated operating_margin exactly (ebit/revenue).
    f["capital_intensity"]=_sd(ttm("capex"),rev_ttm)
    f["asset_turnover"]=_sd(rev_ttm,assets)

    if market_cap and ni_ttm and ni_ttm>0:
        pe=market_cap/ni_ttm; f["pe_ratio"]=round(pe,1); f["earnings_yield"]=ni_ttm/market_cap
        f["valuation_downside_risk"]=min(1.0,pe/50) if pe>0 else None
    if market_cap and eq: f["price_to_book"]=_sd(market_cap,eq)

    sh=cur("diluted_shares"); sh_y=_f(qy.get("diluted_shares"))
    if sh and sh_y and sh_y>0: f["share_dilution"]=(sh/sh_y-1)
    sbc_ttm=ttm("sbc")
    if sbc_ttm is not None and rev_ttm and rev_ttm>0: f["sbc_intensity"]=sbc_ttm/rev_ttm
    # A company with no buybacks is not missing data — it simply did not do the
    # thing this signal looks for, which is the good outcome. Requiring a
    # truthy buyback figure left the signal blank for every such company.
    if f.get("leverage_trend") is not None:
        bb=ttm("buybacks") or 0.0
        f["debt_funded_buyback_flag"]=1.0 if (f["leverage_trend"]>0.02 and bb>0) else 0.0

    if f.get("debt_to_assets") is not None: f["rate_sensitivity"]=f["debt_to_assets"]
    # cyclicality_proxy was a copy of earnings_volatility, already scored
    # under Earnings Quality. Cyclicality is not the same thing as earnings
    # variability, so asserting it from that number was wrong twice over.

    return {k:v for k,v in f.items() if v is not None}
