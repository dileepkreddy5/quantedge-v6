"""Industry Intelligence — sector classification, sector-relative performance, sector
momentum/cycle, industry position (peer bucket percentiles), RS leadership, sector risk,
maturity. From SIC + sector ETF prices + peer_stats bucket. Technical/relative = real."""
import math, statistics as st
from datetime import datetime, date

BUCKET_ETF={"Technology":("XLK","Technology"),"Financials":("XLF","Financials"),
    "Financial":("XLF","Financials"),"Healthcare":("XLV","Healthcare"),"Health Care":("XLV","Healthcare"),
    "Energy":("XLE","Energy"),"Industrials":("XLI","Industrials"),"Consumer Discretionary":("XLY","Consumer Discretionary"),
    "Consumer Staples":("XLP","Consumer Staples"),"Utilities":("XLU","Utilities"),"Materials":("XLB","Materials"),
    "Real Estate":("XLRE","Real Estate"),"Communication Services":("XLC","Communications"),
    "Communications":("XLC","Communications")}

def bucket_to_etf(bucket):
    if not bucket: return None
    return BUCKET_ETF.get(bucket)

def sic_to_sector_etf(sic):
    if not sic: return "SPY","Unknown"
    try: s=int(sic)
    except: return "SPY","Unknown"
    if 100<=s<1000: return "XLB","Materials/Agriculture"
    if 1000<=s<1500: return "XLE","Energy/Mining"
    if 1500<=s<1800: return "XLI","Construction/Industrials"
    if 2000<=s<2400: return "XLP","Consumer Staples"
    if 2400<=s<2800: return "XLB","Materials"
    if 2830<=s<2840: return "XLV","Pharmaceuticals"
    if 2800<=s<2900: return "XLB","Chemicals"
    if 2900<=s<3000: return "XLE","Petroleum"
    if 3570<=s<3580 or 3670<=s<3700: return "XLK","Technology/Hardware"
    if 3000<=s<3600: return "XLI","Manufacturing/Industrials"
    if 3600<=s<3700: return "XLK","Electronics"
    if 3700<=s<3800: return "XLI","Transportation Equipment"
    if 3800<=s<4000: return "XLV","Instruments/Medical"
    if 4000<=s<4800: return "XLI","Transportation"
    if 4800<=s<4900: return "XLC","Communications"
    if 4900<=s<5000: return "XLU","Utilities"
    if 5000<=s<5200: return "XLY","Wholesale"
    if 5200<=s<6000: return "XLY","Retail"
    if 6000<=s<6500: return "XLF","Financials"
    if 6500<=s<6800: return "XLRE","Real Estate"
    if 7370<=s<7380: return "XLK","Computer Services"
    if 7000<=s<7370: return "XLK","Software/Services"
    if 7800<=s<8000: return "XLC","Entertainment"
    if 8000<=s<8100: return "XLV","Health Services"
    return "SPY","Diversified"

CYCLICAL_SECTORS={"XLE","XLB","XLI","XLF","XLY","XLRE"}
DEFENSIVE_SECTORS={"XLP","XLU","XLV"}
def _parse_factors(obj):
    """factors may be a JSON string (from JSONB) or already a dict."""
    fac=obj.get("factors") if isinstance(obj,dict) else None
    if isinstance(fac,str):
        import json
        try: return json.loads(fac)
        except: return {}
    return fac if isinstance(fac,dict) else {}

def _rets(closes): return [(closes[i]/closes[i-1]-1) for i in range(1,len(closes))] if len(closes)>1 else []

def compute_industry_features(sic, stock_closes, sector_closes, spy_closes,
                               market_cap=None, employees=None, list_date=None, peer_bucket=None):
    f={}
    etf,sector_name=sic_to_sector_etf(sic)
    f["_sector_etf"]=etf; f["_sector_name"]=sector_name

    if market_cap:
        f["market_cap_b"]=market_cap/1e9
        f["size_tier"]=3.0 if market_cap>1e11 else 2.0 if market_cap>1e10 else 1.0
    if employees:
        f["employee_scale"]=employees
        if market_cap: f["mcap_per_employee_m"]=market_cap/employees/1e6
    if peer_bucket and peer_bucket.get("available"):
        f["sector_peer_count"]=len(peer_bucket.get("peers",[]))

    if stock_closes and sector_closes and len(stock_closes)>=60 and len(sector_closes)>=60:
        def rs(days):
            if len(stock_closes)>days and len(sector_closes)>days:
                return (stock_closes[-1]/stock_closes[-days]-1)-(sector_closes[-1]/sector_closes[-days]-1)
            return None
        f["rs_sector_1m"]=rs(21); f["rs_sector_3m"]=rs(63); f["rs_sector_6m"]=rs(126)
        if len(stock_closes)>=252: f["rs_sector_1y"]=rs(252)
        sr=_rets(stock_closes[-126:]); er=_rets(sector_closes[-126:]); n=min(len(sr),len(er))
        if n>=30:
            sr=sr[-n:]; er=er[-n:]
            cov=sum((sr[i]-st.mean(sr))*(er[i]-st.mean(er)) for i in range(n))/n
            var_e=st.pvariance(er)
            f["beta_to_sector"]=cov/var_e if var_e>0 else None
            sd_s=st.pstdev(sr); sd_e=st.pstdev(er)
            f["correlation_to_sector"]=cov/(sd_s*sd_e) if (sd_s>0 and sd_e>0) else None
            up_s=[sr[i] for i in range(n) if er[i]>0]; up_e=[er[i] for i in range(n) if er[i]>0]
            dn_s=[sr[i] for i in range(n) if er[i]<0]; dn_e=[er[i] for i in range(n) if er[i]<0]
            if up_e and st.mean(up_e)!=0: f["up_capture"]=st.mean(up_s)/st.mean(up_e)
            if dn_e and st.mean(dn_e)!=0: f["down_capture"]=st.mean(dn_s)/st.mean(dn_e)

    if sector_closes and spy_closes and len(sector_closes)>=126 and len(spy_closes)>=126:
        f["sector_trend_3m"]=sector_closes[-1]/sector_closes[-63]-1
        f["sector_trend_6m"]=sector_closes[-1]/sector_closes[-126]-1
        f["sector_vs_spy_3m"]=(sector_closes[-1]/sector_closes[-63]-1)-(spy_closes[-1]/spy_closes[-63]-1)
        if len(sector_closes)>=252:
            lo=min(sector_closes[-252:]); hi=max(sector_closes[-252:])
            f["sector_52w_position"]=(sector_closes[-1]-lo)/(hi-lo) if hi>lo else 0.5
        f["sector_in_favor"]=1.0 if (f.get("sector_trend_3m",0)>0 and f.get("sector_vs_spy_3m",0)>0) else 0.0

    if peer_bucket and peer_bucket.get("available"):
        me=_parse_factors(peer_bucket.get("me",{})); peers=peer_bucket.get("peers",[])
        def pctile(factor):
            mv=me.get(factor)
            if mv is None: return None
            vals=[_parse_factors(p).get(factor) for p in peers]; vals=[v for v in vals if v is not None]
            if len(vals)<5: return None
            return sum(1 for v in vals if v<mv)/len(vals)
        # percentile vs sector peers on ALL available factors (real, from peer_stats)
        f["momentum_pctile_sector"]=pctile("mom_3m")
        f["sharpe_pctile_sector"]=pctile("sharpe_3m")
        f["quality_pctile_sector"]=pctile("quality_piotroski")
        f["mom12_pctile_sector"]=pctile("mom_12_1")
        f["vol_adj_pctile_sector"]=pctile("vol_adj_return")
        f["mom1m_pctile_sector"]=pctile("mom_1m")
        f["mom6m_pctile_sector"]=pctile("mom_6m")
        f["altman_pctile_sector"]=pctile("quality_altman_z")
        f["obv_pctile_sector"]=pctile("obv_slope_norm")
        f["ma50_pctile_sector"]=pctile("pct_above_ma50")
        f["ma200_pctile_sector"]=pctile("pct_above_ma200")
        f["hurst_pctile_sector"]=pctile("hurst")
        f["volsurge_pctile_sector"]=pctile("volume_surge")
        # composite sector rank (avg of available percentiles) + sector structure
        _pcts=[v for k,v in f.items() if k.endswith("_pctile_sector") and v is not None]
        if _pcts:
            f["composite_sector_rank"]=sum(_pcts)/len(_pcts)
            f["top_quartile_flags"]=sum(1 for p in _pcts if p>=0.75)/len(_pcts)  # share of factors top-quartile
            f["bottom_quartile_flags"]=sum(1 for p in _pcts if p<=0.25)/len(_pcts)
        # sector structure: dispersion of a key factor across peers (concentration)
        mom_vals=[_parse_factors(p).get("mom_3m") for p in peers]; mom_vals=[v for v in mom_vals if v is not None]
        if len(mom_vals)>=10:
            import statistics as _st
            f["sector_momentum_dispersion"]=_st.pstdev(mom_vals)
            f["sector_avg_momentum"]=_st.mean(mom_vals)
            me_mom=me.get("mom_3m")
            if me_mom is not None: f["momentum_vs_sector_avg"]=me_mom-_st.mean(mom_vals)

    if f.get("rs_sector_3m") is not None:
        f["is_sector_leader"]=1.0 if f["rs_sector_3m"]>0 else 0.0
        f["rs_magnitude"]=f["rs_sector_3m"]
    if f.get("momentum_pctile_sector") is not None:
        f["leadership_rank"]=f["momentum_pctile_sector"]

    if sector_closes and len(sector_closes)>=126:
        ser=_rets(sector_closes[-126:])
        f["sector_volatility"]=st.pstdev(ser)*math.sqrt(252)
        peak=sector_closes[-126]; maxdd=0
        for p in sector_closes[-126:]:
            if p>peak: peak=p
            maxdd=min(maxdd,(p-peak)/peak)
        f["sector_drawdown"]=maxdd
    f["is_cyclical"]=1.0 if etf in CYCLICAL_SECTORS else 0.0
    f["is_defensive"]=1.0 if etf in DEFENSIVE_SECTORS else 0.0
    if spy_closes and sector_closes and len(spy_closes)>=126:
        ser=_rets(sector_closes[-126:]); mr=_rets(spy_closes[-126:]); n=min(len(ser),len(mr))
        if n>=30:
            cov=sum((ser[i]-st.mean(ser[:n]))*(mr[i]-st.mean(mr[:n])) for i in range(n))/n
            vm=st.pvariance(mr[:n]); f["sector_beta_to_market"]=cov/vm if vm>0 else None

    if list_date:
        try:
            ld=datetime.strptime(list_date,"%Y-%m-%d").date()
            f["years_public"]=(date.today()-ld).days/365.25
        except: pass
    if market_cap and employees and employees>0:
        f["revenue_per_employee_proxy"]=market_cap/employees

    if sector_closes and len(sector_closes)>=63:
        if len(sector_closes)>21: f["sector_momentum_1m"]=sector_closes[-1]/sector_closes[-21]-1
        if len(sector_closes)>=63:
            r1=sector_closes[-1]/sector_closes[-21]-1; r2=sector_closes[-21]/sector_closes[-42]-1
            f["sector_acceleration"]=r1-r2
    if market_cap and employees: f["capital_efficiency"]=market_cap/employees/1e6

    return {k:v for k,v in f.items() if (v is not None or k.startswith("_"))}
