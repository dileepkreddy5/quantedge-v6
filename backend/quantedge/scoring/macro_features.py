"""Macro Sensitivity — stock's exposure to rate, inflation, dollar, growth-cycle, risk-regime,
commodity, and factor regimes. Computed from price correlation/beta to macro-proxy ETFs.
TLT=rates, UUP=dollar, GLD=inflation/gold, HYG=credit/risk, USO=oil, IWM=small-cap,
VLUE=value factor, MTUM=momentum factor, SPY=market."""
import math, statistics as st

def _rets(closes): return [(closes[i]/closes[i-1]-1) for i in range(1,len(closes))] if len(closes)>1 else []

def _beta_corr(sr, mr):
    n=min(len(sr),len(mr))
    if n<40: return None,None
    sr=sr[-n:]; mr=mr[-n:]
    ms=st.mean(sr); mm=st.mean(mr)
    cov=sum((sr[i]-ms)*(mr[i]-mm) for i in range(n))/n
    vm=st.pvariance(mr); sds=st.pstdev(sr); sdm=st.pstdev(mr)
    beta=cov/vm if vm>0 else None
    corr=cov/(sds*sdm) if (sds>0 and sdm>0) else None
    return beta, corr

def compute_macro_features(stock_closes, proxies):
    """proxies: dict {name: closes[]} for TLT/UUP/GLD/HYG/USO/SPY/IWM/VLUE/MTUM."""
    f={}
    if not stock_closes or len(stock_closes)<60: return {"available":False}
    sr=_rets(stock_closes[-252:])
    def pr(name):
        cl=proxies.get(name)
        return _rets(cl[-252:]) if cl and len(cl)>=60 else None

    # ===== RATE SENSITIVITY (TLT) =====
    tlt=pr("TLT")
    if tlt:
        b,c=_beta_corr(sr,tlt)
        f["rate_beta"]=b  # positive = benefits from falling rates (bond-like); negative = hurt by rates
        f["rate_correlation"]=c
        if b is not None: f["rate_sensitivity_abs"]=abs(b)

    # ===== INFLATION (GLD, USO) =====
    gld=pr("GLD")
    if gld:
        b,c=_beta_corr(sr,gld)
        f["gold_beta"]=b; f["inflation_hedge"]=c
    uso=pr("USO")
    if uso:
        b,c=_beta_corr(sr,uso)
        f["oil_beta"]=b; f["oil_correlation"]=c

    # ===== DOLLAR (UUP) =====
    uup=pr("UUP")
    if uup:
        b,c=_beta_corr(sr,uup)
        f["dollar_beta"]=b  # negative = hurt by strong dollar (multinational)
        f["dollar_correlation"]=c
        if c is not None: f["dollar_sensitivity_abs"]=abs(c)

    # ===== GROWTH CYCLE (IWM small-cap, SPY) =====
    iwm=pr("IWM")
    if iwm:
        b,c=_beta_corr(sr,iwm)
        f["smallcap_beta"]=b; f["cyclical_correlation"]=c
    spy=pr("SPY")
    if spy:
        b,c=_beta_corr(sr,spy)
        f["market_beta"]=b; f["market_correlation"]=c

    # ===== RISK REGIME (HYG credit) =====
    hyg=pr("HYG")
    if hyg:
        b,c=_beta_corr(sr,hyg)
        f["credit_beta"]=b  # high = risk-on sensitive
        f["risk_on_correlation"]=c

    # ===== FACTOR EXPOSURE (VLUE, MTUM) =====
    vlue=pr("VLUE")
    if vlue:
        b,c=_beta_corr(sr,vlue)
        f["value_factor_beta"]=b; f["value_tilt"]=c
    mtum=pr("MTUM")
    if mtum:
        b,c=_beta_corr(sr,mtum)
        f["momentum_factor_beta"]=b; f["momentum_tilt"]=c

    # ===== DEFENSIVENESS =====
    # low market beta + low credit sensitivity = defensive
    if f.get("market_beta") is not None:
        f["defensiveness"]=1.0-min(1.0,f["market_beta"])  # low beta = defensive
    # macro resilience: low absolute sensitivity across factors = insulated
    sens=[abs(f.get(k)) for k in ["rate_beta","dollar_beta","credit_beta"] if f.get(k) is not None]
    if sens: f["macro_resilience"]=1.0-min(1.0,st.mean(sens))

    # volatility of macro exposure (stability of beta over sub-windows)
    if spy and len(sr)>=120:
        half=len(sr)//2
        b1,_=_beta_corr(sr[:half],pr("SPY")[:half]) if pr("SPY") and len(pr("SPY"))>=half else (None,None)
        b2,_=_beta_corr(sr[half:],pr("SPY")[half:]) if pr("SPY") and len(pr("SPY"))>=half else (None,None)
        if b1 is not None and b2 is not None: f["beta_stability"]=1.0-min(1.0,abs(b2-b1))

    return {k:v for k,v in f.items() if v is not None}

if __name__=="__main__":
    import random; random.seed(2)
    def mk(n=252,drift=0.0003,vol=0.015):
        c=[100]
        for _ in range(n): c.append(c[-1]*(1+random.gauss(drift,vol)))
        return c
    stock=mk(); proxies={n:mk() for n in ['TLT','UUP','GLD','HYG','USO','SPY','IWM','VLUE','MTUM']}
    f=compute_macro_features(stock,proxies)
    print(f"{sum(1 for k in f if not k.startswith('_'))} signals")
    for k,v in sorted(f.items()): print(f"  {k:26s} {round(v,4) if isinstance(v,float) else v}")
