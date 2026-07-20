"""ML Models Intelligence — transparent ensemble meta-model. Interpretable factor sub-models
(value, momentum, quality, growth, low-vol) stacked into a blended signal. Honest: rules-based
logistic ensemble of engineered factors, NOT a black-box net."""
import math
def _f(v):
    try: v=float(v); return v if math.isfinite(v) else None
    except: return None
def _logistic(x):
    try: return 1/(1+math.exp(-x))
    except OverflowError: return 0.0 if x<0 else 1.0
def _zband(v, center, scale):
    if v is None: return None
    return _logistic((v-center)/scale)

def compute_ml_features(feats):
    f={}
    def g(*keys):
        for k in keys:
            v=_f(feats.get(k))
            if v is not None: return v
        return None

    ey=g("earnings_yield","fund_earnings_yield"); ocfy=g("ocf_yield","fund_ocf_yield")
    pe=g("pe","mult_pe","fund_pe")
    vparts=[]
    if ey is not None: vparts.append(_zband(ey,0.04,0.03))
    if ocfy is not None: vparts.append(_zband(ocfy,0.05,0.03))
    if pe is not None: vparts.append(1-_zband(pe,25,12))
    if vparts:
        f["value_model_score"]=sum(vparts)/len(vparts)
        f["value_signal"]=1.0 if f["value_model_score"]>0.55 else (0.0 if f["value_model_score"]<0.45 else 0.5)

    m6=g("momentum_6m"); m3=g("momentum_3m"); mq=g("momentum_quality"); maa=g("ma_golden_cross","ma_trend")
    mparts=[]
    if m6 is not None: mparts.append(_zband(m6,0.0,0.15))
    if m3 is not None: mparts.append(_zband(m3,0.0,0.1))
    if mq is not None: mparts.append(_zband(mq,0.0,0.3))
    if maa is not None: mparts.append(_zband(maa,0.0,0.05))
    if mparts:
        f["momentum_model_score"]=sum(mparts)/len(mparts)
        f["momentum_signal"]=1.0 if f["momentum_model_score"]>0.55 else (0.0 if f["momentum_model_score"]<0.45 else 0.5)

    roic=g("roic","roic_level","fund_roic_approx"); roe=g("roe","roe_level","fund_roe")
    nm=g("net_margin","fund_net_margin"); ccl=g("cash_conversion_level","cash_conversion")
    qparts=[]
    if roic is not None: qparts.append(_zband(roic,0.12,0.08))
    if roe is not None: qparts.append(_zband(roe,0.15,0.1))
    if nm is not None: qparts.append(_zband(nm,0.1,0.08))
    if ccl is not None: qparts.append(_zband(ccl,1.0,0.3))
    if qparts:
        f["quality_model_score"]=sum(qparts)/len(qparts)
        f["quality_signal"]=1.0 if f["quality_model_score"]>0.55 else (0.0 if f["quality_model_score"]<0.45 else 0.5)

    rg=g("revenue_growth","revenue_growth_ttm","fund_revenue_growth"); eg=g("earnings_growth_recent","earnings_growth")
    ra=g("revenue_accel"); r40=g("rule_of_40")
    gparts=[]
    if rg is not None: gparts.append(_zband(rg,0.05,0.1))
    if eg is not None: gparts.append(_zband(eg,0.0,0.1))
    if ra is not None: gparts.append(_zband(ra,0.0,0.03))
    if r40 is not None: gparts.append(_zband(r40,0.4,0.15))
    if gparts:
        f["growth_model_score"]=sum(gparts)/len(gparts)
        f["growth_signal"]=1.0 if f["growth_model_score"]>0.55 else (0.0 if f["growth_model_score"]<0.45 else 0.5)

    fc=g("forecast_confidence"); beta=g("market_beta"); ext=g("extension_from_mean")
    lparts=[]
    if fc is not None: lparts.append(fc)
    if beta is not None: lparts.append(1-_zband(beta,1.0,0.4))
    if ext is not None: lparts.append(1-_zband(ext,0.2,0.1))
    if lparts: f["lowvol_model_score"]=sum(lparts)/len(lparts)

    submodels={k:f.get(k) for k in ["value_model_score","momentum_model_score","quality_model_score",
               "growth_model_score","lowvol_model_score"] if f.get(k) is not None}
    if submodels:
        vals=list(submodels.values())
        wmap={"value_model_score":0.20,"momentum_model_score":0.22,"quality_model_score":0.24,
              "growth_model_score":0.22,"lowvol_model_score":0.12}
        wsum=sum(wmap[k] for k in submodels)
        f["ensemble_score"]=sum(submodels[k]*wmap[k] for k in submodels)/wsum if wsum>0 else sum(vals)/len(vals)
        bullish=sum(1 for v in vals if v>0.55); bearish=sum(1 for v in vals if v<0.45)
        f["model_agreement"]=max(bullish,bearish)/len(vals)
        f["ensemble_conviction"]=abs(f["ensemble_score"]-0.5)*2
        f["up_probability"]=_logistic((f["ensemble_score"]-0.5)*6)
        f["n_models_bullish"]=float(bullish)
        f["signal_dispersion"]=1.0-(max(vals)-min(vals)) if len(vals)>1 else None
    return {k:v for k,v in f.items() if v is not None}
