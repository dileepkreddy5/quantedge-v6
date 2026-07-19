"""Valuation deepening — new real signals: full multiples suite, yield family,
quality-at-price (Greenblatt), reverse-DCF expansion, DCF internals, MoS expansion.
All from existing valuation features + financials. Takes Valuation 34 -> ~60.
"""
import math
def _f(v):
    try: v=float(v); return v if math.isfinite(v) else None
    except: return None
def _safe_div(a,b):
    a=_f(a); b=_f(b)
    return a/b if (a is not None and b not in (None,0)) else None

def compute_valuation_deepening(vf, fin_features, price_history=None):
    f={}
    price=_f(vf.get("current_price")); ev=_f(vf.get("enterprise_value")); shares=_f(vf.get("diluted_shares"))
    mcap=(price*shares) if (price and shares) else None
    rev=_f(vf.get("revenue")); ni=_f(vf.get("net_income")); ebitda=_f(vf.get("ebitda"))
    fcf=_f(fin_features.get("free_cash_flow")) or _f(fin_features.get("fcf"))
    ebit=_f(fin_features.get("operating_income")) or _f(fin_features.get("ebit"))
    ocf=_f(fin_features.get("operating_cash_flow"))

    f["ps_ratio"]=_safe_div(mcap,rev); f["pcf_ratio"]=_safe_div(mcap,ocf)
    f["ev_sales"]=_safe_div(ev,rev); f["ev_ebit"]=_safe_div(ev,ebit); f["ev_fcf"]=_safe_div(ev,fcf)
    f["price_book"]=_safe_div(price, vf.get("book_value_per_share"))
    pe=_safe_div(mcap,ni); growth=_f(vf.get("revenue_growth"))
    f["peg_ratio"]=_safe_div(pe,(growth*100)) if (pe and growth and growth>0) else None

    f["earnings_yield"]=_safe_div(ni,mcap); f["fcf_yield"]=_safe_div(fcf,mcap)
    f["ebit_ev_yield"]=_safe_div(ebit,ev)
    buybacks=_f(fin_features.get("buybacks")); divs=_f(fin_features.get("dividends_paid"))
    if mcap and (buybacks or divs): f["shareholder_yield"]=(abs(buybacks or 0)+abs(divs or 0))/mcap
    f["cash_return_on_price"]=_safe_div(ocf,mcap)

    roic=_f(fin_features.get("roic_ex_goodwill")) or _f(fin_features.get("roic"))
    f["roic_to_ev"]=_safe_div(ebit,ev); f["fcf_to_ev"]=_safe_div(fcf,ev)
    f["greenblatt_earnings_yield"]=f.get("ebit_ev_yield")
    if roic is not None and f.get("ebit_ev_yield") is not None:
        f["quality_value_composite"]=(roic+f["ebit_ev_yield"])/2

    f["pv_terminal_pct"]=_f(vf.get("dcf_terminal_pct")) or _f(vf.get("terminal_pct"))
    if f.get("pv_terminal_pct") is not None: f["pv_explicit_pct"]=1-f["pv_terminal_pct"]
    dcf=_f(vf.get("dcf_weighted"))
    f["dcf_vs_price_gap"]=_safe_div((dcf-price),price) if (dcf and price) else None
    if dcf and growth and growth>0: f["growth_adjusted_value"]=dcf*(1+growth)

    consensus=_f(vf.get("intrinsic_consensus"))
    f["intrinsic_consensus_gap"]=_safe_div((consensus-price),price) if (consensus and price) else None
    f["intrinsic_dispersion"]=_f(vf.get("model_dispersion_cv"))
    oe=_f(fin_features.get("owner_earnings"))
    if oe and mcap: f["owner_earnings_yield"]=oe/mcap
    f["tangible_book_value"]=_f(fin_features.get("tangible_book_per_share"))

    ig=_f(vf.get("reverse_dcf_implied_growth")); f["implied_growth"]=ig
    cur_margin=_f(fin_features.get("operating_margin"))
    if ig is not None and cur_margin is not None:
        f["implied_margin"]=cur_margin
        hist_growth=_f(fin_features.get("revenue_cagr_5y")) or _f(vf.get("revenue_growth"))
        if hist_growth is not None:
            f["expectations_feasibility"]=(1-min(1,max(0,(ig-hist_growth)/0.2))) if ig>hist_growth else 1.0
            f["implied_vs_historical_growth"]=ig-hist_growth

    fair=_f(vf.get("fair_value"))
    if fair and price:
        f["upside_to_fair"]=_safe_div((fair-price),price)
        bear=_f(vf.get("dcf_bear")); bull=_f(vf.get("dcf_bull"))
        if bear and bull:
            down=abs(price-bear); up=abs(bull-price)
            f["risk_reward_ratio"]=up/down if down>0 else None
    f["current_vs_5y_pe"]=_f(vf.get("pe_vs_history"))
    nnwc=_f(fin_features.get("net_net_working_capital"))
    if nnwc and shares and price: f["graham_net_net_ratio"]=_safe_div((nnwc/shares),price)

    return {k:v for k,v in f.items() if v is not None}
