"""
QuantEdge v6.0 — Cross-Sectional Panel Builder
================================================
Builds ONE large training panel from many tickers × years of history, so the
ML models train on ~50-100k samples instead of ~81 per-ticker. This is the
foundation that makes rank-IC and SHAP genuinely meaningful.

Pipeline per ticker:
  1. Fetch 5yr daily OHLCV from Polygon
  2. build_historical_feature_matrix() -> (n_samples, n_features) [same engine as live serving]
  3. Triple-barrier labels (Lopez de Prado Ch.3) - path-dependent 21-day forward returns
  4. Stack all tickers into one panel keyed by (date, ticker)

Then CROSS-SECTIONAL RANKING (the key step):
  For each date, rank every feature into [0,1] percentile WITHIN that date's
  cross-section. This is what lets the model learn relative positioning
  (e.g. "top-quintile momentum vs peers today") - the real source of
  cross-sectional alpha. Raw features can't express this; ranked ones can.

Output: parquet panel in MODEL_DIR/panels/ + a build report.

Run locally:  python -m ml.training.build_panel --tickers 300 --years 5
"""
from __future__ import annotations
import os, sys, time, json, argparse, logging
from pathlib import Path
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("build_panel")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "./ml_models"))
PANEL_DIR = MODEL_DIR / "panels"
PANEL_DIR.mkdir(parents=True, exist_ok=True)

POLYGON_KEY = os.environ.get("POLYGON_API_KEY", "")
POLYGON_BASE = "https://api.polygon.io"

# ── Universe: liquid, well-covered names across sectors ──
# Start from a broad liquid set; can be widened. These span every major sector
# so cross-sectional ranking sees real dispersion.
DEFAULT_UNIVERSE = [
    # Mega/large tech
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","AVGO","ORCL","CRM","ADBE","CSCO","AMD","INTC","QCOM","TXN","IBM","NOW","INTU","MU","AMAT","LRCX","ADI","KLAC","SNPS","CDNS","PANW","ANET","FTNT",
    # Financials
    "JPM","BAC","WFC","GS","MS","C","BLK","SPGI","AXP","SCHW","CB","MMC","PGR","USB","PNC","TFC","BK","AIG","MET","PRU","AFL","ALL","TRV",
    # Healthcare
    "LLY","UNH","JNJ","MRK","ABBV","TMO","ABT","DHR","AMGN","PFE","BMY","GILD","CVS","CI","ELV","ISRG","VRTX","REGN","MDT","SYK","BSX","ZTS","HCA","BDX",
    # Consumer
    "WMT","COST","HD","PG","KO","PEP","MCD","NKE","LOW","SBUX","TJX","TGT","DG","DLTR","YUM","CMG","ORLY","AZO","ROST","EL","KMB","CL","GIS","KHC","MDLZ","MNST","KDP","HSY",
    # Industrials/Energy
    "XOM","CVX","GE","RTX","HON","UPS","CAT","DE","LMT","BA","UNP","ADP","NOC","GD","EMR","ETN","ITW","CSX","NSC","WM","PH","COP","SLB","EOG","MPC","PSX","VLO","OXY","KMI","WMB",
    # Comm/Media/Other
    "NFLX","DIS","CMCSA","T","VZ","TMUS","CHTR","V","MA","PYPL","ACN","LIN","APD","SHW","ECL","FCX","NEM","NUE","DOW","TSLA","F","GM","UBER","ABNB","BKNG","MAR","HLT","DAL","UAL",
    # REITs/Utilities
    "PLD","AMT","EQIX","CCI","PSA","O","SPG","NEE","DUK","SO","D","AEP","EXC","SRE","XEL","ED","PEG","WEC",
]

async def _fetch_bars(client: httpx.AsyncClient, ticker: str, years: int) -> Optional[pd.DataFrame]:
    end = date.today(); start = end - timedelta(days=int(years * 365.25) + 10)
    url = (f"{POLYGON_BASE}/v2/aggs/ticker/{ticker}/range/1/day/"
           f"{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_KEY}")
    try:
        r = await client.get(url, timeout=30)
        if r.status_code != 200: return None
        res = (r.json() or {}).get("results", [])
        if not res or len(res) < 400: return None
        df = pd.DataFrame(res)
        df["date"] = pd.to_datetime(df["t"], unit="ms")
        df = df.rename(columns={"o":"open","h":"high","l":"low","c":"close","v":"volume"})
        df = df.set_index("date")[["open","high","low","close","volume"]].sort_index()
        return df
    except Exception as e:
        logger.warning(f"[{ticker}] fetch failed: {e}")
        return None

HORIZONS = [5, 10, 21, 63, 126, 252]  # 1wk, 2wk, 1mo, 3mo, 6mo, 1yr — weeks to years

def _multi_horizon_labels(close: pd.Series, dates: pd.DatetimeIndex,
                           pt_mult: float = 2.0, sl_mult: float = 2.0):
    """Path-dependent forward-return labels for MULTIPLE horizons.
    Returns (labels_dict, valid_idx) where labels_dict[h] is the list of realized
    log-returns at horizon h, aligned with valid_idx. A sample is valid only if the
    LONGEST horizon fits (so all horizons share the same rows). 21d uses triple-barrier
    (path-dependent); longer horizons use terminal return (barriers are a short-horizon
    trading concept). No lookahead: only uses prices after the sample date."""
    log_ret = np.log(close / close.shift(1))
    sigma = log_ret.rolling(20, min_periods=5).std().bfill()
    n = len(close)
    max_h = max(HORIZONS)
    labels = {h: [] for h in HORIZONS}
    valid = []
    for k, d in enumerate(dates):
        try:
            pos = close.index.get_loc(d)
            if pos + max_h >= n:
                continue
            entry = float(close.iloc[pos]); s = float(sigma.iloc[pos])
            if s <= 0 or not np.isfinite(s):
                continue
            row_labels = {}
            ok = True
            for h in HORIZONS:
                if h <= 21:
                    # triple-barrier for the short horizon
                    upper = entry * np.exp(pt_mult * s); lower = entry * np.exp(-sl_mult * s)
                    realized = None
                    for j in range(1, h + 1):
                        p = float(close.iloc[pos + j])
                        if p >= upper: realized = np.log(upper / entry); break
                        if p <= lower: realized = np.log(lower / entry); break
                    if realized is None:
                        realized = np.log(float(close.iloc[pos + h]) / entry)
                else:
                    # terminal log-return for longer horizons
                    realized = np.log(float(close.iloc[pos + h]) / entry)
                if not np.isfinite(realized):
                    ok = False; break
                row_labels[h] = float(realized)
            if not ok:
                continue
            for h in HORIZONS:
                labels[h].append(row_labels[h])
            valid.append(k)
        except Exception:
            continue
    return labels, valid

async def build_panel(tickers: List[str], years: int, step: int, lookback: int) -> pd.DataFrame:
    from ml.features.feature_engineering import FeaturePipeline
    from quantedge.fundamentals.universe_full import ticker_cik_map
    from quantedge.fundamentals.edgar_bulk import company_facts_from_bulk
    from quantedge.fundamentals.pit_fundamentals import point_in_time_fundamentals
    fe = FeaturePipeline()
    logger.info("Loading ticker->CIK map for fundamentals...")
    cik_map = ticker_cik_map()
    rows = []
    ok = 0
    async with httpx.AsyncClient() as client:
        for idx, tk in enumerate(tickers):
            t0 = time.time()
            df = await _fetch_bars(client, tk, years)
            if df is None or len(df) < lookback + 40:
                logger.info(f"[{idx+1}/{len(tickers)}] {tk}: insufficient data, skip")
                continue
            try:
                X, feat_names, dates = fe.build_historical_feature_matrix(
                    df=df, fundamentals={}, lookback_days=lookback, step=step)
            except Exception as e:
                logger.info(f"[{idx+1}/{len(tickers)}] {tk}: feature build failed ({e}), skip")
                continue
            labels, valid = _multi_horizon_labels(df["close"], dates)
            if len(valid) < 30:
                logger.info(f"[{idx+1}/{len(tickers)}] {tk}: only {len(valid)} labels, skip")
                continue
            Xv = X[valid]
            cik = cik_map.get(tk)
            facts = None
            if cik:
                try:
                    facts = company_facts_from_bulk(cik)  # load once, released after ticker
                except Exception:
                    facts = None
            for i in range(len(valid)):
                sample_date = dates[valid[i]]
                row = {"date": sample_date, "ticker": tk}
                for h in HORIZONS:
                    row[f"label_{h}d"] = labels[h][i]
                for j, fn in enumerate(feat_names):
                    row[fn] = float(Xv[i, j])
                if facts is not None:
                    try:
                        pos = df.index.get_loc(sample_date)
                        close_px = float(df["close"].iloc[pos])
                        pit = point_in_time_fundamentals(
                            facts,
                            sample_date.date() if hasattr(sample_date, "date") else sample_date,
                            price=close_px,
                        )
                        for fk, fv in pit.items():
                            row[fk] = fv
                    except Exception:
                        pass
                rows.append(row)
            facts = None  # release SEC JSON before next ticker
            ok += 1
            logger.info(f"[{idx+1}/{len(tickers)}] {tk}: +{len(valid)} samples ({time.time()-t0:.1f}s) | panel={len(rows)}")
            await __import__("asyncio").sleep(0.05)  # gentle pacing
    panel = pd.DataFrame(rows)
    logger.info(f"PANEL BUILT: {len(panel)} rows from {ok} tickers, {panel.shape[1]-3} features")
    return panel

def add_cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """For each date, rank each feature into [0,1] percentile within the cross-section.
    This is the key transform enabling relative-value learning."""
    feat_cols = [c for c in panel.columns if c not in ("date","ticker") and not c.startswith("label")]
    logger.info(f"Cross-sectional ranking {len(feat_cols)} features across {panel['date'].nunique()} dates...")
    # winsorize raw features first (fat tails), then rank within date
    ranked = panel.copy()
    # rank within each date -> percentile [0,1]; dates with <5 names left raw-median 0.5
    def _rank_group(g):
        for c in feat_cols:
            if g[c].notna().sum() >= 5:
                g[c + "_csrank"] = g[c].rank(pct=True)
            else:
                g[c + "_csrank"] = 0.5
        return g
    ranked = ranked.groupby("date", group_keys=False).apply(_rank_group)
    csrank_cols = [c + "_csrank" for c in feat_cols]
    logger.info(f"Added {len(csrank_cols)} cross-sectional rank features")
    return ranked, feat_cols, csrank_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", type=int, default=len(DEFAULT_UNIVERSE))
    ap.add_argument("--years", type=int, default=5)
    ap.add_argument("--step", type=int, default=5, help="days between samples per ticker")
    ap.add_argument("--lookback", type=int, default=252)
    args = ap.parse_args()

    if not POLYGON_KEY:
        logger.error("POLYGON_API_KEY not set. Export it first."); sys.exit(1)

    universe = DEFAULT_UNIVERSE[:args.tickers]
    logger.info(f"Building panel: {len(universe)} tickers, {args.years}yr, step={args.step}, lookback={args.lookback}")

    import asyncio
    panel = asyncio.run(build_panel(universe, args.years, args.step, args.lookback))
    if panel.empty:
        logger.error("Empty panel — aborting"); sys.exit(1)

    panel, feat_cols, csrank_cols = add_cross_sectional_ranks(panel)

    stamp = datetime.now().strftime("%Y%m%d")
    out = PANEL_DIR / f"panel_{stamp}.parquet"
    panel.to_parquet(out, index=False)

    meta = {
        "built_at": datetime.now().isoformat(),
        "n_rows": len(panel), "n_tickers": panel["ticker"].nunique(),
        "n_dates": panel["date"].nunique(),
        "raw_features": feat_cols, "csrank_features": csrank_cols,
        "years": args.years, "step": args.step, "lookback": args.lookback,
        "date_range": [str(panel["date"].min()), str(panel["date"].max())],
        "label_stats": {h: {"mean": float(panel[f"label_{h}d"].mean()), "std": float(panel[f"label_{h}d"].std())}
                        for h in [5,10,21,63,126,252] if f"label_{h}d" in panel.columns},
    }
    (PANEL_DIR / f"panel_{stamp}_meta.json").write_text(json.dumps(meta, indent=2))

    logger.info("="*60)
    logger.info(f"PANEL SAVED: {out}")
    logger.info(f"  {meta['n_rows']} rows | {meta['n_tickers']} tickers | {meta['n_dates']} dates")
    logger.info(f"  {len(feat_cols)} raw + {len(csrank_cols)} cross-sectional-rank features")
    logger.info(f"  label horizons: {list(meta['label_stats'].keys())}")
    logger.info(f"  date range: {meta['date_range'][0][:10]} → {meta['date_range'][1][:10]}")
    logger.info("="*60)

if __name__ == "__main__":
    main()
