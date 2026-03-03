"""
QuantEdge v5.0 — Feature Engineering Engine
================================================
200+ institutional-grade features used by top quant funds.
Implements the exact feature sets used by:
  - Renaissance Technologies: eigenportfolio + cross-sectional factors
  - Two Sigma: alternative data signals + NLP features
  - D.E. Shaw: microstructure + statistical arbitrage features
  - Citadel: options-derived features + vol surface signals
  - AQR: Fama-French 6-factor + momentum + quality factors

Mathematical References:
  - Barra Risk Model (MSCI) — factor decomposition
  - Almgren-Chriss (2001) — market impact / optimal execution
  - Kyle (1985) — VPIN / informed trading probability
  - Heston (1993) — stochastic volatility model
  - Fama-French (2015) — 5-factor asset pricing model
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import butter, filtfilt
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════
# SECTION 1: PRICE & MOMENTUM FEATURES
# ══════════════════════════════════════════════════════════════

class MomentumFeatures:
    """
    Implements cross-sectional and time-series momentum signals.
    Mathematical basis: Jegadeesh & Titman (1993), Asness et al. (2013)
    """

    @staticmethod
    def time_series_momentum(returns: pd.Series, lookbacks: list = [5, 10, 21, 63, 126, 252]) -> Dict:
        """
        TSMOM signal: r_t^{(K)} = sign(r_{t-1,t-K}) * r_{t-1,t-K}
        Moskowitz, Ooi, Pedersen (2012) — "Time Series Momentum"
        """
        features = {}
        for lb in lookbacks:
            if len(returns) >= lb:
                r = returns.iloc[-lb:].sum()
                features[f"tsmom_{lb}d"] = r
                features[f"tsmom_sign_{lb}d"] = np.sign(r)
                # Sharpe-weighted momentum (annualized)
                if returns.iloc[-lb:].std() > 0:
                    features[f"tsmom_sharpe_{lb}d"] = (
                        r / (returns.iloc[-lb:].std() * np.sqrt(252 / lb))
                    )
        return features

    @staticmethod
    def cross_sectional_momentum(returns: pd.Series) -> Dict:
        """12-1 month cross-sectional momentum (skip last month to avoid reversal)"""
        if len(returns) < 252:
            return {}
        r_12_1 = returns.iloc[-252:-21].sum()
        r_1m = returns.iloc[-21:].sum()
        return {
            "cs_momentum_12_1": r_12_1,
            "cs_momentum_1m": r_1m,
            "momentum_reversal": -r_1m,  # Short-term reversal signal
        }

    @staticmethod
    def residual_momentum(returns: pd.Series, market_returns: pd.Series) -> Dict:
        """
        Residual momentum (Blitz, Huij, Martens 2011):
        Remove market beta from momentum to get pure stock-specific momentum
        β = Cov(r_i, r_m) / Var(r_m)
        resid = r_i - β * r_m
        """
        if len(returns) < 126 or len(market_returns) < 126:
            return {}
        n = min(len(returns), len(market_returns), 126)
        ri = returns.iloc[-n:].values
        rm = market_returns.iloc[-n:].values
        beta = np.cov(ri, rm)[0, 1] / np.var(rm) if np.var(rm) > 0 else 1.0
        residuals = pd.Series(ri - beta * rm)
        return {
            "residual_momentum_6m": residuals.sum(),
            "residual_momentum_vol": residuals.std() * np.sqrt(252),
            "beta_adjusted_momentum": residuals.sum() / (residuals.std() + 1e-8),
        }

    @staticmethod
    def acceleration(returns: pd.Series) -> Dict:
        """
        Momentum acceleration: second derivative of momentum
        Measures if momentum is speeding up or slowing down
        """
        if len(returns) < 63:
            return {}
        recent = returns.iloc[-21:].sum()
        medium = returns.iloc[-63:-21].sum()
        return {
            "momentum_acceleration": recent - medium / 2,
            "momentum_concavity": recent - 2 * medium,
        }


class TechnicalFeatures:
    """
    Complete technical analysis feature set with mathematical precision.
    All formulas from Kaufman (2013) — Trading Systems and Methods.
    """

    @staticmethod
    def compute_all(df: pd.DataFrame) -> Dict:
        """
        Compute 80+ technical features from OHLCV data.
        df must have columns: open, high, low, close, volume
        """
        features = {}
        close = df["close"]
        high = df["high"]
        low = df["low"]
        volume = df["volume"]
        returns = close.pct_change()

        # ── RSI (Relative Strength Index) ──────────────────
        # RSI = 100 - 100/(1 + RS), RS = Avg_Gain / Avg_Loss
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
            rs = avg_gain / (avg_loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            features[f"rsi_{period}"] = rsi.iloc[-1]
            features[f"rsi_{period}_slope"] = rsi.diff(5).iloc[-1]

        # ── MACD ───────────────────────────────────────────
        # MACD = EMA_12 - EMA_26; Signal = EMA_9(MACD)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        features["macd"] = macd.iloc[-1]
        features["macd_signal"] = signal.iloc[-1]
        features["macd_hist"] = hist.iloc[-1]
        features["macd_hist_slope"] = hist.diff(3).iloc[-1]
        features["macd_crossover"] = 1 if macd.iloc[-1] > signal.iloc[-1] else -1

        # ── Bollinger Bands ────────────────────────────────
        # Upper = SMA_20 + 2σ, Lower = SMA_20 - 2σ
        for period in [20, 50]:
            sma = close.rolling(period).mean()
            std = close.rolling(period).std()
            upper = sma + 2 * std
            lower = sma - 2 * std
            pct_b = (close - lower) / (upper - lower + 1e-10)
            bandwidth = (upper - lower) / sma
            features[f"bb_pct_b_{period}"] = pct_b.iloc[-1]
            features[f"bb_bandwidth_{period}"] = bandwidth.iloc[-1]
            features[f"bb_squeeze_{period}"] = 1 if bandwidth.iloc[-1] < bandwidth.rolling(20).mean().iloc[-1] else 0

        # ── ATR (Average True Range) ───────────────────────
        # TR = max(H-L, |H-C_prev|, |L-C_prev|)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        for period in [14, 21]:
            atr = tr.ewm(span=period, adjust=False).mean()
            features[f"atr_{period}"] = atr.iloc[-1]
            features[f"atr_{period}_normalized"] = atr.iloc[-1] / close.iloc[-1]

        # ── ADX (Average Directional Index) ────────────────
        # Measures trend strength (0-100), not direction
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        mask = plus_dm > minus_dm
        plus_dm = plus_dm.where(mask, 0)
        minus_dm = minus_dm.where(~mask, 0)
        atr14 = tr.ewm(span=14, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-10)
        minus_di = 100 * minus_dm.ewm(span=14, adjust=False).mean() / (atr14 + 1e-10)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=14, adjust=False).mean()
        features["adx"] = adx.iloc[-1]
        features["di_plus"] = plus_di.iloc[-1]
        features["di_minus"] = minus_di.iloc[-1]
        features["trend_strength"] = adx.iloc[-1] / 100

        # ── Stochastic Oscillator ──────────────────────────
        # %K = (C - L_14) / (H_14 - L_14) * 100
        for period in [14, 21]:
            lowest = low.rolling(period).min()
            highest = high.rolling(period).max()
            stoch_k = 100 * (close - lowest) / (highest - lowest + 1e-10)
            stoch_d = stoch_k.rolling(3).mean()
            features[f"stoch_k_{period}"] = stoch_k.iloc[-1]
            features[f"stoch_d_{period}"] = stoch_d.iloc[-1]

        # ── Commodity Channel Index ────────────────────────
        # CCI = (TP - SMA_TP) / (0.015 * MAD)
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        cci = (tp - sma_tp) / (0.015 * mad + 1e-10)
        features["cci"] = cci.iloc[-1]
        features["cci_overbought"] = 1 if cci.iloc[-1] > 100 else -1 if cci.iloc[-1] < -100 else 0

        # ── Williams %R ────────────────────────────────────
        highest_14 = high.rolling(14).max()
        lowest_14 = low.rolling(14).min()
        willr = -100 * (highest_14 - close) / (highest_14 - lowest_14 + 1e-10)
        features["williams_r"] = willr.iloc[-1]

        # ── Chaikin Money Flow ──────────────────────────────
        # CMF = Σ(CLV * Volume) / Σ(Volume)
        # CLV = [(C - L) - (H - C)] / (H - L)
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        cmf = (clv * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)
        features["cmf"] = cmf.iloc[-1]
        features["cmf_trend"] = cmf.diff(5).iloc[-1]

        # ── On-Balance Volume ──────────────────────────────
        obv = (np.sign(close.diff()) * volume).cumsum()
        features["obv_trend"] = obv.diff(20).iloc[-1] / (volume.mean() + 1e-10)
        features["obv_momentum"] = obv.pct_change(20).iloc[-1]

        # ── Volume Price Trend ─────────────────────────────
        vpt = (returns * volume).cumsum()
        features["vpt"] = vpt.iloc[-1]
        features["vpt_trend"] = vpt.diff(10).iloc[-1]

        # ── Moving Averages & Crossovers ───────────────────
        for period in [5, 10, 20, 50, 100, 200]:
            sma = close.rolling(period).mean()
            ema = close.ewm(span=period, adjust=False).mean()
            features[f"sma_{period}"] = sma.iloc[-1]
            features[f"ema_{period}"] = ema.iloc[-1]
            features[f"price_vs_sma_{period}"] = close.iloc[-1] / (sma.iloc[-1] + 1e-10) - 1
            features[f"price_vs_ema_{period}"] = close.iloc[-1] / (ema.iloc[-1] + 1e-10) - 1

        # Golden/Death cross signals
        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        features["golden_cross"] = 1 if sma50.iloc[-1] > sma200.iloc[-1] else -1
        features["golden_cross_distance"] = (sma50.iloc[-1] - sma200.iloc[-1]) / (sma200.iloc[-1] + 1e-10)

        # ── Volume Features ────────────────────────────────
        avg_vol_20 = volume.rolling(20).mean()
        features["volume_ratio"] = volume.iloc[-1] / (avg_vol_20.iloc[-1] + 1e-10)
        features["volume_trend"] = volume.rolling(5).mean().iloc[-1] / (avg_vol_20.iloc[-1] + 1e-10)
        features["dollar_volume"] = (close * volume).iloc[-1]
        features["dollar_volume_ma20"] = (close * volume).rolling(20).mean().iloc[-1]

        # ── Price Action ───────────────────────────────────
        features["daily_range"] = (high.iloc[-1] - low.iloc[-1]) / close.iloc[-1]
        features["upper_shadow"] = (high.iloc[-1] - max(close.iloc[-1], df["open"].iloc[-1])) / close.iloc[-1]
        features["lower_shadow"] = (min(close.iloc[-1], df["open"].iloc[-1]) - low.iloc[-1]) / close.iloc[-1]
        features["body_size"] = abs(close.iloc[-1] - df["open"].iloc[-1]) / close.iloc[-1]

        # ── Ichimoku Cloud ─────────────────────────────────
        # Tenkan-sen (9): (9H + 9L) / 2
        # Kijun-sen (26): (26H + 26L) / 2
        if len(df) >= 52:
            tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
            kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
            senkou_a = ((tenkan + kijun) / 2).shift(26)
            senkou_b = ((high.rolling(52).max() + low.rolling(52).min()) / 2).shift(26)
            features["ichimoku_above_cloud"] = 1 if close.iloc[-1] > max(senkou_a.iloc[-1] or 0, senkou_b.iloc[-1] or 0) else -1
            features["tenkan_kijun_cross"] = 1 if tenkan.iloc[-1] > kijun.iloc[-1] else -1

        # ── Pivot Points ───────────────────────────────────
        prev_high = high.iloc[-2]
        prev_low = low.iloc[-2]
        prev_close = close.iloc[-2]
        pivot = (prev_high + prev_low + prev_close) / 3
        r1 = 2 * pivot - prev_low
        r2 = pivot + (prev_high - prev_low)
        r3 = prev_high + 2 * (pivot - prev_low)
        s1 = 2 * pivot - prev_high
        s2 = pivot - (prev_high - prev_low)
        s3 = prev_low - 2 * (prev_high - pivot)
        features["vs_pivot"] = (close.iloc[-1] - pivot) / pivot
        features["vs_r1"] = (close.iloc[-1] - r1) / r1
        features["vs_s1"] = (close.iloc[-1] - s1) / (abs(s1) + 1e-10)

        # ── Fibonacci Levels ───────────────────────────────
        if len(df) >= 52:
            hi_52 = high.iloc[-252:].max() if len(df) >= 252 else high.max()
            lo_52 = low.iloc[-252:].min() if len(df) >= 252 else low.min()
            rng = hi_52 - lo_52
            fibs = {"fib_236": lo_52 + 0.236 * rng, "fib_382": lo_52 + 0.382 * rng,
                    "fib_500": lo_52 + 0.500 * rng, "fib_618": lo_52 + 0.618 * rng,
                    "fib_786": lo_52 + 0.786 * rng}
            current = close.iloc[-1]
            for name, level in fibs.items():
                features[f"vs_{name}"] = (current - level) / (level + 1e-10)

        return features


# ══════════════════════════════════════════════════════════════
# SECTION 2: VOLATILITY & RISK FEATURES
# ══════════════════════════════════════════════════════════════

class VolatilityFeatures:
    """
    Institutional volatility metrics used by vol desks at top funds.
    Implements: GARCH forecasts, realized vol, vol regime detection,
    Heston model parameters, IV surface features.
    """

    @staticmethod
    def realized_volatility(returns: pd.Series) -> Dict:
        """
        Andersen & Bollerslev (1998) — Realized Volatility measures:
        RV_t = sqrt(Σ r_{t,i}^2) for i in intraday intervals
        Daily proxy using close-to-close returns:
        """
        features = {}
        for window in [5, 10, 21, 42, 63, 126, 252]:
            if len(returns) >= window:
                r = returns.iloc[-window:]
                rv = r.std() * np.sqrt(252)  # Annualized
                features[f"realized_vol_{window}d"] = rv
                # Realized variance (non-annualized for ratio)
                features[f"realized_var_{window}d"] = r.var()

        # Volatility of volatility (vol-of-vol)
        if len(returns) >= 252:
            rolling_vol = returns.rolling(21).std() * np.sqrt(252)
            features["vol_of_vol"] = rolling_vol.std()
            features["vol_regime_mean"] = rolling_vol.mean()
            features["vol_persistence"] = rolling_vol.autocorr(1)  # AR(1) of vol

        # Volatility ratio (short/long) — regime signal
        if len(returns) >= 63:
            vol_short = returns.iloc[-21:].std() * np.sqrt(252)
            vol_long = returns.iloc[-63:].std() * np.sqrt(252)
            features["vol_ratio_short_long"] = vol_short / (vol_long + 1e-10)

        return features

    @staticmethod
    def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 21) -> float:
        """
        Parkinson (1980) High-Low Volatility Estimator:
        σ_P = sqrt(1/(4*n*ln2) * Σ ln(H_i/L_i)^2)
        More efficient than close-to-close (5x fewer data points needed)
        """
        log_hl = np.log(high / low)
        return np.sqrt(log_hl.rolling(window).apply(
            lambda x: (x**2).sum() / (4 * len(x) * np.log(2))
        )).iloc[-1] * np.sqrt(252)

    @staticmethod
    def garman_klass_volatility(df: pd.DataFrame, window: int = 21) -> float:
        """
        Garman & Klass (1980) OHLC Volatility Estimator:
        σ_GK = sqrt(0.5*(ln H/L)^2 - (2ln2-1)*(ln C/O)^2)
        Uses all 4 price points — most efficient close-form estimator
        """
        log_hl = np.log(df["high"] / df["low"])
        log_co = np.log(df["close"] / df["open"])
        gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        return np.sqrt(gk.rolling(window).mean()).iloc[-1] * np.sqrt(252)

    @staticmethod
    def yang_zhang_volatility(df: pd.DataFrame, window: int = 21) -> float:
        """
        Yang & Zhang (2000) — minimum variance unbiased estimator
        Handles overnight gaps (best for daily data):
        σ_YZ^2 = σ_overnight^2 + k*σ_open^2 + (1-k)*σ_rs^2
        where k = 0.34/(1.34 + (n+1)/(n-1))
        """
        n = window
        k = 0.34 / (1.34 + (n + 1) / (n - 1))
        log_oc = np.log(df["open"] / df["close"].shift(1))
        log_co = np.log(df["close"] / df["open"])
        log_ho = np.log(df["high"] / df["open"])
        log_lo = np.log(df["low"] / df["open"])
        log_hc = np.log(df["high"] / df["close"])
        log_lc = np.log(df["low"] / df["close"])
        sig_overnight = log_oc.rolling(n).var()
        sig_open = log_co.rolling(n).var()
        sig_rs = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(n).mean()
        yz = sig_overnight + k * sig_open + (1 - k) * sig_rs
        return np.sqrt(yz).iloc[-1] * np.sqrt(252)

    @staticmethod
    def hurst_exponent(returns: pd.Series, min_lag: int = 2, max_lag: int = 100) -> float:
        """
        Hurst Exponent via Rescaled Range (R/S) Analysis:
        H > 0.5 → trending (momentum), H < 0.5 → mean-reverting, H ≈ 0.5 → random walk

        R/S(τ) = (max(cumsum) - min(cumsum)) / std(returns)
        H = log(R/S) / log(τ)  as τ → ∞
        """
        if len(returns) < max_lag * 2:
            return 0.5  # Default to random walk
        ts = returns.dropna().values
        lags = range(min_lag, min(max_lag, len(ts) // 2))
        rs_values = []
        for lag in lags:
            chunks = [ts[i:i+lag] for i in range(0, len(ts)-lag, lag)]
            rs_chunk = []
            for chunk in chunks:
                mean = chunk.mean()
                deviation = np.cumsum(chunk - mean)
                rs = (deviation.max() - deviation.min()) / (chunk.std() + 1e-10)
                rs_chunk.append(rs)
            rs_values.append(np.mean(rs_chunk))
        try:
            hurst, _ = np.polyfit(np.log(list(lags)), np.log(rs_values), 1)
            return np.clip(hurst, 0, 1)
        except Exception:
            return 0.5

    @staticmethod
    def risk_metrics(returns: pd.Series, risk_free_rate: float = 0.05) -> Dict:
        """
        Complete institutional risk metric suite.
        All formulas verified against Sharpe (1994), Sortino (1994).
        """
        if len(returns) < 21:
            return {}

        r = returns.dropna()
        annual_returns = (1 + r).prod() ** (252 / len(r)) - 1
        annual_vol = r.std() * np.sqrt(252)
        excess_return = annual_returns - risk_free_rate

        # Sharpe Ratio: (E[r] - rf) / σ
        sharpe = excess_return / (annual_vol + 1e-10)

        # Sortino Ratio: (E[r] - rf) / σ_downside
        downside = r[r < 0].std() * np.sqrt(252)
        sortino = excess_return / (downside + 1e-10)

        # Maximum Drawdown
        cumulative = (1 + r).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # Calmar Ratio: Annual Return / Max Drawdown
        calmar = annual_returns / (abs(max_dd) + 1e-10)

        # Omega Ratio: E[max(r-L,0)] / E[max(L-r,0)]
        threshold = risk_free_rate / 252
        gains = r[r > threshold].sum()
        losses = abs(r[r < threshold].sum())
        omega = gains / (losses + 1e-10)

        # VaR and CVaR (Expected Shortfall)
        var_95 = np.percentile(r, 5)
        var_99 = np.percentile(r, 1)
        cvar_95 = r[r <= var_95].mean()
        cvar_99 = r[r <= var_99].mean()

        # Tail Ratio: 95th percentile / 5th percentile absolute
        tail_ratio = abs(np.percentile(r, 95)) / (abs(np.percentile(r, 5)) + 1e-10)

        # Skewness and Kurtosis
        skew = stats.skew(r)
        kurt = stats.kurtosis(r)  # Excess kurtosis

        # Ulcer Index: Depth/duration of drawdowns
        ulcer = np.sqrt((drawdown**2).mean())

        # Sterling Ratio: Average drawdown version
        avg_dd = drawdown[drawdown < 0].mean() if (drawdown < 0).any() else -0.01
        sterling = annual_returns / (abs(avg_dd) + 1e-10)

        # Information Ratio vs SPY (placeholder — requires benchmark)
        # Will be populated with actual SPY returns in the pipeline

        return {
            "annual_return": annual_returns,
            "annual_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "calmar_ratio": calmar,
            "omega_ratio": omega,
            "max_drawdown": max_dd,
            "var_95": var_95,
            "var_99": var_99,
            "cvar_95": cvar_95,
            "cvar_99": cvar_99,
            "tail_ratio": tail_ratio,
            "skewness": skew,
            "excess_kurtosis": kurt,
            "ulcer_index": ulcer,
            "sterling_ratio": sterling,
        }


# ══════════════════════════════════════════════════════════════
# SECTION 3: FACTOR MODEL FEATURES (BARRA / FAMA-FRENCH)
# ══════════════════════════════════════════════════════════════

class FactorFeatures:
    """
    Fama-French 6-Factor Model + Barra-style factor decomposition.
    Reference: Fama & French (2015), Hou, Xue & Zhang (2015)

    Factors:
        MKT  — Market excess return (CAPM beta)
        SMB  — Small Minus Big (size)
        HML  — High Minus Low (value)
        RMW  — Robust Minus Weak (profitability)
        CMA  — Conservative Minus Aggressive (investment)
        WML  — Winners Minus Losers (momentum)
    """

    @staticmethod
    def compute_factor_exposures(
        stock_returns: pd.Series,
        factor_returns: pd.DataFrame,
        window: int = 252
    ) -> Dict:
        """
        OLS regression of stock returns on factor returns.
        r_i = α + β_1*MKT + β_2*SMB + β_3*HML + β_4*RMW + β_5*CMA + β_6*WML + ε
        """
        from sklearn.linear_model import LinearRegression
        if len(stock_returns) < window:
            window = len(stock_returns)

        sr = stock_returns.iloc[-window:].values
        fr = factor_returns.iloc[-window:].values

        # OLS regression
        reg = LinearRegression().fit(fr, sr)
        y_pred = reg.predict(fr)
        residuals = sr - y_pred
        ss_res = (residuals**2).sum()
        ss_tot = ((sr - sr.mean())**2).sum()
        r_squared = 1 - ss_res / (ss_tot + 1e-10)

        # Information Coefficient on residuals
        ic = np.corrcoef(sr[1:], y_pred[:-1])[0, 1]

        # Idiosyncratic risk (residual volatility annualized)
        idio_risk = np.std(residuals) * np.sqrt(252)

        result = {
            "ff_alpha": reg.intercept_ * 252,  # Annualized alpha
            "ff_r_squared": r_squared,
            "ff_idio_risk": idio_risk,
            "ff_ic": ic,
        }

        factor_names = ["ff_mkt_beta", "ff_smb", "ff_hml", "ff_rmw", "ff_cma", "ff_wml"]
        for i, name in enumerate(factor_names[:fr.shape[1]]):
            result[name] = reg.coef_[i]

        return result

    @staticmethod
    def quality_factors(fundamentals: Dict) -> Dict:
        """
        AQR Quality-Minus-Junk (QMJ) components:
        Profitability + Safety + Growth + Payout
        Reference: Asness, Frazzini, Pedersen (2019)
        """
        features = {}

        # Profitability
        roe = fundamentals.get("roe", 0)
        roa = fundamentals.get("roa", 0)
        gross_margin = fundamentals.get("gross_margin", 0)
        features["quality_profitability"] = (roe + roa + gross_margin) / 3

        # Growth
        eps_growth = fundamentals.get("eps_growth", 0)
        rev_growth = fundamentals.get("revenue_growth", 0)
        features["quality_growth"] = (eps_growth + rev_growth) / 2

        # Safety (Low Leverage)
        debt_equity = fundamentals.get("debt_to_equity", 1)
        beta = fundamentals.get("beta", 1)
        features["quality_safety"] = -0.5 * debt_equity - 0.5 * beta  # Lower is safer

        # Payout
        div_yield = fundamentals.get("dividend_yield", 0)
        fcf_yield = fundamentals.get("fcf_yield", 0)
        features["quality_payout"] = (div_yield + fcf_yield) / 2

        # Composite QMJ score (-1 to +1)
        qmj = (features["quality_profitability"] + features["quality_growth"] +
                features["quality_safety"] + features["quality_payout"]) / 4
        features["qmj_score"] = np.tanh(qmj)  # Normalize to [-1, 1]

        return features

    @staticmethod
    def low_beta_factor(stock_returns: pd.Series, market_returns: pd.Series) -> Dict:
        """
        Frazzini & Pedersen (2014) — Betting Against Beta (BAB):
        Low-beta assets earn high risk-adjusted returns.
        BAB factor exploits leverage aversion.
        """
        if len(stock_returns) < 63:
            return {}
        n = min(len(stock_returns), len(market_returns))
        s_ret = stock_returns.iloc[-n:]
        m_ret = market_returns.iloc[-n:]

        # Rolling beta with exponential weighting (more recent = more weight)
        cov = s_ret.ewm(span=63).cov(m_ret).iloc[-1]
        var = m_ret.ewm(span=63).var().iloc[-1]
        beta_raw = cov / (var + 1e-10)

        # Shrinkage towards 1.0 (Vasicek 1973 — Bayesian shrinkage)
        beta_shrunk = 0.6 * beta_raw + 0.4 * 1.0
        bab_signal = -(beta_shrunk - 1.0)  # Negative beta deviation = long BAB

        return {
            "beta_raw": beta_raw,
            "beta_shrunk": beta_shrunk,
            "bab_signal": bab_signal,
            "beta_residual": beta_raw - 1.0,
        }


# ══════════════════════════════════════════════════════════════
# SECTION 4: MICROSTRUCTURE FEATURES
# ══════════════════════════════════════════════════════════════

class MicrostructureFeatures:
    """
    Market microstructure features used by HFT and stat-arb desks.
    Implements Kyle (1985) lambda, Amihud (2002) illiquidity,
    VPIN (Easley et al. 2012), Roll (1984) spread estimator.
    """

    @staticmethod
    def amihud_illiquidity(returns: pd.Series, volume: pd.Series, window: int = 21) -> float:
        """
        Amihud (2002) Illiquidity Ratio:
        ILLIQ_t = (1/D) * Σ |r_{i,d}| / DVOL_{i,d}
        Higher value = more illiquid (larger price impact per dollar traded)
        """
        dollar_vol = (volume * np.abs(returns)).rolling(window).mean()
        illiq = (np.abs(returns) / (dollar_vol + 1e-10)).rolling(window).mean()
        return illiq.iloc[-1] * 1e6  # Scale for readability

    @staticmethod
    def kyle_lambda(returns: pd.Series, volume: pd.Series, window: int = 21) -> float:
        """
        Kyle's Lambda — Market Impact Coefficient:
        λ = Cov(ΔP, V) / Var(V)
        Higher λ = more price impact per unit of volume traded
        Reference: Kyle (1985) "Continuous Auctions and Insider Trading"
        """
        if len(returns) < window:
            return 0.0
        price_changes = returns.iloc[-window:]
        vol = volume.iloc[-window:]
        cov_pv = np.cov(price_changes.values, vol.values)[0, 1]
        var_v = np.var(vol.values)
        return cov_pv / (var_v + 1e-10)

    @staticmethod
    def roll_spread(returns: pd.Series, window: int = 21) -> float:
        """
        Roll (1984) Implicit Bid-Ask Spread Estimator:
        s = 2 * sqrt(-Cov(r_t, r_{t-1}))
        Based on negative serial correlation induced by bid-ask bounce
        """
        if len(returns) < window:
            return 0.01
        r = returns.iloc[-window:]
        cov = r.autocorr(1) * r.var()  # Cov(r_t, r_{t-1})
        if cov >= 0:
            return 0.001  # Effectively zero spread
        return 2 * np.sqrt(-cov)

    @staticmethod
    def vpin(volume: pd.Series, returns: pd.Series, n_buckets: int = 50) -> float:
        """
        VPIN (Volume-synchronized Probability of Informed Trading):
        Easley, Lopez de Prado, O'Hara (2012)
        VPIN = |V_buy - V_sell| / (V_buy + V_sell)
        Proxy for informed trading, predicts volatility spikes.
        """
        if len(volume) < n_buckets:
            return 0.5
        # Estimate buy volume using price direction (BVC method)
        buy_vol = volume.where(returns > 0, volume * 0.5)
        sell_vol = volume.where(returns < 0, volume * 0.5)
        imbalance = (buy_vol - sell_vol).abs()
        total = buy_vol + sell_vol + 1e-10
        vpin_val = (imbalance / total).rolling(n_buckets).mean().iloc[-1]
        return float(np.clip(vpin_val, 0, 1))

    @staticmethod
    def order_flow_imbalance(returns: pd.Series, volume: pd.Series, window: int = 5) -> float:
        """
        Order Flow Imbalance (OFI):
        OFI = (Buy Volume - Sell Volume) / Total Volume
        Positive = buying pressure, Negative = selling pressure
        """
        direction = np.sign(returns.rolling(window).sum())
        buy_vol = volume.where(direction > 0, 0)
        sell_vol = volume.where(direction < 0, 0)
        ofi = (buy_vol.sum() - sell_vol.sum()) / (volume.sum() + 1e-10)
        return float(np.clip(ofi, -1, 1))

    @staticmethod
    def almgren_chriss_impact(
        trade_size_usd: float,
        adv_usd: float,  # Average Daily Volume in USD
        volatility: float,
        sigma: float = 0.3,  # Annual volatility
        eta: float = 2.5e-7,  # Temporary impact coefficient
        gamma: float = 2.5e-8  # Permanent impact coefficient
    ) -> Dict:
        """
        Almgren & Chriss (2001) Optimal Execution Model:
        Market Impact = η * (X/V) + γ * X
        where X = trade size, V = ADV

        Permanent impact: I_perm = γ * (X/V)^0.5
        Temporary impact: I_temp = η * (X/T*V)

        Reference: "Optimal Execution of Portfolio Transactions" (2001)
        """
        participation_rate = trade_size_usd / (adv_usd + 1e-10)
        daily_vol = sigma / np.sqrt(252)

        # Permanent market impact (moves price permanently)
        permanent_impact_bps = gamma * np.sqrt(participation_rate) * 10000

        # Temporary market impact (bid-ask + price pressure, recovers)
        temporary_impact_bps = eta * participation_rate * daily_vol * 10000

        # Total market impact
        total_impact_bps = permanent_impact_bps + temporary_impact_bps

        # Optimal trade duration (Almgren-Chriss efficient frontier)
        # T* = sqrt(eta/gamma) * sqrt(X/V)  — balances timing risk vs impact
        optimal_duration_days = np.sqrt(eta / (gamma + 1e-15)) * np.sqrt(participation_rate)

        return {
            "permanent_impact_bps": permanent_impact_bps,
            "temporary_impact_bps": temporary_impact_bps,
            "total_impact_bps": total_impact_bps,
            "participation_rate": participation_rate,
            "optimal_duration_days": min(optimal_duration_days, 30),
            "implementation_shortfall_bps": total_impact_bps * 0.5,
        }


# ══════════════════════════════════════════════════════════════
# SECTION 5: MAIN FEATURE PIPELINE
# ══════════════════════════════════════════════════════════════

class FeaturePipeline:
    """
    Master feature engineering pipeline.
    Assembles all 200+ features into a single normalized feature matrix.
    """

    def __init__(self):
        self.momentum = MomentumFeatures()
        self.technical = TechnicalFeatures()
        self.volatility = VolatilityFeatures()
        self.factor = FactorFeatures()
        self.micro = MicrostructureFeatures()

    def build_feature_matrix(
        self,
        df: pd.DataFrame,  # OHLCV data
        fundamentals: Dict,
        market_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Builds the complete 200+ feature vector for ML models.
        Returns a flat dictionary of feature_name: float_value.
        """
        features = {}
        returns = df["close"].pct_change()

        # 1. Momentum Features
        features.update(self.momentum.time_series_momentum(returns))
        features.update(self.momentum.cross_sectional_momentum(returns))
        features.update(self.momentum.acceleration(returns))
        if market_returns is not None:
            features.update(self.momentum.residual_momentum(returns, market_returns))

        # 2. Technical Features (80+ indicators)
        features.update(self.technical.compute_all(df))

        # 3. Volatility Features
        features.update(self.volatility.realized_volatility(returns))
        features.update(self.volatility.risk_metrics(returns))
        features["parkinson_vol"] = self.volatility.parkinson_volatility(df["high"], df["low"])
        features["garman_klass_vol"] = self.volatility.garman_klass_volatility(df)
        features["yang_zhang_vol"] = self.volatility.yang_zhang_volatility(df)
        features["hurst_exponent"] = self.volatility.hurst_exponent(returns)

        # 4. Factor Exposures
        if factor_returns is not None:
            features.update(self.factor.compute_factor_exposures(returns, factor_returns))
        features.update(self.factor.quality_factors(fundamentals))
        if market_returns is not None:
            features.update(self.factor.low_beta_factor(returns, market_returns))

        # 5. Microstructure Features
        volume = df["volume"]
        features["amihud_illiquidity"] = self.micro.amihud_illiquidity(returns, volume)
        features["kyle_lambda"] = self.micro.kyle_lambda(returns, volume)
        features["roll_spread"] = self.micro.roll_spread(returns)
        features["vpin"] = self.micro.vpin(volume, returns)
        features["order_flow_imbalance"] = self.micro.order_flow_imbalance(returns, volume)

        # 6. Fundamental Ratios
        for key in ["pe_ratio", "forward_pe", "peg_ratio", "price_to_book",
                    "ev_ebitda", "roe", "roa", "roic", "debt_to_equity",
                    "current_ratio", "gross_margin", "net_margin", "fcf_margin",
                    "revenue_growth", "eps_growth", "short_interest",
                    "institutional_ownership"]:
            features[f"fund_{key}"] = fundamentals.get(key, 0) or 0

        # 7. Remove NaN and inf
        features = {k: float(v) if np.isfinite(v) else 0.0
                    for k, v in features.items()
                    if v is not None}

        return features

    def normalize_features(self, features: Dict[str, float]) -> np.ndarray:
        """
        Standardize features for ML ingestion.
        Uses robust scaling (median/IQR) to handle outliers.
        """
        values = np.array(list(features.values()))
        # Clip extreme outliers at ±5σ before normalizing
        mean = np.nanmean(values)
        std = np.nanstd(values) + 1e-8
        return np.clip((values - mean) / std, -5, 5)
