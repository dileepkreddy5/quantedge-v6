// ============================================================
// QuantEdge v6.0 — Methodology
// How skill is measured and what the platform will not claim.
// ============================================================

import React from 'react';
import { useNavigate } from 'react-router-dom';
import PanelSkillMatrix from '../components/ui/PanelSkillMatrix';

const C = {
  s0: '#100a07', s2: '#241610', b1: '#3a2920', gold: '#daa520',
  caramel: '#d4956c', cocoa: '#8a7560', dust: '#9d8b7a',
  latte: '#d4c4b0', cream: '#f4e8d8', warn: '#f59e0b',
};
const mono = "'Fira Code',monospace";

const LIMITS = [
  ['OPTIONS ANALYTICS', 'Not available. The Polygon Stocks Starter plan returns 403 for the options chain. Gamma exposure and vol-surface metrics are therefore not computed rather than estimated.'],
  ['PRICE HISTORY', 'Five years of daily aggregates — the plan ceiling. Long-horizon models cannot be validated on a window shorter than several times their forecast horizon.'],
  ['FORECAST SKILL', 'Published per horizon with its sample size and t-statistic. Horizons that have not cleared significance are labelled as such on every prediction.'],
  ['SUPPLY-CHAIN GRAPH', 'Curated, not universe-wide. 10-K filings disclose customer concentration without naming the customer, so comprehensive supplier graphs are a paid product.'],
  ['ACCESS MODEL', 'Single-owner platform. Analysis is public; watchlist and portfolio require the owner account. Auth sits behind an interface so a managed identity provider is one class, not a rewrite.'],
];

const PIPELINE = [
  ['01', 'DATA', 'Price, fundamentals and news fetched concurrently from Polygon; SEC bulk fundamentals read locally from the nightly companyfacts archive.'],
  ['02', 'FEATURES', '152 cross-sectional-rank features survive the dead-and-constant filter each night.'],
  ['03', 'LABELING', 'Forward returns at six horizons, plus triple-barrier meta-labels for signal quality.'],
  ['04', 'ALPHA MODELS', 'Panel ensemble, GJR-GARCH, HMM regime, Kalman trend, Monte Carlo, FinBERT.'],
  ['05', 'RISK ENGINE', 'CVaR under historical and Cornish-Fisher methods, position limits, volatility targeting.'],
  ['06', 'CONSTRUCTION', 'Volatility scaling and a drawdown governor that cuts leverage as drawdown deepens.'],
];

export default function Methodology() {
  const navigate = useNavigate();
  return (
    <div style={{ minHeight: '100vh', background: C.s0, color: C.cream,
                  fontFamily: "'Outfit',sans-serif", paddingTop: 90 }}>
      <div style={{ maxWidth: 1400, margin: '0 auto', padding: '0 4rem 40px' }}>
        <button onClick={() => navigate('/')} style={{
          background: 'none', border: `1px solid ${C.b1}`, color: C.dust,
          fontFamily: mono, fontSize: 10, letterSpacing: 1.5, padding: '8px 14px',
          borderRadius: 4, cursor: 'pointer', marginBottom: 30,
        }}>← BACK</button>

        <div style={{ fontFamily: mono, fontSize: 10, letterSpacing: 3, color: C.cocoa, marginBottom: 12 }}>
          METHODOLOGY
        </div>
        <h1 style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 68, letterSpacing: 2,
                     margin: '0 0 20px', lineHeight: 1 }}>
          HOW SKILL IS MEASURED
        </h1>
        <p style={{ color: C.latte, fontSize: 16.5, lineHeight: 1.8, maxWidth: 820, marginBottom: 10 }}>
          Forecast quality here is cross-sectional rank information coefficient on held-out
          dates the models never saw — not a backtested return curve. Significance is a
          Newey-West t on the per-date IC series, with the standard error widened to absorb
          the autocorrelation that overlapping forward windows induce.
        </p>
        <p style={{ color: C.dust, fontSize: 15, lineHeight: 1.8, maxWidth: 820, marginBottom: 50 }}>
          Both the full-sample figure and the non-overlapping subsample are published side
          by side, each with its own date count. Where fewer than five independent windows
          fit inside the validation period, the independent estimate is reported as
          unmeasurable rather than as a number — a mean of two Spearman coefficients is not
          an information coefficient.
        </p>
      </div>

      <PanelSkillMatrix />

      <div style={{ maxWidth: 1400, margin: '0 auto', padding: '0 4rem 80px' }}>
        <h2 style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 44, letterSpacing: 2,
                     margin: '60px 0 24px' }}>
          THE PIPELINE
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(320px,1fr))', gap: 14 }}>
          {PIPELINE.map(([n, t, d]) => (
            <div key={n} style={{
              background: 'linear-gradient(165deg,#241610,#180e0a)',
              border: `1px solid rgba(212,149,108,0.14)`, borderRadius: 8, padding: '26px 24px',
            }}>
              <div style={{
                fontFamily: "'Bebas Neue',sans-serif", fontSize: 54, lineHeight: 1, marginBottom: 12,
                background: 'linear-gradient(160deg,#daa520,#8a5a1e 55%,#3a2920)',
                WebkitBackgroundClip: 'text', backgroundClip: 'text',
                WebkitTextFillColor: 'transparent', color: '#8a5a1e',
              }}>{n}</div>
              <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 2, color: C.gold, marginBottom: 10 }}>{t}</div>
              <div style={{ fontSize: 14, color: C.latte, lineHeight: 1.75 }}>{d}</div>
            </div>
          ))}
        </div>

        <h2 style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 44, letterSpacing: 2,
                     margin: '70px 0 10px' }}>
          WHAT THIS WILL NOT CLAIM
        </h2>
        <p style={{ color: C.dust, fontSize: 15, lineHeight: 1.8, maxWidth: 820, marginBottom: 26 }}>
          Every limit below is a measured constraint, not a disclaimer. Where data is
          unavailable, the signal is marked as needing a source rather than estimated.
        </p>
        <div style={{ display: 'grid', gap: 10 }}>
          {LIMITS.map(([k, v]) => (
            <div key={k} style={{
              display: 'grid', gridTemplateColumns: '220px 1fr', gap: 24, padding: '20px 24px',
              background: 'linear-gradient(90deg,#241610,#100a07)',
              border: `1px solid ${C.b1}`, borderLeft: `3px solid ${C.warn}`, borderRadius: 8,
            }}>
              <div style={{ fontFamily: mono, fontSize: 11, letterSpacing: 1.6, color: C.caramel }}>{k}</div>
              <div style={{ fontSize: 14.5, color: C.latte, lineHeight: 1.75 }}>{v}</div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
