// ============================================================
// QuantEdge v6.0 — Landing Page for quant.dileepkapu.com
// Matches dileepkapu.com espresso/chocolate aesthetic
// Public — no auth required
// ============================================================

import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';

const GRAIN = `url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='3' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`;

const MODELS = [
  { icon: '🧠', name: 'LSTM + TFT',      desc: 'Deep learning price forecasts across 5 horizons' },
  { icon: '🌡', name: 'HMM REGIME',      desc: '5-state market regime detection in real time' },
  { icon: '📊', name: 'GJR-GARCH',       desc: 'Asymmetric volatility — VaR, CVaR, tail risk' },
  { icon: '💬', name: 'FINBERT NLP',     desc: 'SEC filings, Reddit & news sentiment signals' },
  { icon: '⚙',  name: 'OPTIONS GEX',    desc: 'Gamma exposure, vol surface, dealer flows' },
  { icon: '🎲', name: 'MONTE CARLO',     desc: '100K Merton jump-diffusion paths' },
  { icon: '📐', name: 'TRIPLE BARRIER',  desc: 'Meta-labeling for ML signal quality' },
  { icon: '🛡', name: 'RISK ENGINE',     desc: 'Drawdown, CVaR, Kelly, portfolio optimization' },
];

const TICKERS = ['AAPL','NVDA','TSLA','SPY','QQQ','MSFT','AMZN','META','GOOGL','BRK-B'];

export default function Landing() {
  const navigate = useNavigate();
  const [visible, setVisible] = useState(false);
  const [activeTicker, setActiveTicker] = useState('NVDA');

  useEffect(() => {
    setTimeout(() => setVisible(true), 100);
  }, []);

  const go = (ticker?: string) => {
    navigate(`/dashboard${ticker ? `?ticker=${ticker}` : ''}`);
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: '#1a0f0a',
      fontFamily: "'Outfit', sans-serif",
      color: '#f4e8d8',
      overflowX: 'hidden',
    }}>
      {/* Grain */}
      <div style={{ position: 'fixed', inset: 0, backgroundImage: GRAIN, opacity: 0.025, pointerEvents: 'none', zIndex: 0 }} />

      {/* Ambient glow */}
      <div style={{
        position: 'fixed', inset: 0, zIndex: 0,
        background: `radial-gradient(ellipse at 15% 25%, rgba(212,149,108,0.07) 0%, transparent 55%),
                     radial-gradient(ellipse at 85% 75%, rgba(218,165,32,0.05) 0%, transparent 55%)`,
      }} />

      {/* ── Nav ── */}
      <nav style={{
        position: 'fixed', top: 0, left: 0, right: 0, zIndex: 100,
        background: 'rgba(26,15,10,0.95)', backdropFilter: 'blur(20px)',
        borderBottom: '1px solid rgba(212,149,108,0.12)',
        padding: '0 4rem',
      }}>
        <div style={{ maxWidth: 1400, margin: '0 auto', display: 'flex', alignItems: 'center', height: 64 }}>
          <div style={{
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: 22, letterSpacing: 6, color: '#daa520',
          }}>QUANTEDGE</div>
          <div style={{
            fontFamily: "'Fira Code', monospace",
            fontSize: 8, color: '#4a3428', letterSpacing: 2, marginLeft: 10, paddingTop: 2,
          }}>v6.0</div>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 24, alignItems: 'center' }}>
            <a href="https://dileepkapu.com" target="_blank" rel="noopener noreferrer"
              style={{ color: '#9d8b7a', fontSize: 12, fontWeight: 600, textDecoration: 'none', letterSpacing: 1 }}>
              dileepkapu.com
            </a>
            <button onClick={() => navigate('/login')} style={{
              background: 'none', border: '1px solid rgba(212,149,108,0.3)',
              color: '#d4c4b0', fontFamily: "'Fira Code', monospace",
              fontSize: 10, letterSpacing: 2, padding: '7px 16px', borderRadius: 4,
              cursor: 'pointer',
            }}>LOGIN</button>
            <button onClick={() => go()} style={{
              background: 'linear-gradient(135deg,#daa520,#b8860b)',
              border: 'none', color: '#1a0f0a',
              fontFamily: "'Fira Code', monospace", fontWeight: 700,
              fontSize: 10, letterSpacing: 2, padding: '8px 20px', borderRadius: 4,
              cursor: 'pointer',
            }}>LAUNCH →</button>
          </div>
        </div>
      </nav>

      {/* ── Hero ── */}
      <section style={{
        position: 'relative', zIndex: 1,
        maxWidth: 1400, margin: '0 auto',
        padding: '140px 4rem 100px',
        opacity: visible ? 1 : 0,
        transform: visible ? 'none' : 'translateY(24px)',
        transition: 'opacity 0.8s ease, transform 0.8s ease',
      }}>
        {/* Eyebrow */}
        <div style={{
          fontFamily: "'Fira Code', monospace", fontSize: 10,
          color: '#9d8b7a', letterSpacing: 4,
          textTransform: 'uppercase', marginBottom: 24,
          display: 'flex', alignItems: 'center', gap: 10,
        }}>
          <span style={{ display: 'inline-block', width: 32, height: 1, background: '#daa520' }} />
          INSTITUTIONAL · QUANTITATIVE · ANALYTICS
        </div>

        {/* Headline */}
        <h1 style={{
          fontFamily: "'Bebas Neue', sans-serif",
          fontSize: 'clamp(48px, 8vw, 100px)',
          lineHeight: 0.95, letterSpacing: 2,
          color: '#f4e8d8', margin: '0 0 32px',
        }}>
          INSTITUTIONAL<br />
          <span style={{ color: '#daa520' }}>QUANT</span> ANALYTICS<br />
          FOR EVERY STOCK
        </h1>

        {/* Subhead */}
        <p style={{
          fontSize: 18, color: '#9d8b7a', maxWidth: 540,
          lineHeight: 1.7, marginBottom: 48,
        }}>
          8 ML models. 200+ signals. GJR-GARCH volatility, HMM regime detection,
          FinBERT sentiment, Monte Carlo risk — the same tools used by quant hedge funds.
          <span style={{ color: '#d4c4b0' }}> Free. No login required.</span>
        </p>

        {/* CTA row */}
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 48 }}>
          <button onClick={() => go()} style={{
            background: 'linear-gradient(135deg,#daa520,#b8860b)',
            border: 'none', color: '#1a0f0a',
            fontFamily: "'Fira Code', monospace", fontWeight: 700,
            fontSize: 12, letterSpacing: 3, padding: '16px 36px',
            borderRadius: 4, cursor: 'pointer',
            boxShadow: '0 8px 30px rgba(218,165,32,0.25)',
            transition: 'all 0.2s',
          }}>
            ANALYZE A STOCK →
          </button>
          <button onClick={() => go('SPY')} style={{
            background: 'none', border: '1px solid rgba(212,149,108,0.3)',
            color: '#d4c4b0', fontFamily: "'Fira Code', monospace",
            fontSize: 11, letterSpacing: 2, padding: '16px 28px',
            borderRadius: 4, cursor: 'pointer',
          }}>
            TRY SPY DEMO
          </button>
        </div>

        {/* Quick tickers */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 16 }}>
          <span style={{ fontFamily: "'Fira Code', monospace", fontSize: 9, color: '#4a3428', letterSpacing: 2, marginRight: 4, alignSelf: 'center' }}>QUICK ANALYZE:</span>
          {TICKERS.map(t => (
            <button key={t} onClick={() => go(t)}
              onMouseOver={e => { (e.target as any).style.borderColor='#daa520'; (e.target as any).style.color='#daa520'; }}
              onMouseOut={e => { (e.target as any).style.borderColor='rgba(212,149,108,0.2)'; (e.target as any).style.color='#9d8b7a'; }}
              style={{
                background: 'none', border: '1px solid rgba(212,149,108,0.2)',
                color: '#9d8b7a', fontFamily: "'Fira Code', monospace",
                fontSize: 10, padding: '6px 12px', borderRadius: 4, cursor: 'pointer',
                transition: 'all 0.12s',
              }}>
              {t}
            </button>
          ))}
        </div>

        {/* Trust bar */}
        <div style={{
          marginTop: 20,
          fontFamily: "'Fira Code', monospace", fontSize: 9,
          color: '#4a3428', letterSpacing: 2,
          display: 'flex', gap: 24, flexWrap: 'wrap',
        }}>
          <span>✓ NO LOGIN TO USE</span>
          <span>✓ 8 ML MODELS</span>
          <span>✓ 200+ SIGNALS</span>
          <span>✓ 10Y PRICE HISTORY</span>
          <span>✓ OPTIONS GEX</span>
          <span>✓ REAL-TIME STREAM</span>
        </div>
      </section>

      {/* ── Models grid ── */}
      <section style={{ position: 'relative', zIndex: 1, padding: '80px 4rem' }}>
        <div style={{ maxWidth: 1400, margin: '0 auto' }}>
          <div style={{ marginBottom: 48 }}>
            <div style={{ fontFamily: "'Fira Code', monospace", fontSize: 9, color: '#9d8b7a', letterSpacing: 4, marginBottom: 12 }}>
              THE ENGINE
            </div>
            <h2 style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 48, color: '#f4e8d8', letterSpacing: 2, margin: 0 }}>
              8 INSTITUTIONAL ML MODELS
            </h2>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
            gap: 16,
          }}>
            {MODELS.map((m, i) => (
              <div key={m.name}
                style={{
                  background: '#241510',
                  border: '1px solid rgba(212,149,108,0.1)',
                  borderRadius: 8, padding: '24px 20px',
                  opacity: visible ? 1 : 0,
                  transform: visible ? 'none' : 'translateY(16px)',
                  transition: `opacity 0.6s ease ${0.05 * i}s, transform 0.6s ease ${0.05 * i}s`,
                  cursor: 'pointer',
                }}
                onClick={() => go()}
                onMouseOver={e => { (e.currentTarget as any).style.borderColor='rgba(212,149,108,0.35)'; (e.currentTarget as any).style.background='#2d1e18'; }}
                onMouseOut={e => { (e.currentTarget as any).style.borderColor='rgba(212,149,108,0.1)'; (e.currentTarget as any).style.background='#241510'; }}
              >
                <div style={{ fontSize: 28, marginBottom: 12 }}>{m.icon}</div>
                <div style={{ fontFamily: "'Fira Code', monospace", fontSize: 10, color: '#daa520', letterSpacing: 2, marginBottom: 8 }}>
                  {m.name}
                </div>
                <div style={{ fontSize: 13, color: '#9d8b7a', lineHeight: 1.6 }}>{m.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── How it works ── */}
      <section style={{ position: 'relative', zIndex: 1, padding: '80px 4rem' }}>
        <div style={{ maxWidth: 1400, margin: '0 auto' }}>
          <div style={{ marginBottom: 48 }}>
            <div style={{ fontFamily: "'Fira Code', monospace", fontSize: 9, color: '#9d8b7a', letterSpacing: 4, marginBottom: 12 }}>HOW IT WORKS</div>
            <h2 style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 48, color: '#f4e8d8', letterSpacing: 2, margin: 0 }}>
              THREE STEPS
            </h2>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 2 }}>
            {[
              { n: '01', title: 'ENTER ANY TICKER', desc: 'Type any US stock symbol. No account needed. Completely free.' },
              { n: '02', title: '45 SECONDS', desc: '8 ML models run in parallel. 200+ features computed. 10 years of price history analyzed.' },
              { n: '03', title: 'INSTITUTIONAL REPORT', desc: 'ML forecasts, regime state, risk metrics, sentiment scores, options flow. The full picture.' },
            ].map((s, i) => (
              <div key={s.n} style={{
                padding: '40px 32px',
                borderLeft: i === 0 ? '4px solid #daa520' : '1px solid rgba(212,149,108,0.15)',
              }}>
                <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 64, color: '#2d1e18', lineHeight: 1, marginBottom: 16 }}>{s.n}</div>
                <div style={{ fontFamily: "'Fira Code', monospace", fontSize: 12, color: '#daa520', letterSpacing: 2, marginBottom: 10 }}>{s.title}</div>
                <div style={{ fontSize: 14, color: '#9d8b7a', lineHeight: 1.7 }}>{s.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA band ── */}
      <section style={{
        position: 'relative', zIndex: 1,
        margin: '0 4rem 80px',
        background: '#241510',
        border: '1px solid rgba(212,149,108,0.15)',
        borderLeft: '4px solid #daa520',
        borderRadius: 8,
        padding: '60px 48px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        flexWrap: 'wrap', gap: 32,
        maxWidth: 1400,
      }}>
        <div>
          <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 36, color: '#f4e8d8', letterSpacing: 2, marginBottom: 8 }}>
            READY TO ANALYZE YOUR FIRST STOCK?
          </div>
          <div style={{ fontFamily: "'Outfit', sans-serif", fontSize: 14, color: '#9d8b7a' }}>
            Free. No login. Institutional-grade quant analysis in 45 seconds.
          </div>
        </div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap' }}>
          {['AAPL','NVDA','TSLA','SPY'].map(t => (
            <button key={t} onClick={() => go(t)} style={{
              background: t === 'NVDA' ? 'linear-gradient(135deg,#daa520,#b8860b)' : '#2d1e18',
              border: `1px solid ${t === 'NVDA' ? 'transparent' : 'rgba(212,149,108,0.2)'}`,
              color: t === 'NVDA' ? '#1a0f0a' : '#d4c4b0',
              fontFamily: "'Fira Code', monospace", fontWeight: t === 'NVDA' ? 700 : 400,
              fontSize: 11, letterSpacing: 2, padding: '12px 22px',
              borderRadius: 4, cursor: 'pointer',
            }}>
              {t} →
            </button>
          ))}
        </div>
      </section>

      {/* ── Footer ── */}
      <footer style={{
        position: 'relative', zIndex: 1,
        background: '#2d1e18',
        borderTop: '1px solid rgba(212,149,108,0.1)',
        padding: '32px 4rem',
      }}>
        <div style={{ maxWidth: 1400, margin: '0 auto', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 16 }}>
          <div>
            <div style={{ fontFamily: "'Bebas Neue', sans-serif", fontSize: 18, letterSpacing: 4, color: '#daa520', marginBottom: 4 }}>QUANTEDGE</div>
            <div style={{ fontFamily: "'Fira Code', monospace", fontSize: 9, color: '#4a3428', letterSpacing: 2 }}>
              Built by{' '}
              <a href="https://dileepkapu.com" target="_blank" rel="noopener noreferrer"
                style={{ color: '#9d8b7a', textDecoration: 'none' }}>
                Dileep Kumar Reddy Kapu
              </a>
            </div>
          </div>
          <div style={{ fontFamily: "'Fira Code', monospace", fontSize: 9, color: '#4a3428', letterSpacing: 1, textAlign: 'right' }}>
            © 2026 · Not financial advice · For research only
          </div>
        </div>
      </footer>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;600;800&family=Fira+Code:wght@400;600&display=swap');
        * { scrollbar-width: thin; scrollbar-color: #3a2920 #1a0f0a; }
        html { scroll-behavior: smooth; }
      `}</style>
    </div>
  );
}
