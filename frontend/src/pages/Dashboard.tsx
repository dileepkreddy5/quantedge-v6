// ============================================================
// QuantEdge v6.0 — Dashboard
// PUBLIC — no login required to analyze
// Login only to save watchlist/portfolio
// ============================================================

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuthStore, api } from '../auth/authStore';
import toast from 'react-hot-toast';
import {
  SignalPanel,
  WallStreetPanel,
  PortfolioPanel,
  PerformancePanel,
  MLModelsPanel,
  VolatilityPanel,
  RegimePanel,
  OptionsPanel,
  SentimentPanel,
  MonteCarloPanel,
  RiskPanel,
  FundamentalsPanel,
  ScenarioPanel,
  Watchlist,
} from '../components/ui';
import PriceChart from '../components/charts/PriceChart';

const TABS = [
  { id: 'overview',     label: '⬡ OVERVIEW' },
  { id: 'ml',          label: '🧠 ML MODELS' },
  { id: 'volatility',  label: '📊 VOLATILITY' },
  { id: 'regime',      label: '🌡 REGIME' },
  { id: 'sentiment',   label: '💬 SENTIMENT' },
  { id: 'montecarlo',  label: '🎲 MONTE CARLO' },
  { id: 'risk',        label: '🛡 RISK' },
  { id: 'fundamental', label: '📋 FUNDAMENTALS' },
  { id: 'watchlist',   label: '★ WATCHLIST' },
  { id: 'wallstreet',  label: '🏦 WALL ST.' },
  { id: 'portfolio',   label: '⚖ PORTFOLIO' },
  { id: 'performance', label: '📈 PERFORMANCE' },
];

const QUICK_TICKERS = ['AAPL', 'NVDA', 'TSLA', 'SPY', 'QQQ', 'MSFT', 'AMZN', 'META', 'GOOGL', 'BRK-B'];

export default function Dashboard() {
  const navigate = useNavigate();
  const { logout, isAuthenticated } = useAuthStore();
  const [searchParams] = useSearchParams();
  const [ticker, setTicker] = useState('');
  const [inputTicker, setInputTicker] = useState('');

  // Auto-run analysis if ?ticker=XXX in URL (from landing page)
  useEffect(() => {
    const urlTicker = searchParams.get('ticker');
    if (urlTicker && urlTicker.trim()) {
      const sym = urlTicker.toUpperCase().trim();
      setInputTicker(sym);
      const timer = setTimeout(() => runAnalysis(sym), 500);
      return () => clearTimeout(timer);
    }
  }, []); // eslint-disable-line
  const [activeTab, setActiveTab] = useState('overview');
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState('');
  const [elapsed, setElapsed] = useState(0);
  const timerRef = useRef<any>(null);

  const runAnalysis = useCallback(async (sym?: string) => {
    const symbol = (sym || inputTicker).toUpperCase().trim();
    if (!symbol) return;
    setTicker(symbol);
    setInputTicker(symbol);
    setData(null);
    setLoading(true);
    setElapsed(0);
    setActiveTab('overview');

    // Progress messages
    const msgs = [
      'Fetching 10Y price history...',
      'Computing 200+ features...',
      'Running GARCH volatility model...',
      'Running HMM regime classifier...',
      'Running LSTM price prediction...',
      'Running XGBoost + LightGBM...',
      'Running FinBERT sentiment...',
      'Computing options GEX + Greeks...',
      'Running Monte Carlo (100K paths)...',
      'Assembling institutional report...',
    ];
    let msgIdx = 0;
    setLoadingMsg(msgs[0]);
    const msgTimer = setInterval(() => {
      msgIdx = (msgIdx + 1) % msgs.length;
      setLoadingMsg(msgs[msgIdx]);
    }, 3200);

    // Elapsed timer
    const start = Date.now();
    timerRef.current = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);

    try {
      const res = await api.post('/api/v6/analyze', {
        req: {
          ticker: symbol,
          include_options: true,
          include_sentiment: true,
          mc_paths: 100000,
        }
      });
      setData(res.data.data);
      toast.success(`Analysis complete: ${symbol}`, { icon: '✅' });
    } catch (err: any) {
      const msg = err?.response?.data?.detail || 'Analysis failed';
      toast.error(msg);
    } finally {
      clearInterval(msgTimer);
      clearInterval(timerRef.current);
      setLoading(false);
    }
  }, [inputTicker]);

  const handleLogout = async () => {
    await logout();
    navigate('/');
  };

  const signal = data?.overall_signal || 'NEUTRAL';
  const signalColor = signal.includes('BUY') ? '#22c55e' : signal.includes('SELL') ? '#ef4444' : '#f59e0b';

  return (
    <div style={{ minHeight: '100vh', background: '#1a0f0a', fontFamily: "'Outfit', sans-serif", color: '#f4e8d8' }}>
      
      {/* ── Header ── */}
      <header style={{
        borderBottom: '1px solid rgba(212,149,108,0.12)',
        background: 'rgba(26,15,10,0.95)',
        backdropFilter: 'blur(20px)',
        position: 'sticky', top: 0, zIndex: 100,
        padding: '0 20px',
      }}>
        <div style={{ maxWidth: 1600, margin: '0 auto', display: 'flex', alignItems: 'center', gap: 16, height: 56 }}>
          
          {/* Logo */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
            <div style={{
              fontFamily: "'Bebas Neue', sans-serif",
              fontSize: 22, letterSpacing: 5, color: '#daa520',
            }}>QUANTEDGE</div>
            <div style={{
              fontFamily: "'Fira Code', monospace",
              fontSize: 8, color: '#4a3428', letterSpacing: 2, paddingTop: 2,
            }}>v6.0</div>
            {data && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 4, marginLeft: 4 }}>
                <div style={{ width: 5, height: 5, borderRadius: '50%', background: '#22c55e', boxShadow: '0 0 6px #22c55e', animation: 'pulse 1.5s infinite' }} />
                <span style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: '#22c55e', letterSpacing: 1 }}>LIVE</span>
              </div>
            )}
          </div>

          {/* Search bar */}
          <div style={{ flex: 1, maxWidth: 480, display: 'flex', gap: 8 }}>
            <div style={{ flex: 1, position: 'relative' }}>
              <input
                value={inputTicker}
                onChange={e => setInputTicker(e.target.value.toUpperCase())}
                onKeyDown={e => e.key === 'Enter' && runAnalysis()}
                placeholder="Enter ticker  e.g. AAPL, NVDA, SPY..."
                style={{
                  width: '100%', background: '#241510',
                  border: '1px solid rgba(212,149,108,0.2)', borderRadius: 6,
                  color: '#f4e8d8', fontFamily: "'Fira Code', monospace",
                  fontSize: 12, padding: '8px 12px', outline: 'none',
                }}
              />
            </div>
            <button
              onClick={() => runAnalysis()}
              disabled={loading || !inputTicker}
              style={{
                background: loading ? '#3a2920' : 'linear-gradient(135deg,#daa520,#b8860b)',
                color: '#1a0f0a', fontFamily: "'Fira Code',monospace",
                fontWeight: 700, fontSize: 10, letterSpacing: 2,
                padding: '8px 18px', border: 'none', borderRadius: 6,
                cursor: loading || !inputTicker ? 'not-allowed' : 'pointer',
                opacity: !inputTicker ? 0.5 : 1,
                transition: 'all 0.15s', whiteSpace: 'nowrap',
              }}
            >
              {loading ? (
                <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ width: 10, height: 10, border: '2px solid #4a3428', borderTopColor: '#daa520', borderRadius: '50%', display: 'inline-block', animation: 'spin 0.7s linear infinite' }} />
                  {elapsed}s
                </span>
              ) : 'ANALYZE →'}
            </button>
          </div>

          {/* Quick tickers */}
          <div style={{ display: 'flex', gap: 4, flexWrap: 'nowrap', overflow: 'hidden' }}>
            {QUICK_TICKERS.slice(0, 6).map(t => (
              <button key={t} onClick={() => runAnalysis(t)}
                style={{
                  background: ticker === t ? 'rgba(218,165,32,0.15)' : 'transparent',
                  border: `1px solid ${ticker === t ? '#daa520' : 'rgba(212,149,108,0.15)'}`,
                  color: ticker === t ? '#daa520' : '#9d8b7a',
                  fontFamily: "'Fira Code',monospace", fontSize: 9,
                  padding: '4px 8px', borderRadius: 4, cursor: 'pointer',
                  transition: 'all 0.12s',
                }}
              >{t}</button>
            ))}
          </div>

          {/* Right side */}
          <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 12, flexShrink: 0 }}>
            {data && (
              <div style={{
                fontFamily: "'Fira Code',monospace", fontSize: 9,
                color: signalColor, letterSpacing: 2,
                background: `${signalColor}15`, padding: '4px 10px',
                border: `1px solid ${signalColor}40`, borderRadius: 4,
              }}>
                {signal} · {data.overall_score}/100
              </div>
            )}
            <button onClick={() => navigate('/')}
              style={{ background: 'none', border: 'none', color: '#4a3428', fontFamily: "'Fira Code',monospace", fontSize: 9, padding: '5px 6px', cursor: 'pointer', letterSpacing: 1 }}>
              ← HOME
            </button>
            {isAuthenticated ? (
              <button onClick={handleLogout}
                style={{ background: 'none', border: '1px solid rgba(212,149,108,0.2)', color: '#9d8b7a', fontFamily: "'Fira Code',monospace", fontSize: 9, padding: '5px 10px', borderRadius: 4, cursor: 'pointer', letterSpacing: 1 }}>
                LOGOUT
              </button>
            ) : (
              <button onClick={() => navigate('/login')}
                style={{ background: 'linear-gradient(135deg,#daa520,#b8860b)', border: 'none', color: '#1a0f0a', fontFamily: "'Fira Code',monospace", fontWeight: 700, fontSize: 9, padding: '5px 12px', borderRadius: 4, cursor: 'pointer', letterSpacing: 1 }}>
                LOGIN →
              </button>
            )}
          </div>
        </div>
      </header>

      {/* ── Loading overlay ── */}
      {loading && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(26,15,10,0.85)',
          display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
          zIndex: 200, backdropFilter: 'blur(12px)',
        }}>
          <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 40, color: '#daa520', letterSpacing: 8, marginBottom: 12 }}>
            ANALYZING {inputTicker}
          </div>
          <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 11, color: '#9d8b7a', letterSpacing: 2, marginBottom: 32 }}>
            {loadingMsg}
          </div>
          {/* Progress bar */}
          <div style={{ width: 360, height: 3, background: '#2d1e18', borderRadius: 2, overflow: 'hidden', marginBottom: 20 }}>
            <div style={{
              height: '100%', background: 'linear-gradient(90deg,#daa520,#22c55e)',
              borderRadius: 2, width: `${Math.min((elapsed / 45) * 100, 95)}%`,
              transition: 'width 1s ease',
            }} />
          </div>
          <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 10, color: '#4a3428' }}>
            {elapsed}s elapsed · 8 ML models · 200+ features
          </div>
        </div>
      )}

      <main style={{ maxWidth: 1600, margin: '0 auto', padding: '16px 20px' }}>

        {/* ── No data state ── */}
        {!data && !loading && (
          <div style={{ textAlign: 'center', paddingTop: 80 }}>
            <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 56, color: '#2d1e18', letterSpacing: 6, marginBottom: 12 }}>
              QUANTEDGE
            </div>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 11, color: '#4a3428', letterSpacing: 3, marginBottom: 40 }}>
              INSTITUTIONAL · QUANTITATIVE · ANALYTICS · v6.0
            </div>
            <div style={{ fontFamily: "'Outfit',sans-serif", color: '#9d8b7a', fontSize: 14, marginBottom: 40 }}>
              Enter a ticker above and run institutional-grade analysis
            </div>
            <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', justifyContent: 'center' }}>
              {QUICK_TICKERS.map(t => (
                <button key={t} onClick={() => runAnalysis(t)}
                  style={{
                    background: '#2d1e18', border: '1px solid rgba(212,149,108,0.2)',
                    color: '#d4c4b0', fontFamily: "'Fira Code',monospace",
                    fontSize: 12, padding: '10px 20px', borderRadius: 6,
                    cursor: 'pointer', transition: 'all 0.15s',
                  }}
                  onMouseOver={e => { (e.target as any).style.borderColor = '#daa520'; (e.target as any).style.color = '#daa520'; }}
                  onMouseOut={e => { (e.target as any).style.borderColor = 'rgba(212,149,108,0.2)'; (e.target as any).style.color = '#d4c4b0'; }}
                >{t}</button>
              ))}
            </div>

            {/* Feature grid */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginTop: 60, maxWidth: 900, margin: '60px auto 0' }}>
              {[
                { icon: '🧠', title: 'LSTM + TFT', desc: 'Deep learning return forecasts\n5W, 2W, 1M, 3M, 1Y horizons' },
                { icon: '🌡', title: 'HMM REGIME', desc: '5-state market regime detector\nBull/Bear/Mean-Revert' },
                { icon: '📊', title: 'GJR-GARCH', desc: 'Asymmetric volatility model\nVaR & CVaR with Student-t' },
                { icon: '💬', title: 'FINBERT NLP', desc: 'SEC filings + Reddit + News\nInstitutional sentiment signals' },
                { icon: '⚙', title: 'OPTIONS GEX', desc: 'Gamma exposure & vol surface\nMax pain & dealer flows' },
                { icon: '🎲', title: 'MONTE CARLO', desc: '100K paths, Merton jump diffusion\nFat-tailed return distributions' },
              ].map(f => (
                <div key={f.title} style={{
                  background: '#2d1e18', border: '1px solid rgba(212,149,108,0.1)',
                  borderRadius: 8, padding: '20px 16px', textAlign: 'left',
                }}>
                  <div style={{ fontSize: 24, marginBottom: 8 }}>{f.icon}</div>
                  <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 10, color: '#daa520', letterSpacing: 2, marginBottom: 4 }}>{f.title}</div>
                  <div style={{ fontFamily: "'Outfit',sans-serif", fontSize: 11, color: '#9d8b7a', whiteSpace: 'pre-line', lineHeight: 1.6 }}>{f.desc}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ── Data loaded ── */}
        {data && (
          <>
            {/* ── Ticker header ── */}
            <TickerHeader data={data} ticker={ticker} />

            {/* ── Tabs ── */}
            <div style={{ display: 'flex', gap: 2, borderBottom: '1px solid rgba(212,149,108,0.12)', marginBottom: 20, overflowX: 'auto' }}>
              {TABS.map(tab => (
                <button key={tab.id} onClick={() => setActiveTab(tab.id)}
                  style={{
                    fontFamily: "'Fira Code',monospace", fontSize: 9, letterSpacing: 1.5,
                    padding: '10px 14px', background: 'none', border: 'none',
                    borderBottom: `2px solid ${activeTab === tab.id ? '#daa520' : 'transparent'}`,
                    color: activeTab === tab.id ? '#daa520' : '#9d8b7a',
                    cursor: 'pointer', whiteSpace: 'nowrap', transition: 'all 0.12s',
                  }}>
                  {tab.label}
                </button>
              ))}
            </div>

            {/* ── Tab content ── */}
            <div style={{ animation: 'fadeIn 0.3s ease' }}>
              {activeTab === 'overview'    && <OverviewTab data={data} ticker={ticker} onAnalyze={runAnalysis} />}
              {activeTab === 'ml'          && <MLModelsPanel data={data} />}
              {activeTab === 'volatility'  && <VolatilityPanel data={data} />}
              {activeTab === 'regime'      && <RegimePanel data={data} />}
              {activeTab === 'sentiment'   && <SentimentPanel data={data} />}
              {activeTab === 'montecarlo'  && <MonteCarloPanel data={data} />}
              {activeTab === 'risk'        && <RiskPanel data={data} />}
              {activeTab === 'fundamental' && <FundamentalsPanel data={data} />}
              {activeTab === 'watchlist'   && <Watchlist onAnalyze={runAnalysis} />}
              {activeTab === 'wallstreet'  && <WallStreetPanel data={data} />}
              {activeTab === 'portfolio'   && <PortfolioPanel data={data} />}
              {activeTab === 'performance' && <PerformancePanel data={data} />}
            </div>
          </>
        )}
      </main>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;600;800&family=Fira+Code:wght@400;600&display=swap');
        @keyframes fadeIn { from{opacity:0;transform:translateY(6px);}to{opacity:1;transform:none;} }
        @keyframes pulse { 0%,100%{opacity:1;}50%{opacity:0.3;} }
        @keyframes spin { to{transform:rotate(360deg);} }
        * { scrollbar-width: thin; scrollbar-color: #3a2920 #1a0f0a; }
        input::placeholder { color: #4a3428; }
        input:focus { border-color: rgba(218,165,32,0.5) !important; outline: none; }
      `}</style>
    </div>
  );
}

// ── Ticker Header ─────────────────────────────────────────────
function TickerHeader({ data, ticker }: { data: any; ticker: string }) {
  const change = data.change || 0;
  const changePct = data.change_pct || 0;
  const isUp = change >= 0;
  const signal = data.overall_signal || 'NEUTRAL';
  const signalColor = signal.includes('BUY') ? '#22c55e' : signal.includes('SELL') ? '#ef4444' : '#f59e0b';

  return (
    <div style={{
      background: '#241510', border: '1px solid rgba(212,149,108,0.12)',
      borderRadius: 8, padding: '16px 20px', marginBottom: 20,
      display: 'flex', flexWrap: 'wrap', gap: 20, alignItems: 'center',
    }}>
      {/* Ticker + name */}
      <div>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
          <span style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 28, letterSpacing: 4, color: '#daa520' }}>{ticker}</span>
          <span style={{ fontFamily: "'Outfit',sans-serif", fontSize: 13, color: '#9d8b7a' }}>{data.name || ''}</span>
        </div>
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: '#4a3428', letterSpacing: 2 }}>
          {data.exchange || ''} · {data.sector || ''} · {data.industry || ''}
        </div>
      </div>

      {/* Price */}
      <div>
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 26, fontWeight: 700, color: '#f4e8d8' }}>
          ${data.price?.toFixed(2) || '—'}
        </div>
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 11, color: isUp ? '#22c55e' : '#ef4444' }}>
          {isUp ? '+' : ''}{change.toFixed(2)} ({isUp ? '+' : ''}{changePct.toFixed(2)}%)
        </div>
      </div>

      {/* Signal */}
      <div style={{
        padding: '8px 16px',
        background: `${signalColor}12`,
        border: `1px solid ${signalColor}40`,
        borderRadius: 6, textAlign: 'center',
      }}>
        <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 20, color: signalColor, letterSpacing: 3 }}>{signal}</div>
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: signalColor + 'aa', letterSpacing: 2 }}>COMPOSITE SIGNAL</div>
      </div>

      {/* Score gauge */}
      <div style={{ minWidth: 120 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
          <span style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: '#9d8b7a', letterSpacing: 1 }}>SCORE</span>
          <span style={{ fontFamily: "'Fira Code',monospace", fontSize: 12, color: signalColor, fontWeight: 700 }}>{data.overall_score || 50}/100</span>
        </div>
        <div style={{ height: 6, background: '#1a0f0a', borderRadius: 3, overflow: 'hidden' }}>
          <div style={{
            height: '100%', borderRadius: 3,
            width: `${data.overall_score || 50}%`,
            background: `linear-gradient(90deg, ${signalColor}88, ${signalColor})`,
            transition: 'width 1.2s cubic-bezier(0.4,0,0.2,1)',
          }} />
        </div>
      </div>

      {/* Key stats */}
      <div style={{ display: 'flex', gap: 20, marginLeft: 'auto', flexWrap: 'wrap' }}>
        {[
          { label: 'MKT CAP', value: formatLarge(data.market_cap) },
          { label: 'VOL (Ann)', value: pct(data.annual_vol) },
          { label: 'SHARPE', value: num2(data.sharpe_ratio) },
          { label: 'REGIME', value: (data.current_regime || '—').replace('_', ' ') },
          { label: '1Y PRED', value: pct((data.predicted_return_1y || 0) / 100) },
        ].map(s => (
          <div key={s.label} style={{ textAlign: 'center' }}>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: '#4a3428', letterSpacing: 2, marginBottom: 2 }}>{s.label}</div>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 12, color: '#d4c4b0', fontWeight: 600 }}>{s.value}</div>
          </div>
        ))}
      </div>

      {/* Data quality */}
      <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: '#4a3428', letterSpacing: 1 }}>
        DATA {data.data_quality?.score || 0}% · {data.analysis_duration_seconds || 0}s
      </div>
    </div>
  );
}

// ── Overview Tab ──────────────────────────────────────────────
function OverviewTab({ data, ticker, onAnalyze }: { data: any; ticker: string; onAnalyze: (t: string) => void }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: 12 }}>
      {/* Price chart - spans 8 cols */}
      <div style={{ gridColumn: 'span 8' }}>
        <PriceChart ticker={ticker} data={data} />
      </div>
      {/* Signal panel - spans 4 cols */}
      <div style={{ gridColumn: 'span 4' }}>
        <SignalPanel data={data} />
      </div>
      {/* ML Predictions - spans 6 */}
      <div style={{ gridColumn: 'span 6' }}>
        <MLSummary data={data} />
      </div>
      {/* Scenarios - spans 6 */}
      <div style={{ gridColumn: 'span 6' }}>
        <ScenarioPanel data={data} compact />
      </div>
      {/* GARCH vol - spans 4 */}
      <div style={{ gridColumn: 'span 4' }}>
        <GarchSummary data={data} />
      </div>
      {/* Regime - spans 4 */}
      <div style={{ gridColumn: 'span 4' }}>
        <RegimeSummary data={data} />
      </div>
      {/* Sentiment - spans 4 */}
      <div style={{ gridColumn: 'span 4' }}>
        <SentimentSummary data={data} />
      </div>
    </div>
  );
}

// ── Mini panels for overview ──────────────────────────────────
function MLSummary({ data }: { data: any }) {
  const preds = data.ml_predictions?.ensemble || {};
  const horizons = [
    { label: '1W', key: 'pred_5d' },
    { label: '2W', key: 'pred_10d' },
    { label: '1M', key: 'pred_21d' },
    { label: '3M', key: 'pred_63d' },
    { label: '1Y', key: 'pred_252d' },
  ];
  return (
    <Panel title="ML ENSEMBLE PREDICTIONS" icon="🧠">
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {horizons.map(h => {
          const val = preds[h.key] ?? null;
          const color = val === null ? '#4a3428' : val > 0 ? '#22c55e' : '#ef4444';
          return (
            <div key={h.key} style={{ flex: 1, minWidth: 64, background: '#1a0f0a', borderRadius: 6, padding: '10px 8px', textAlign: 'center', border: `1px solid ${color}30` }}>
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: '#9d8b7a', letterSpacing: 2, marginBottom: 4 }}>{h.label}</div>
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 14, fontWeight: 700, color }}>
                {val !== null ? `${val > 0 ? '+' : ''}${val?.toFixed(1)}%` : '—'}
              </div>
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 12, fontFamily: "'Fira Code',monospace", fontSize: 9, color: '#4a3428' }}>
        <span>CONFIDENCE: {pct1(preds.confidence)}</span>
        <span>DISAGREEMENT: {preds.model_disagreement?.toFixed(1) ?? '—'}%</span>
      </div>
    </Panel>
  );
}

function GarchSummary({ data }: { data: any }) {
  const g = data.garch || {};
  const rows = [
    { label: 'Annual Vol', value: pct(g.current_annual_vol) },
    { label: 'VaR 95% (1d)', value: pct(g.var_95_daily) },
    { label: 'CVaR 95% (1d)', value: pct(g.cvar_95_daily) },
    { label: 'Persistence α+β', value: num3(g.persistence) },
    { label: 'Leverage Effect', value: g.leverage_effect ? 'YES' : 'NO' },
    { label: 'Vol Regime', value: g.vol_regime || '—' },
  ];
  return (
    <Panel title="GJR-GARCH VOLATILITY" icon="📊">
      {rows.map(r => <DataRow key={r.label} label={r.label} value={r.value} />)}
    </Panel>
  );
}

function RegimeSummary({ data }: { data: any }) {
  const regime = data.regime || {};
  const probs = regime.regime_probabilities || {};
  const current = data.current_regime || 'UNKNOWN';
  const color = current.includes('BULL') ? '#22c55e' : current.includes('BEAR') ? '#ef4444' : '#f59e0b';
  return (
    <Panel title="HMM REGIME" icon="🌡">
      <div style={{ textAlign: 'center', padding: '8px 0 12px' }}>
        <div style={{ fontFamily: "'Bebas Neue',sans-serif", fontSize: 20, color, letterSpacing: 3 }}>
          {current.replace(/_/g, ' ')}
        </div>
        <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: '#9d8b7a' }}>
          {pct1(regime.confidence)} confidence · {Math.round(regime.expected_duration_days || 0)}d expected
        </div>
      </div>
      {Object.entries(probs).slice(0, 4).map(([name, p]: [string, any]) => (
        <div key={name} style={{ marginBottom: 5 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 2 }}>
            <span style={{ fontFamily: "'Fira Code',monospace", fontSize: 8, color: name === current ? '#daa520' : '#9d8b7a' }}>{name.replace(/_/g, ' ')}</span>
            <span style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: '#d4c4b0' }}>{pct1(p)}</span>
          </div>
          <div style={{ height: 3, background: '#1a0f0a', borderRadius: 2 }}>
            <div style={{ height: '100%', background: name === current ? color : '#3a2920', borderRadius: 2, width: `${(p || 0) * 100}%`, transition: 'width 1s ease' }} />
          </div>
        </div>
      ))}
    </Panel>
  );
}

function SentimentSummary({ data }: { data: any }) {
  const s = data.sentiment || {};
  const score = s.composite || 0;
  const color = score > 0.1 ? '#22c55e' : score < -0.1 ? '#ef4444' : '#f59e0b';
  const headlines = s.headlines || [];
  return (
    <Panel title="NLP SENTIMENT" icon="💬">
      <div style={{ display: 'flex', gap: 12, marginBottom: 12 }}>
        {[
          { label: 'NEWS', value: s.news?.score, sub: s.news?.label },
          { label: 'REDDIT', value: s.reddit?.score, sub: s.reddit?.label },
          { label: 'COMPOSITE', value: score, sub: s.label },
        ].map(item => (
          <div key={item.label} style={{ flex: 1, textAlign: 'center', background: '#1a0f0a', borderRadius: 6, padding: '8px 4px' }}>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 7, color: '#9d8b7a', letterSpacing: 1, marginBottom: 4 }}>{item.label}</div>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 13, color: (item.value ?? 0) > 0 ? '#22c55e' : '#ef4444', fontWeight: 700 }}>
              {item.value != null ? (item.value > 0 ? '+' : '') + item.value.toFixed(2) : '—'}
            </div>
            <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 7, color: '#4a3428' }}>{item.sub || ''}</div>
          </div>
        ))}
      </div>
      {headlines.slice(0, 2).map((h: string, i: number) => (
        <div key={i} style={{ fontFamily: "'Outfit',sans-serif", fontSize: 10, color: '#9d8b7a', borderLeft: '2px solid #3a2920', paddingLeft: 8, marginBottom: 6, lineHeight: 1.4 }}>
          {h.slice(0, 80)}{h.length > 80 ? '...' : ''}
        </div>
      ))}
    </Panel>
  );
}

// ── Shared components ─────────────────────────────────────────
function Panel({ title, icon, children }: { title: string; icon: string; children: React.ReactNode }) {
  return (
    <div style={{ background: '#241510', border: '1px solid rgba(212,149,108,0.1)', borderRadius: 8, padding: 16, height: '100%' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 14, paddingBottom: 10, borderBottom: '1px solid rgba(212,149,108,0.08)' }}>
        <span style={{ fontSize: 14 }}>{icon}</span>
        <span style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: '#9d8b7a', letterSpacing: 2 }}>{title}</span>
      </div>
      {children}
    </div>
  );
}

function DataRow({ label, value }: { label: string; value: string }) {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', padding: '4px 0', borderBottom: '1px solid rgba(212,149,108,0.06)' }}>
      <span style={{ fontFamily: "'Outfit',sans-serif", fontSize: 11, color: '#9d8b7a' }}>{label}</span>
      <span style={{ fontFamily: "'Fira Code',monospace", fontSize: 11, color: '#d4c4b0', fontWeight: 600 }}>{value}</span>
    </div>
  );
}

// ── Formatters ────────────────────────────────────────────────
function pct(v: number | null | undefined): string {
  if (v == null) return '—';
  return `${(v * 100).toFixed(2)}%`;
}
function pct1(v: number | null | undefined): string {
  if (v == null) return '—';
  return `${(v * 100).toFixed(1)}%`;
}
function num2(v: number | null | undefined): string {
  if (v == null) return '—';
  return v.toFixed(2);
}
function num3(v: number | null | undefined): string {
  if (v == null) return '—';
  return v.toFixed(3);
}
function formatLarge(v: number | null | undefined): string {
  if (!v) return '—';
  if (v >= 1e12) return `$${(v / 1e12).toFixed(2)}T`;
  if (v >= 1e9)  return `$${(v / 1e9).toFixed(1)}B`;
  if (v >= 1e6)  return `$${(v / 1e6).toFixed(1)}M`;
  return `$${v.toFixed(0)}`;
}
