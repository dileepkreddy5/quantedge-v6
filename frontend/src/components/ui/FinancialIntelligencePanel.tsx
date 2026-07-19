import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

const EXPLAIN: Record<string,string> = {
  Piotroski: '9-point financial-health test. 7-9 = strong (profitable, improving, low-debt); 0-3 = weak.',
  'Altman Z': 'Bankruptcy-risk gauge. Above 3 = safe, below 1.8 = distress. Higher is safer.',
  'Beneish M': 'Earnings-manipulation detector. Below -1.78 = clean books; above = worth scrutiny.',
  'ROIC (ex-GW)': 'Return on capital actually deployed, excluding acquisition goodwill. Above ~10% is good; higher = compounds capital efficiently.',
  'ROIC−WACC': 'How much the firm earns above its cost of capital. Positive = creating value; bigger is better.',
  'Owner Earnings': "Buffett's measure of true owner cash: profit + depreciation − maintenance capex.",
  'FCF Margin': 'Share of revenue that becomes free cash. Above ~10% is strong — cash left after running the business.',
  'Cash Cycle': 'Days to turn operations into cash. Negative = suppliers fund your operations (a strength).',
  'Shareholder Yield': 'Dividends + buybacks as % of market cap — total cash returned to shareholders.',
};



function heat(score: number | null): string {
  if (score == null) return '#2a2a2a';
  if (score >= 85) return '#0f6e56';
  if (score >= 70) return '#1d9e75';
  if (score >= 55) return '#8a7519';
  if (score >= 40) return '#a35a1d';
  return '#7a2320';
}
function heatText(score: number | null): string {
  if (score == null) return '#777';
  if (score >= 70) return '#d8f5ea';
  if (score >= 40) return '#f5e0c0';
  return '#f5c0c0';
}
function fmt(id: string, v: number | null): string {
  if (v == null) return '—';
  const pct = ['gross_margin','operating_margin','net_margin','ebitda_margin','fcf_margin',
    'roic','roic_ex_goodwill','roic_wacc_spread','roe','roa','revenue_growth','fcf_growth',
    'revenue_cagr_3y','earnings_cagr_3y','rd_intensity','cogs_ratio','capex_intensity',
    'shareholder_yield','buyback_yield','dividend_yield','owner_earnings_yield',
    'effective_tax_rate','goodwill_ratio','equity_ratio','reinvestment_rate','accruals_ratio'];
  const days = ['dso','dio','dpo','cash_conversion_cycle'];
  const ratio = ['current_ratio','quick_ratio','cash_ratio','debt_to_equity','debt_to_ebitda',
    'net_debt_to_ebitda','asset_turnover','asset_turnover_eff','equity_multiplier',
    'fcf_conversion','ocf_to_net_income','earnings_quality','dividend_coverage','sbc_dilution_ratio'];
  if (pct.includes(id)) return (v * 100).toFixed(1) + '%';
  if (days.includes(id)) return v.toFixed(1) + ' days';
  if (ratio.includes(id)) return v.toFixed(2) + '×';
  if (id === 'piotroski_f') return v.toFixed(0) + '/9';
  if (id === 'altman_z' || id === 'beneish_m') return v.toFixed(2);
  if (id === 'share_count_trend') return (v * 100).toFixed(2) + '%/q';
  const dollarsB = ['net_debt','enterprise_value'];
  if (dollarsB.includes(id)) {
    if (Math.abs(v) >= 1e12) return '$' + (v/1e12).toFixed(2) + 'T';
    if (Math.abs(v) >= 1e9) return '$' + (v/1e9).toFixed(1) + 'B';
    return '$' + (v/1e6).toFixed(0) + 'M';
  }
  const perShare = ['book_value_per_share','tangible_bvps'];
  if (perShare.includes(id)) return '$' + v.toFixed(2);
  if (id === 'interest_coverage') return v.toFixed(1) + '×';
  if (id === 'deferred_rev_to_revenue' || id === 'deferred_rev_growth' ||
      id === 'revenue_cagr_5y' || id === 'earnings_cagr_5y' || id === 'retained_earnings_ratio') return (v*100).toFixed(1) + '%';
  if (id === 'ev_ebitda' || id === 'ev_revenue' || id === 'price_to_book' || id === 'adj_debt_to_ebitda') return v.toFixed(2) + '×';
  return v.toFixed(3);
}
interface Sig { id: string; label: string; weight: number; status: string; evidence: string;
  raw_value: number | null; score: number | null; method?: string; }
interface Cat { id: string; label: string; weight: number; score: number | null;
  confidence: number; n_signals: number; n_scored: number; signals: Sig[]; }
interface Tree { label: string; weight: number; score: number | null; confidence: number; categories: Cat[]; }
interface FinData { ticker: string; available: boolean; score: number | null; confidence: number;
  weight_in_conviction: number; coverage: { scored: number; total: number };
  market_cap: number | null; wacc_used: number; n_quarters: number; tree: Tree;
  key_metrics: Record<string, number | null>; reason?: string; }
function bn(v: number | null): string {
  if (v == null) return '—';
  if (Math.abs(v) >= 1e12) return '$' + (v / 1e12).toFixed(2) + 'T';
  if (Math.abs(v) >= 1e9) return '$' + (v / 1e9).toFixed(1) + 'B';
  if (Math.abs(v) >= 1e6) return '$' + (v / 1e6).toFixed(0) + 'M';
  return '$' + v.toFixed(0);
}
export default function FinancialIntelligencePanel({ ticker }: { ticker: string }) {
  const [data, setData] = useState<FinData | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState('');
  const [openSet, setOpenSet] = useState<Set<string>>(new Set());
  const [allOpen, setAllOpen] = useState(true);
  useEffect(() => {
    if (!ticker) return;
    setLoading(true); setErr(''); setData(null);
    api.get(`/api/v6/financial/${ticker}`)
      .then(r => { const d = r.data?.data; if (!d?.available) setErr(d?.reason || 'No financial data'); else setData(d); })
      .catch(e => setErr(e?.message || 'Request failed'))
      .finally(() => setLoading(false));
  }, [ticker]);

  useEffect(() => {
    if (data) { setOpenSet(new Set(data.tree.categories.map(c => c.id))); setAllOpen(true); }
  }, [data]);

  const isOpen = (id: string) => openSet.has(id);
  const toggle = (id: string) => setOpenSet(prev => {
    const n = new Set(prev); n.has(id) ? n.delete(id) : n.add(id); return n; });
  const toggleAll = () => {
    if (!data) return;
    if (allOpen) { setOpenSet(new Set()); setAllOpen(false); }
    else { setOpenSet(new Set(data.tree.categories.map(c => c.id))); setAllOpen(true); }
  };

  if (!ticker) return <div style={{ color: '#9d8b7a', padding: 24 }}>Enter a ticker to load Financial Intelligence.</div>;
  if (loading) return <div style={{ color: '#daa520', padding: 24 }}>Computing Financial Intelligence — hybrid Polygon + SEC EDGAR, 58 signals, 6 models…</div>;
  if (err) return <div style={{ color: '#c0705a', padding: 24 }}>Financial Intelligence: {err}</div>;
  if (!data) return null;
  const km = data.key_metrics;
  const cards = [
    { k: 'ROIC (ex-GW)', v: fmt('roic_ex_goodwill', km.roic_ex_goodwill) },
    { k: 'ROIC−WACC', v: fmt('roic_wacc_spread', km.roic_wacc_spread) },
    { k: 'Owner Earnings', v: bn(km.owner_earnings) },
    { k: 'FCF Margin', v: fmt('fcf_margin', km.fcf_margin) },
    { k: 'Cash Cycle', v: fmt('cash_conversion_cycle', km.cash_conversion_cycle) },
    { k: 'Shareholder Yield', v: fmt('shareholder_yield', km.shareholder_yield) },
  ];
  const badges = [
    { k: 'Piotroski', v: fmt('piotroski_f', km.piotroski_f), c: (km.piotroski_f ?? 0) >= 7 ? '#1d9e75' : '#8a7519' },
    { k: 'Altman Z', v: fmt('altman_z', km.altman_z), c: (km.altman_z ?? 0) >= 3 ? '#1d9e75' : (km.altman_z ?? 0) >= 1.8 ? '#8a7519' : '#7a2320' },
    { k: 'Beneish M', v: fmt('beneish_m', km.beneish_m), c: (km.beneish_m ?? 0) < -1.78 ? '#1d9e75' : '#a35a1d' },
  ];
  return (
    <div style={{ padding: '8px 4px', color: '#e8ddd0' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 24, marginBottom: 20, flexWrap: 'wrap' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: 10 }}>
          <span style={{ fontSize: 46, fontWeight: 700, color: heat(data.score), lineHeight: 1 }}>
            {data.score?.toFixed(1) ?? '—'}</span>
          <span style={{ fontSize: 16, color: '#9d8b7a' }}>/100</span>
        </div>
        <div style={{ fontSize: 12, color: '#9d8b7a', lineHeight: 1.7 }}>
          <div>FINANCIAL INTELLIGENCE · 18% of conviction</div>
          <div>Coverage {data.coverage.scored}/{data.coverage.total} signals live · confidence {(data.confidence * 100).toFixed(0)}%</div>
          <div>Market cap {bn(data.market_cap)} · WACC {(data.wacc_used * 100).toFixed(1)}% · {data.n_quarters} quarters</div>
        </div>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 8 }}>
          {badges.map(b => (
            <div key={b.k} title={EXPLAIN[b.k] || ''} style={{ background: b.c, color: '#fff', padding: '6px 12px', borderRadius: 8, fontSize: 12, fontWeight: 600, textAlign: 'center', cursor: 'help' }}>
              <div style={{ opacity: 0.85, fontSize: 10 }}>{b.k} ⓘ</div>{b.v}</div>
          ))}
        </div>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(130px,1fr))', gap: 10, marginBottom: 22 }}>
        {cards.map(c => (
          <div key={c.k} title={EXPLAIN[c.k] || ''} style={{ background: '#181818', border: '1px solid #2a2a2a', borderRadius: 10, padding: '12px 14px', cursor: EXPLAIN[c.k] ? 'help' : 'default' }}>
            <div style={{ fontSize: 11, color: '#9d8b7a', marginBottom: 4 }}>{c.k}{EXPLAIN[c.k] ? ' ⓘ' : ''}</div>
            <div style={{ fontSize: 20, fontWeight: 600, color: '#daa520' }}>{c.v}</div>
          </div>
        ))}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <span style={{ fontSize: 12, color: '#9d8b7a', letterSpacing: 1 }}>12 CATEGORIES · {data.coverage.scored} live signals</span>
        <span onClick={toggleAll} style={{ fontSize: 11, color: '#daa520', cursor: 'pointer', border: '1px solid #3a3020', borderRadius: 6, padding: '4px 10px' }}>
          {allOpen ? '▲ Collapse all' : '▼ Expand all'}</span>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(300px,1fr))', gap: 10, alignItems: 'start' }}>
        {data.tree.categories.map(cat => (
          <div key={cat.id} style={{ borderRadius: 10, overflow: 'hidden', border: '1px solid #2a2a2a' }}>
            <div onClick={() => toggle(cat.id)}
              style={{ background: heat(cat.score), padding: '12px 14px', cursor: 'pointer' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span style={{ fontSize: 12, fontWeight: 600, color: heatText(cat.score) }}>{cat.label.replace(' Intelligence', '')}</span>
                <span style={{ fontSize: 18, fontWeight: 700, color: heatText(cat.score) }}>{cat.score?.toFixed(0) ?? '—'}</span>
              </div>
              <div style={{ fontSize: 10, color: heatText(cat.score), opacity: 0.75, marginTop: 3 }}>
                wt {cat.weight.toFixed(2)} · {cat.n_scored}/{cat.n_signals} live · conf {(cat.confidence * 100).toFixed(0)}%</div>
            </div>
            {isOpen(cat.id) && (
              <div style={{ background: '#141414', padding: 10 }}>
                {cat.signals.map(s => {
                  const isRef = s.method === 'reference';
                  const isPending = s.method === 'needs_source';
                  return (
                  <div key={s.id} style={{ marginBottom: 9, opacity: isPending ? 0.5 : 1 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12 }}>
                      <span style={{ color: '#cdbfae' }}>{s.label}
                        {isRef && <span style={{ fontSize: 9, color: '#7a86c0', marginLeft: 6, border: '1px solid #3a4060', borderRadius: 4, padding: '1px 4px' }}>REF</span>}
                        {isPending && <span style={{ fontSize: 9, color: '#8a7519', marginLeft: 6, border: '1px solid #4a4020', borderRadius: 4, padding: '1px 4px' }}>SOON</span>}
                      </span>
                      <span style={{ color: isRef ? '#9db0d8' : heat(s.score), fontWeight: 600 }}>
                        {isPending ? 'pending' : fmt(s.id, s.raw_value)}{!isRef && !isPending ? ' · ' + (s.score?.toFixed(0) ?? '—') : ''}</span>
                    </div>
                    {!isRef && !isPending && (
                    <div style={{ height: 5, background: '#242424', borderRadius: 3, marginTop: 3, overflow: 'hidden' }}>
                      <div style={{ height: '100%', width: `${s.score ?? 0}%`, background: heat(s.score) }} />
                    </div>)}
                    <div style={{ fontSize: 10, color: '#6f665b', marginTop: 2 }}>{s.evidence}</div>
                  </div>
                  );
                })}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
