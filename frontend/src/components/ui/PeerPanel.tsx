// ============================================================
// QuantEdge v6.0 — Peer Comparison Panel (interactive)
// Where a ticker ranks among its sector-bucket peers, by factor.
// ============================================================

import React, { useEffect, useState, useCallback } from 'react';
import { api } from '../../auth/authStore';

const C = {
  border: '#3a2920', gold: '#daa520', text: '#d4c4b0', textDim: '#8a7560',
  green: '#22c55e', red: '#ef4444', panel: '#1a0f0a', me: '#daa520',
};

interface FactorPct { key: string; label: string; value: number; percentile: number; peer_median: number; }
interface PeerRow { ticker: string; name: string; market_cap: number | null; mom_3m: number | null; mom_6m: number | null; pct_above_ma200: number | null; roic: number | null; net_margin: number | null; gross_margin: number | null; revenue_growth: number | null; pe: number | null; is_me: boolean; }
interface FundFactor { key: string; label: string; value: number; percentile: number; peer_median: number; rank: number; of: number; }
interface PeerData {
  available: boolean; reason?: string; ticker: string; name: string;
  bucket: string; peer_count: number; factors: FactorPct[]; fund_factors?: FundFactor[]; peers: PeerRow[];
}

const fmtCap = (c: number | null) => c ? (c >= 1e12 ? `$${(c/1e12).toFixed(1)}T` : c >= 1e9 ? `$${(c/1e9).toFixed(0)}B` : `$${(c/1e6).toFixed(0)}M`) : '—';
const pctColor = (p: number) => p >= 66 ? C.green : p >= 33 ? C.gold : C.red;

interface Props { data?: any; ticker?: string; onAnalyze?: (t: string) => void; }

const PeerPanel: React.FC<Props> = ({ ticker: tickerProp, data: analysisData, onAnalyze }) => {
  const ticker = (tickerProp || analysisData?.ticker || analysisData?.symbol || '').toUpperCase();
  const [pd, setPd] = useState<PeerData | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(false);
  const [factorKey, setFactorKey] = useState('mom_3m');
  const [scoreData, setScoreData] = useState<any>(null);
  const [eco, setEco] = useState<any>(null);

  const load = useCallback(async () => {
    if (!ticker) { setLoading(false); return; }
    setLoading(true); setErr(false);
    try {
      const res = await api.get(`/api/v6/peers/${ticker}`);
      setPd(res.data?.data || null);
      try {
        const sr = await api.get(`/api/v6/peers_score/${ticker}`);
        if (sr.data?.data?.available) setScoreData(sr.data.data);
      } catch { /* score is optional enhancement */ }
      try {
        const er = await api.get(`/api/v6/ecosystem/${ticker}`);
        if (er.data?.data?.available) setEco(er.data.data);
      } catch { /* correlation view is optional */ }
    } catch { setErr(true); }
    finally { setLoading(false); }
  }, [ticker]);

  useEffect(() => { load(); }, [load]);

  if (!ticker) return <div style={{ color: C.textDim, padding: 20 }}>Analyze a stock to see peer comparison.</div>;
  if (loading) return <div style={{ color: C.textDim, padding: 20 }}>Loading peer comparison…</div>;
  if (err || !pd) return <div style={{ color: C.textDim, padding: 20 }}>Couldn’t load peer data for {ticker}.</div>;
  if (!pd.available) return <div style={{ color: C.textDim, padding: 20 }}>{ticker} isn’t in the scanned universe yet ({pd.reason || 'no data'}).</div>;

  const valueOf = (p: PeerRow): number | null => {
    if (factorKey === 'mom_3m') return p.mom_3m;
    if (factorKey === 'mom_6m') return p.mom_6m;
    if (factorKey === 'pct_above_ma200') return p.pct_above_ma200;
    return p.mom_3m;
  };
  const factorLabels: Record<string,string> = { mom_3m:'3M Momentum', mom_6m:'6M Momentum', pct_above_ma200:'Above 200D MA' };

  // distribution: map peers' values to positions
  const vals = pd.peers.map(valueOf).filter(v => v != null) as number[];
  const minV = Math.min(...vals), maxV = Math.max(...vals), range = (maxV - minV) || 1;
  const sortedPeers = [...pd.peers].sort((a,b) => ((valueOf(b) ?? -1e9) - (valueOf(a) ?? -1e9)));

  return (
    <div style={{ color: C.text }}>
      {/* Header */}
      <div style={{ display:'flex', alignItems:'baseline', gap:12, flexWrap:'wrap', marginBottom:6 }}>
        <span style={{ color: C.gold, fontWeight:700, fontSize:15 }}>{ticker} vs {pd.bucket.toUpperCase()} PEERS</span>
        <span style={{ color: C.textDim, fontSize:12 }}>{pd.peer_count} comparable companies</span>
      </div>
      {scoreData && (() => {
        const s = scoreData.score; const rating = scoreData.peers_rating || '';
        const km = scoreData.key_metrics || {};
        const rc = s==null?'#8a7560':s>=72?'#22c55e':s>=58?'#4ade80':s>=44?'#daa520':s>=30?'#f59e0b':'#ef4444';
        const pctl = (v:number|null|undefined)=> v==null?'—':Math.round(v*100)+'th';
        return (
          <div style={{ display:'flex', alignItems:'center', gap:20, flexWrap:'wrap', background:'#140c08',
                        border:`1px solid ${C.border}`, borderRadius:12, padding:'12px 18px', marginBottom:16 }}>
            <div style={{ display:'flex', alignItems:'baseline', gap:6 }}>
              <span style={{ fontSize:34, fontWeight:700, color:rc, lineHeight:1 }}>{s?.toFixed(0) ?? '—'}</span>
              <span style={{ fontSize:13, color:C.textDim }}>/100</span>
            </div>
            <div style={{ minWidth:110 }}>
              <div style={{ fontSize:16, fontWeight:700, color:rc }}>{rating}</div>
              <div style={{ fontSize:10, color:C.textDim }}>overall peer standing</div>
            </div>
            <div style={{ display:'flex', gap:16, flexWrap:'wrap', flex:1, justifyContent:'flex-end' }}>
              {[['Quality',km.quality_composite],['Profitability',km.profitability_composite],
                ['Growth',km.revenue_growth_rank],['ROIC',km.roic_rank],['Valuation',km.pe_rank]].map(([l,v]:any)=>(
                <div key={l} style={{ textAlign:'center' }}>
                  <div style={{ fontSize:9, color:C.textDim, textTransform:'uppercase', letterSpacing:0.5 }}>{l}</div>
                  <div style={{ fontSize:15, fontWeight:600, color: v==null?'#8a7560':v>=0.66?C.green:v>=0.33?C.gold:C.red }}>{pctl(v)}</div>
                </div>
              ))}
            </div>
          </div>
        );
      })()}
      <div style={{ fontSize:11, color:C.textDim, marginBottom:18, fontStyle:'italic' }}>
        Percentile = share of sector peers this stock ranks above on each factor. Higher is stronger. Contextual, not predictive.
      </div>

      {/* Percentile bars (all factors) */}
      <div style={{ display:'flex', flexDirection:'column', gap:10, marginBottom:24 }}>
        {pd.factors.map(f => (
          <div key={f.key} style={{ display:'flex', alignItems:'center', gap:12 }}>
            <div style={{ width:120, fontSize:12, color:C.text, textAlign:'right' }}>{f.label}</div>
            <div style={{ flex:1, position:'relative', height:22, background:'#120a07', borderRadius:5, overflow:'hidden' }}>
              <div style={{ position:'absolute', left:0, top:0, bottom:0, width:`${f.percentile}%`, background:`${pctColor(f.percentile)}33`, borderRight:`2px solid ${pctColor(f.percentile)}` }} />
              <div style={{ position:'absolute', left:8, top:0, bottom:0, display:'flex', alignItems:'center', fontSize:11, color:C.textDim }}>
                val {f.value} · median {f.peer_median}
              </div>
            </div>
            <div style={{ width:54, textAlign:'right', fontFamily:"'Fira Code',monospace", fontWeight:700, fontSize:14, color:pctColor(f.percentile) }}>
              {f.percentile}%
            </div>
          </div>
        ))}
      </div>

      {/* Fundamental percentile bars (quality / profitability / growth / valuation vs peers) */}
      {pd.fund_factors && pd.fund_factors.length > 0 && (
        <>
          <div style={{ color:C.gold, fontWeight:700, fontSize:13, marginBottom:4, marginTop:4 }}>FUNDAMENTALS vs PEERS</div>
          <div style={{ fontSize:11, color:C.textDim, marginBottom:12, fontStyle:'italic' }}>
            How {ticker} ranks on quality, profitability, growth &amp; valuation across the sector.
          </div>
          <div style={{ display:'flex', flexDirection:'column', gap:10, marginBottom:24 }}>
            {pd.fund_factors.map(f => {
              const disp = f.label.includes('Margin')||f.label.includes('ROIC')||f.label.includes('ROE')||f.label.includes('Growth')
                ? `${(f.value*100).toFixed(1)}%` : f.value.toFixed(1);
              const medDisp = f.label.includes('Margin')||f.label.includes('ROIC')||f.label.includes('ROE')||f.label.includes('Growth')
                ? `${(f.peer_median*100).toFixed(1)}%` : f.peer_median.toFixed(1);
              return (
                <div key={f.key} style={{ display:'flex', alignItems:'center', gap:12 }}>
                  <div style={{ width:120, fontSize:12, color:C.text, textAlign:'right' }}>{f.label}</div>
                  <div style={{ flex:1, position:'relative', height:22, background:'#120a07', borderRadius:5, overflow:'hidden' }}>
                    <div style={{ position:'absolute', left:0, top:0, bottom:0, width:`${f.percentile}%`, background:`${pctColor(f.percentile)}33`, borderRight:`2px solid ${pctColor(f.percentile)}` }} />
                    <div style={{ position:'absolute', left:8, top:0, bottom:0, display:'flex', alignItems:'center', fontSize:11, color:C.textDim }}>
                      {disp} · median {medDisp}
                    </div>
                  </div>
                  <div style={{ width:64, textAlign:'right', fontSize:10, color:C.textDim, fontFamily:"'Fira Code',monospace" }}>#{f.rank}/{f.of}</div>
                  <div style={{ width:54, textAlign:'right', fontFamily:"'Fira Code',monospace", fontWeight:700, fontSize:14, color:pctColor(f.percentile) }}>
                    {f.percentile}%
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}

      {/* Interactive distribution strip */}
      <div style={{ marginBottom:8, display:'flex', gap:6, flexWrap:'wrap', alignItems:'center' }}>
        <span style={{ fontSize:11, color:C.textDim, marginRight:4 }}>DISTRIBUTION:</span>
        {['mom_3m','mom_6m','pct_above_ma200'].map(k => (
          <button key={k} onClick={() => setFactorKey(k)}
            style={{ fontSize:10, padding:'4px 9px', borderRadius:4, cursor:'pointer',
              border:`1px solid ${factorKey===k?C.gold:C.border}`, background: factorKey===k?`${C.gold}18`:'transparent',
              color: factorKey===k?C.gold:C.textDim, fontFamily:"'Fira Code',monospace" }}>
            {factorLabels[k]}
          </button>
        ))}
      </div>
      <div style={{ position:'relative', height:54, background:'#120a07', borderRadius:6, marginBottom:24, padding:'0 2px' }}>
        {pd.peers.map((p,i) => {
          const v = valueOf(p); if (v == null) return null;
          const x = ((v - minV) / range) * 96 + 2;
          const isMe = p.is_me;
          return <div key={i} title={`${p.ticker}: ${v}`} style={{
            position:'absolute', left:`${x}%`, top: isMe?6:'50%', transform: isMe?'translateX(-50%)':'translate(-50%,-50%)',
            width: isMe?12:6, height: isMe?12:6, borderRadius:'50%',
            background: isMe?C.me:`${C.textDim}99`, border: isMe?`2px solid ${C.gold}`:'none',
            zIndex: isMe?5:1, boxShadow: isMe?`0 0 8px ${C.gold}`:'none' }} />;
        })}
        <div style={{ position:'absolute', bottom:4, left:6, fontSize:9, color:C.textDim }}>weaker</div>
        <div style={{ position:'absolute', bottom:4, right:6, fontSize:9, color:C.textDim }}>stronger →</div>
        <div style={{ position:'absolute', top:4, left:'50%', transform:'translateX(-50%)', fontSize:9, color:C.gold }}>● {ticker}</div>
      </div>

      {/* Peer table */}
      <div style={{ color:C.gold, fontWeight:700, fontSize:13, marginBottom:8 }}>PEERS — RANKED BY {factorLabels[factorKey].toUpperCase()}</div>
      <div style={{ maxHeight:320, overflowY:'auto' }}>
        <table style={{ width:'100%', borderCollapse:'collapse', fontSize:12 }}>
          <thead>
            <tr style={{ color:C.textDim, fontSize:10, textAlign:'left' }}>
              <th style={{ padding:'6px 8px' }}>TICKER</th><th style={{ padding:'6px 8px' }}>NAME</th>
              <th style={{ padding:'6px 8px', textAlign:'right' }}>MKT CAP</th>
              <th style={{ padding:'6px 8px', textAlign:'right' }}>{factorLabels[factorKey]}</th>
              <th style={{ padding:'6px 8px', textAlign:'right' }}>ROIC</th>
              <th style={{ padding:'6px 8px', textAlign:'right' }}>NET MGN</th>
              <th style={{ padding:'6px 8px', textAlign:'right' }}>REV GR</th>
              <th style={{ padding:'6px 8px', textAlign:'right' }}>P/E</th>
            </tr>
          </thead>
          <tbody>
            {sortedPeers.slice(0,40).map((p,i) => {
              const v = valueOf(p);
              return (
                <tr key={i} onClick={() => onAnalyze && onAnalyze(p.ticker)}
                  style={{ cursor: onAnalyze?'pointer':'default', background: p.is_me?`${C.gold}14`:'transparent',
                    borderBottom:`1px solid ${C.border}` }}>
                  <td style={{ padding:'7px 8px', fontWeight:700, color: p.is_me?C.gold:C.text }}>{p.ticker}{p.is_me?' ★':''}</td>
                  <td style={{ padding:'7px 8px', color:C.textDim }}>{p.name}</td>
                  <td style={{ padding:'7px 8px', textAlign:'right', color:C.textDim }}>{fmtCap(p.market_cap)}</td>
                  <td style={{ padding:'7px 8px', textAlign:'right', fontFamily:"'Fira Code',monospace",
                    color: (v??0)>=0?C.green:C.red }}>{v!=null?`${v>0?'+':''}${Number(v).toFixed(1)}%`:'—'}</td>
                  <td style={{ padding:'7px 8px', textAlign:'right', fontFamily:"'Fira Code',monospace", color: p.roic==null?C.textDim:(p.roic>=0.1?C.green:p.roic>=0?C.text:C.red) }}>{p.roic==null?'—':(p.roic*100).toFixed(0)+'%'}</td>
                  <td style={{ padding:'7px 8px', textAlign:'right', fontFamily:"'Fira Code',monospace", color: p.net_margin==null?C.textDim:(p.net_margin>=0.1?C.green:p.net_margin>=0?C.text:C.red) }}>{p.net_margin==null?'—':(p.net_margin*100).toFixed(1)+'%'}</td>
                  <td style={{ padding:'7px 8px', textAlign:'right', fontFamily:"'Fira Code',monospace", color: p.revenue_growth==null?C.textDim:(p.revenue_growth>=0.1?C.green:p.revenue_growth>=0?C.text:C.red) }}>{p.revenue_growth==null?'—':(p.revenue_growth>=0?'+':'')+(p.revenue_growth*100).toFixed(0)+'%'}</td>
                  <td style={{ padding:'7px 8px', textAlign:'right', fontFamily:"'Fira Code',monospace", color: p.pe==null?C.textDim:(p.pe>0&&p.pe<20?C.green:p.pe>40?C.red:C.text) }}>{p.pe==null?'—':p.pe.toFixed(0)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {eco && eco.movers?.length > 0 && (() => {
        const floor = eco.significance_floor ?? 0.25;
        const strength = (c:number) =>
          c >= 0.6 ? { t:'moves closely', col:C.green }
          : c >= 0.4 ? { t:'moderate', col:C.gold }
          : c >= floor ? { t:'loose', col:C.textDim }
          : { t:'weak — mostly market', col:C.textDim };
        const top = eco.movers[0]?.correlation ?? 0;
        const allWeak = top < floor;
        return (
          <div style={{ marginTop:22 }}>
            <div style={{ color:C.gold, fontWeight:700, fontSize:13, marginBottom:4 }}>MOVES WITH THIS STOCK</div>
            <div style={{ color:C.textDim, fontSize:11, lineHeight:1.55, marginBottom:12 }}>
              Measured co-movement of daily returns over the past year across {eco.n_compared} companies.
              This is an observed association, not an asserted business relationship — two stocks may move
              together because one supplies the other, because they share customers, or simply because both
              track the market.
            </div>

            {allWeak && (
              <div style={{ padding:'10px 12px', background:'rgba(212,149,108,0.05)', borderLeft:`2px solid ${C.gold}`,
                borderRadius:4, marginBottom:12, fontSize:11.5, color:C.text, lineHeight:1.55 }}>
                Nothing in the universe moves closely with {ticker} right now — the strongest association is only{' '}
                {top.toFixed(2)}. Over this window it has been trading on its own news rather than with any group,
                which usually means company-specific events are dominating.
              </div>
            )}

            <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(230px, 1fr))', gap:8 }}>
              {eco.movers.slice(0,12).map((m:any) => {
                const s = strength(m.correlation);
                return (
                  <div key={m.ticker} style={{ background:'#1a0f0a', borderRadius:6, padding:'9px 11px',
                    borderLeft:`2px solid ${s.col}55` }}>
                    <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline' }}>
                      <span style={{ fontFamily:"'Fira Code',monospace", fontSize:12, fontWeight:700, color:C.text }}>{m.ticker}</span>
                      <span style={{ fontFamily:"'Fira Code',monospace", fontSize:13, fontWeight:700, color:s.col }}>
                        {m.correlation.toFixed(2)}
                      </span>
                    </div>
                    <div style={{ fontSize:10, color:C.textDim, marginTop:2, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>{m.name}</div>
                    <div style={{ display:'flex', justifyContent:'space-between', marginTop:4 }}>
                      <span style={{ fontSize:9, color:s.col }}>{s.t}</span>
                      <span style={{ fontSize:9, color:C.textDim }}>
                        {m.same_sector ? 'same industry' : m.sector}
                        {m.beta_to_this != null ? ` · β ${m.beta_to_this}` : ''}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {eco.inverse?.length > 0 && eco.inverse[0].correlation < -0.15 && (
              <div style={{ marginTop:14 }}>
                <div style={{ color:C.gold, fontSize:11, letterSpacing:1, marginBottom:6 }}>MOVES AGAINST IT</div>
                <div style={{ display:'flex', gap:8, flexWrap:'wrap' }}>
                  {eco.inverse.filter((m:any)=>m.correlation < -0.15).slice(0,6).map((m:any)=>(
                    <div key={m.ticker} style={{ background:'#1a0f0a', borderRadius:5, padding:'6px 10px' }}>
                      <span style={{ fontFamily:"'Fira Code',monospace", fontSize:11, color:C.text }}>{m.ticker}</span>
                      <span style={{ fontFamily:"'Fira Code',monospace", fontSize:11, color:C.red, marginLeft:8 }}>{m.correlation.toFixed(2)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        );
      })()}
    </div>
  );
};

export default PeerPanel;
