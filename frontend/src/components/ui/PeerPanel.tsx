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
  const [rel, setRel] = useState<any>(null);

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
        const rr = await api.get(`/api/v6/relationships/${ticker}`);
        if (rr.data?.data?.available) setRel(rr.data.data);
      } catch { /* filing coverage is uneven */ }
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
  // The searched ticker is the reference point, not a competitor to itself.
  // Its standing against this group is already stated in the header percentiles.
  const sortedPeers = pd.peers.filter((p: any) => !p.is_me)
    .sort((a,b) => ((valueOf(b) ?? -1e9) - (valueOf(a) ?? -1e9)));

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

      {/* What actually separates this company from its group */}
      {(() => {
        const ff = pd.fund_factors || [];
        const get = (k: string) => ff.find((x: any) => x.key === k);
        const fmtPct = (v: any) => v == null ? null : `${(Number(v) * 100).toFixed(1)}%`;
        const strengths = ff.filter((x: any) => x.percentile >= 70)
                            .sort((a: any, b: any) => b.percentile - a.percentile);
        const weaknesses = ff.filter((x: any) => x.percentile <= 30)
                             .sort((a: any, b: any) => a.percentile - b.percentile);
        const pe = get('fund_pe');
        const mom = (pd.factors || []).find((x: any) => x.key === 'mom_3m');
        if (!ff.length) return null;
        const phrase = (x: any) => `${x.label.replace(' (cheap)','')} ${x.value} against a peer median of ${x.peer_median}`;
        return (
          <div style={{ background:'#241510', border:`1px solid ${C.border}`, borderRadius:8,
            padding:'14px 18px', marginBottom:22, lineHeight:1.65, fontSize:12.5, color:C.text }}>
            {strengths.length > 0 && (
              <>Stands out on <b style={{color:C.green}}>{strengths.slice(0,3).map((x:any)=>x.label.replace(' (cheap)','').toLowerCase()).join(', ')}</b>
                {' '}— {phrase(strengths[0])}. </>
            )}
            {weaknesses.length > 0 && (
              <>Weakest on <b style={{color:C.red}}>{weaknesses.slice(0,2).map((x:any)=>x.label.replace(' (cheap)','').toLowerCase()).join(' and ')}</b>
                {' '}— {phrase(weaknesses[0])}. </>
            )}
            {pe && pe.value != null && pe.peer_median != null && Number(pe.peer_median) > 0 && (
              <>The market prices it at <b style={{color: Number(pe.value) > Number(pe.peer_median) * 1.3 ? C.red : C.textDim}}>
                {(Number(pe.value) / Number(pe.peer_median)).toFixed(1)}×</b> the peer P/E
                {Number(pe.value) > Number(pe.peer_median) * 1.3
                  ? ' — the premium is the trade-off for the strengths above.'
                  : Number(pe.value) < Number(pe.peer_median) * 0.7
                    ? ' — cheaper than the group, which is worth understanding before assuming it is a bargain.'
                    : ', broadly in line with the group.'} </>
            )}
            {mom && mom.percentile != null && (
              <>Recent three-month momentum sits at the <b>{mom.percentile}th percentile</b> of the group.</>
            )}
          </div>
        );
      })()}

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

      {rel && (() => {
        const groups = [
          { key:'supplies_this',    label:'SUPPLIES THIS COMPANY',
            note:'Companies whose own filings name this one as a customer — they benefit when it grows.' },
          { key:'buys_from',        label:'NAMED AS ITS CUSTOMERS',
            note:'Companies this one describes buying from or selling to in its filing.' },
          { key:'competitors',      label:'NAMED AS COMPETITORS',
            note:'Rivals this company identifies in its own filing.' },
          { key:'named_as_rival_by',label:'NAME IT AS A RIVAL',
            note:'Companies that list this one as competition.' },
        ].filter(g => (rel[g.key] || []).length > 0);
        if (!groups.length) return null;
        return (
          <div style={{ marginBottom:26 }}>
            <div style={{ color:C.gold, fontWeight:700, fontSize:13, marginBottom:4 }}>DISCLOSED RELATIONSHIPS</div>
            <div style={{ fontSize:11, color:C.textDim, marginBottom:14, lineHeight:1.55 }}>
              Read from 10-K filings, with the disclosing sentence kept as evidence. Suppliers must report customer
              concentration, so they name their buyers; the buyers rarely name anyone. Absence here means nothing was
              disclosed, not that no relationship exists.
            </div>
            {groups.map(g => (
              <div key={g.key} style={{ marginBottom:14 }}>
                <div style={{ fontSize:10, color:C.gold, letterSpacing:1.5, marginBottom:2 }}>{g.label}</div>
                <div style={{ fontSize:10, color:C.textDim, marginBottom:7 }}>{g.note}</div>
                <div style={{ display:'flex', gap:7, flexWrap:'wrap' }}>
                  {(rel[g.key] || []).slice(0,14).map((x:any,i:number) => (
                    <div key={i} title={x.evidence || ''}
                      onClick={() => x.ticker && onAnalyze && onAnalyze(x.ticker)}
                      style={{ background:'#1a0f0a', border:`1px solid ${C.border}`, borderRadius:5,
                        padding:'6px 10px', cursor: x.ticker && onAnalyze ? 'pointer':'default', maxWidth:230 }}>
                      <div style={{ fontFamily:"'Fira Code',monospace", fontSize:11,
                        color: x.ticker ? C.gold : C.textDim }}>
                        {x.ticker || '—'}
                      </div>
                      <div style={{ fontSize:10, color:C.text, whiteSpace:'nowrap', overflow:'hidden', textOverflow:'ellipsis' }}>
                        {x.name}
                      </div>
                      {x.filing_date && (
                        <div style={{ fontSize:8.5, color:C.textDim }}>{x.filing_date}</div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        );
      })()}

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

    </div>
  );
};

export default PeerPanel;
