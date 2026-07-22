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
  const [relPerf, setRelPerf] = useState<any>(null);
  const [picked, setPicked] = useState<string[]>([]);
  const [win, setWin] = useState<number>(252);
  const [hoverI, setHoverI] = useState<number|null>(null);

  const load = useCallback(async () => {
    if (!ticker) { setLoading(false); return; }
    setLoading(true); setErr(false);
    try {
      api.get(`/api/v6/peers/${ticker}/relative`)
        .then(r => {
          const rp = r.data?.data?.available ? r.data.data : null;
          setRelPerf(rp);
          setPicked((rp?.roster || []).slice(0, 5).map((x:any) => x.ticker));
        })
        .catch(() => setRelPerf(null));
      const res = await api.get(`/api/v6/peers/${ticker}`);
      setPd(res.data?.data || null);
      try {
        const sr = await api.get(`/api/v6/peers_score/${ticker}`);
        if (sr.data?.data?.available) setScoreData(sr.data.data);
      } catch { /* score is optional enhancement */ }
      try {
        const rr = await api.get(`/api/v6/relationships/${ticker}`);
        if (rr.data?.data) setRel(rr.data.data);
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
  // SIC 7370 holds Alphabet and Snap alongside a gambling operator and a cloud
  // host, so neither momentum nor market cap identifies a rival. Business shape
  // does: a company with comparable margins, returns and growth competes for the
  // same customers and capital, whatever its size.
  const me = pd.peers.find((p: any) => p.is_me);
  const rivalScore = (p: any) => {
    if (!me) return 0;
    const dims: Array<[any, any, number]> = [
      [p.net_margin,     me.net_margin,     1.0],
      [p.gross_margin,   me.gross_margin,   0.8],
      [p.roic,           me.roic,           0.8],
      [p.revenue_growth, me.revenue_growth, 0.6],
    ];
    let d = 0, w = 0;
    for (const [a, b, weight] of dims) {
      if (a == null || b == null) continue;
      d += weight * Math.abs(Number(a) - Number(b));
      w += weight;
    }
    // Size still matters a little — a $3B and a $1.6T business rarely compete
    // head-on — but it is a tiebreaker, not the ordering.
    let sizeGap = 0;
    if (p.market_cap && me.market_cap) {
      sizeGap = Math.abs(Math.log10((p.market_cap || 1) / me.market_cap)) * 0.08;
    }
    return w > 0 ? (d / w) + sizeGap : 99;
  };

  // Dual-class listings are one company: GOOGL and GOOG carry identical
  // fundamentals and would otherwise occupy two of the ten rival slots.
  const dedupeShareClasses = (list: any[]) => {
    const seen = new Set<string>();
    return list.filter((p: any) => {
      const base = (p.name || p.ticker || '')
        .toLowerCase()
        .replace(/\s+(class\s+[a-c]|series\s+[a-c]).*$/, '')
        .replace(/\s+(common stock|capital stock|ordinary shares|subordinate voting).*$/, '')
        .replace(/[^a-z0-9]/g, '')
        .slice(0, 18);
      if (!base || seen.has(base)) return false;
      seen.add(base);
      return true;
    });
  };

  const sortedPeers = dedupeShareClasses(
    pd.peers.filter((p: any) => !p.is_me).sort((a: any, b: any) => rivalScore(a) - rivalScore(b))
  );

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
        // Margins and returns are stored as fractions; multiples are not. Printing
        // 0.29 where the bar below says 29.0% makes the sentence read as a different
        // number than the chart.
        const isRatio = (k: string) => /_(pe|ps|pb|ev)$/.test(k) || /P\/(E|S|B)/.test(k);
        const fmt = (k: string, v: any) => {
          if (v == null) return '—';
          const n = Number(v);
          return isRatio(k) ? `${n.toFixed(1)}x` : `${(n * 100).toFixed(1)}%`;
        };
        const phrase = (x: any) =>
          `${x.label.replace(' (cheap)','')} ${fmt(x.key, x.value)} against a peer median of ${fmt(x.key, x.peer_median)}`;
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
            {pe && pe.value != null && pe.peer_median != null && Number(pe.peer_median) > 0 && (() => {
              const ps = get('fund_ps');
              const peR = Number(pe.value) / Number(pe.peer_median);
              const psR = ps && ps.value != null && Number(ps.peer_median) > 0
                ? Number(ps.value) / Number(ps.peer_median) : null;
              // Earnings and sales multiples can disagree — a company can look
              // cheap on profits and expensive on revenue. Reporting one alone
              // misleads, so both are stated when they diverge.
              const split = psR != null && ((peR < 0.9 && psR > 1.3) || (peR > 1.3 && psR < 0.9));
              return (
                <>The market prices it at <b style={{color: peR > 1.25 ? C.red : peR < 0.85 ? C.green : C.textDim}}>
                  {peR.toFixed(1)}×</b> the peer P/E
                  {split && psR != null && (
                    <> but <b style={{color: psR > 1 ? C.red : C.green}}>{psR.toFixed(1)}×</b> the peer P/S
                      {peR < 1 ? ' — cheap on earnings, expensive on revenue, which usually means unusually high margins'
                               : ' — expensive on earnings, cheap on revenue, which usually means compressed margins'}</>
                  )}
                {Number(pe.value) > Number(pe.peer_median) * 1.25
                  ? ' — the premium is the trade-off for the strengths above.'
                  : Number(pe.value) < Number(pe.peer_median) * 0.85
                    ? ' — a discount to the group, worth understanding before assuming it is a bargain.'
                    : ', broadly in line with the group.'} </>
              );
            })()}
            {mom && mom.percentile != null && (
              <>Recent three-month momentum sits at the <b>{mom.percentile}th percentile</b> of the group.</>
            )}
          </div>
        );
      })()}

      {relPerf && relPerf.dates?.length > 20 && (() => {
        const LINE = ['#6ea8d8','#7aa874','#c98f6c','#a98bc4','#c9b06c'];
        const W = 1000, H = 170, PAD = { l: 46, r: 108, t: 10, b: 20 };
        const all = relPerf.dates as string[];
        const n = Math.min(win, all.length - 1);
        const i0 = all.length - 1 - n;
        const dates = all.slice(i0);
        // Rebase to the window start so every line begins at zero.
        const cut = (t:string) => {
          const raw = relPerf.by_ticker[t];
          if (!raw) return null;
          const seg = raw.slice(i0);
          const b = seg.find((v:any) => v != null);
          if (b == null) return null;
          return seg.map((v:any) => v == null ? null : ((1 + v/100) / (1 + b/100) - 1) * 100);
        };
        const lines = [{ t: relPerf.ticker, v: cut(relPerf.ticker), c: C.gold, me: true },
          ...picked.map((t:string, i:number) => ({ t, v: cut(t), c: LINE[i % LINE.length], me: false }))]
          .filter(l => l.v);
        const flat = lines.flatMap(l => (l.v as any[]).filter(v => v != null));
        if (!flat.length) return null;
        const lo = Math.min(...flat, 0), hi = Math.max(...flat, 0);
        const pad = (hi - lo) * 0.1 || 1;
        const sx = (i:number) => PAD.l + (i/(dates.length-1)) * (W-PAD.l-PAD.r);
        const sy = (v:number) => H-PAD.b - ((v-(lo-pad))/((hi+pad)-(lo-pad))) * (H-PAD.t-PAD.b);
        const path = (vs:any[]) => vs.map((v,i)=> v==null?null:`${i===0||vs[i-1]==null?'M':'L'}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).filter(Boolean).join(' ');
        const toggle = (t:string) => setPicked(p => p.includes(t) ? (p.length>1 ? p.filter(x=>x!==t) : p) : (p.length<5 ? [...p,t] : p));
        const WINS = [['1M',21],['3M',63],['6M',126],['1Y',252],['2Y',504]] as [string,number][];

        return (
          <>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'baseline', flexWrap:'wrap', gap:8, marginTop:4 }}>
              <div style={{ color:C.gold, fontWeight:700, fontSize:13 }}>PERFORMANCE VS NAMED RIVALS</div>
              <div style={{ display:'flex', gap:4 }}>
                {WINS.filter(([,d])=>d<=all.length).map(([l,d])=>(
                  <button key={l} onClick={()=>setWin(d)} style={{ fontFamily:'monospace', fontSize:10, padding:'3px 8px', cursor:'pointer',
                    background: win===d?'rgba(218,165,32,0.12)':'transparent', color: win===d?C.gold:C.textDim,
                    border:`1px solid ${win===d?C.gold:C.border}`, borderRadius:3 }}>{l}</button>
                ))}
              </div>
            </div>
            <div style={{ fontSize:11, color:C.textDim, marginBottom:8, fontStyle:'italic' }}>
              Cumulative return rebased to zero at the window start. Pick 1-5 rivals below.
              {relPerf.group_kind === 'industry'
                ? ' Group matched on exact 4-digit SIC — precise classification, though shared coding does not guarantee a shared business.'
                : ` Group matched at ${relPerf.group_kind} level — a broad comparison, read it loosely.`}
            </div>
            <svg viewBox={`0 0 ${W} ${H}`} style={{ width:'100%', display:'block' }}>
              <line x1={PAD.l} y1={sy(0)} x2={W-PAD.r} y2={sy(0)} stroke={C.border} strokeDasharray="3 4" />
              {lines.map(l => <path key={l.t} d={path(l.v as any[])} fill="none" stroke={l.c} strokeWidth={l.me?2:1.2} opacity={l.me?1:0.85} />)}
              {[lo-pad, (lo+hi)/2, hi+pad].map((v,i)=>(
                <text key={i} x={PAD.l-6} y={sy(v)+3} fill={C.textDim} fontSize="9.5" fontFamily="monospace" textAnchor="end">
                  {v>=0?'+':''}{v.toFixed(0)}%
                </text>
              ))}
              {(() => {
                // End labels collide when lines finish within a few percent of
                // each other. Lay them out top-down with a minimum gap.
                const ends = lines.map(l => { const vs = l.v as any[]; return { l, last: vs[vs.length-1] }; })
                  .filter(e => e.last != null).sort((a,b) => (b.last as number) - (a.last as number));
                let prevY = -Infinity;
                return ends.map(e => {
                  let y = sy(e.last as number);
                  if (y - prevY < 12) y = prevY + 12;
                  prevY = y;
                  return (
                    <text key={e.l.t} x={W-PAD.r+6} y={y+3} fill={e.l.c} fontSize="10" fontFamily="monospace" fontWeight={e.l.me?700:400}>
                      {e.l.t} {(e.last as number)>=0?'+':''}{(e.last as number).toFixed(0)}%
                    </text>
                  );
                });
              })()}
              {hoverI != null && hoverI < dates.length && (
                <g>
                  <line x1={sx(hoverI)} y1={PAD.t} x2={sx(hoverI)} y2={H-PAD.b} stroke="rgba(212,149,108,0.45)" strokeWidth={1} />
                  {lines.map(l => { const v=(l.v as any[])[hoverI];
                    return v==null?null:<circle key={l.t} cx={sx(hoverI)} cy={sy(v)} r={3} fill={l.c} />; })}
                </g>
              )}
              <rect x={PAD.l} y={PAD.t} width={W-PAD.l-PAD.r} height={H-PAD.t-PAD.b} fill="transparent"
                onMouseLeave={()=>setHoverI(null)}
                onMouseMove={(e:any)=>{
                  const r = e.currentTarget.getBoundingClientRect();
                  const f = (e.clientX - r.left) / r.width;
                  setHoverI(Math.max(0, Math.min(dates.length-1, Math.round(f*(dates.length-1)))));
                }} />
              <text x={PAD.l} y={H-4} fill={C.textDim} fontSize="9" fontFamily="monospace">{dates[0]}</text>
              <text x={W-PAD.r} y={H-4} fill={C.textDim} fontSize="9" fontFamily="monospace" textAnchor="end">{dates[dates.length-1]}</text>
            </svg>
            <div style={{ minHeight:22, marginTop:2, display:'flex', gap:14, flexWrap:'wrap', alignItems:'baseline' }}>
              {hoverI != null ? (<>
                <span style={{ fontFamily:'monospace', fontSize:10.5, color:C.text }}>{dates[hoverI]}</span>
                {lines.map(l => { const v=(l.v as any[])[hoverI]; const px=relPerf.px_ticker?.[l.t]?.[i0+hoverI];
                  return v==null?null:(
                    <span key={l.t} style={{ fontFamily:'monospace', fontSize:10.5, color:l.c }}>
                      {l.t} {v>=0?'+':''}{v.toFixed(1)}%{px!=null?` ($${px})`:''}
                    </span>); })}
              </>) : (
                <span style={{ fontFamily:'monospace', fontSize:10, color:C.textDim }}>Hover the chart for prices on any date</span>
              )}
            </div>
            {relPerf.windows?.length > 0 && (
              <div style={{ display:'grid', gridTemplateColumns:`repeat(${relPerf.windows.length}, 1fr)`, gap:6, marginTop:8 }}>
                {relPerf.windows.map((w:any)=>(
                  <div key={w.window} style={{ background:'rgba(212,149,108,0.04)', borderRadius:4, padding:'6px 9px' }}>
                    <div style={{ fontFamily:'monospace', fontSize:9, color:C.textDim, letterSpacing:1 }}>{w.window} VS PEER MEDIAN</div>
                    <div style={{ fontFamily:'monospace', fontSize:14, fontWeight:700, marginTop:1,
                      color: w.relative_pts>=0 ? C.green : C.red }}>
                      {w.relative_pts>=0?'+':''}{w.relative_pts}pts
                    </div>
                  </div>
                ))}
              </div>
            )}
            <div style={{ display:'flex', gap:5, flexWrap:'wrap', marginTop:10, marginBottom:6 }}>
              {relPerf.roster.map((r:any)=>{
                const on = picked.includes(r.ticker);
                return (
                  <button key={r.ticker} onClick={()=>toggle(r.ticker)} title={fmtCap(r.market_cap)}
                    style={{ fontFamily:'monospace', fontSize:10, padding:'3px 9px', cursor:'pointer', borderRadius:3,
                      background: on?'rgba(218,165,32,0.10)':'transparent', color: on?C.text:C.textDim,
                      border:`1px solid ${on?'rgba(218,165,32,0.5)':C.border}` }}>
                    {r.ticker}
                  </button>
                );
              })}
              <span style={{ fontFamily:'monospace', fontSize:9.5, color:C.textDim, alignSelf:'center', marginLeft:4 }}>
                {picked.length} of 5 selected
              </span>
            </div>
            <div style={{ fontSize:10, color:C.textDim, marginBottom:22, fontFamily:'monospace' }}>
              {relPerf.n_sessions} sessions stored — the window extends as the nightly sync accumulates history.
            </div>
          </>
        );
      })()}

      {/* Growth ranking against the ten largest peers */}
      {pd.peers && pd.peers.length > 2 && (() => {
        const rows = pd.peers
          .filter(p => p.revenue_growth != null && p.market_cap != null)
          .sort((a, b) => (b.market_cap || 0) - (a.market_cap || 0))
          .slice(0, 10)
          .sort((a, b) => (b.revenue_growth || 0) - (a.revenue_growth || 0));
        if (rows.length < 3) return null;
        const meIn = rows.some(r => r.is_me);
        const meRow = pd.peers.find(p => p.is_me && p.revenue_growth != null);
        const all = meIn || !meRow ? rows : [...rows, meRow].sort((a,b)=>(b.revenue_growth||0)-(a.revenue_growth||0));
        const vals = all.map(r => r.revenue_growth as number);
        const lo = Math.min(0, ...vals), hi = Math.max(...vals);
        const span = (hi - lo) || 1;
        const zero = ((0 - lo) / span) * 100;
        const meRank = all.findIndex(r => r.is_me) + 1;

        return (
          <>
            <div style={{ color:C.gold, fontWeight:700, fontSize:13, marginBottom:4, marginTop:4 }}>GROWTH VS LARGEST RIVALS</div>
            <div style={{ fontSize:11, color:C.textDim, marginBottom:10, fontStyle:'italic' }}>
              Revenue growth for the ten biggest companies in the group, ranked.
              {meRank > 0 ? ` ${ticker} places ${meRank} of ${all.length}.` : ''}
            </div>
            <div style={{ marginBottom:24 }}>
              {all.map(r => {
                const v = r.revenue_growth as number;
                const w = (Math.abs(v) / span) * 100;
                const left = v >= 0 ? zero : zero - w;
                return (
                  <div key={r.ticker} title={`${r.ticker} — ${r.name}\ngrowth ${(v*100).toFixed(1)}%  ·  cap ${fmtCap(r.market_cap)}`}
                    style={{ display:'grid', gridTemplateColumns:'58px 1fr 72px 62px', gap:10, alignItems:'center', marginBottom:4, cursor:'default' }}>
                    <span style={{ fontFamily:'monospace', fontSize:11, fontWeight: r.is_me?700:400, color: r.is_me?C.gold:C.text }}>{r.ticker}</span>
                    <div style={{ position:'relative', height:16, background:'rgba(212,149,108,0.05)', borderRadius:2 }}>
                      <div style={{ position:'absolute', left:`${zero}%`, top:0, bottom:0, width:1, background:C.border }} />
                      <div style={{ position:'absolute', left:`${left}%`, width:`${w}%`, top:3, bottom:3, borderRadius:2,
                        background: r.is_me ? C.gold : v>=0 ? 'rgba(34,197,94,0.45)' : 'rgba(239,68,68,0.45)' }} />
                    </div>
                    <span style={{ fontFamily:'monospace', fontSize:11, textAlign:'right', color: r.is_me?C.gold:(v>=0?C.green:C.red) }}>
                      {v>=0?'+':''}{(v*100).toFixed(1)}%
                    </span>
                    <span style={{ fontFamily:'monospace', fontSize:10, textAlign:'right', color:C.textDim }}>{fmtCap(r.market_cap)}</span>
                  </div>
                );
              })}
              <div style={{ fontSize:10, color:C.textDim, marginTop:8, fontFamily:'monospace' }}>
                Ten largest of {pd.peers.length} peers by market cap · hover a row for detail
              </div>
            </div>
          </>
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
        if (!groups.length) {
          return (
            <div style={{ marginBottom:26 }}>
              <div style={{ color:C.gold, fontWeight:700, fontSize:13, marginBottom:4 }}>DISCLOSED RELATIONSHIPS</div>
              <div style={{ fontSize:11, color:C.textDim, lineHeight:1.55 }}>
                Nothing disclosed for {ticker} yet. Relationships are read from 10-K filings — suppliers must report
                customer concentration, so they name their buyers, while buyers rarely name anyone. Coverage grows as
                more filings are processed.
              </div>
            </div>
          );
        }
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
                  {(() => {
                    const seen = new Set<string>();
                    return (rel[g.key] || []).filter((x:any) => {
                      const k = (x.ticker || x.name || '').toLowerCase().replace(/[^a-z0-9]/g,'').slice(0,14);
                      if (!k || seen.has(k)) return false;
                      seen.add(k); return true;
                    }).slice(0,14);
                  })().map((x:any,i:number) => (
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
      <div style={{ color:C.gold, fontWeight:700, fontSize:13, marginBottom:8 }}>CLOSEST RIVALS BY BUSINESS PROFILE</div>
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
            {sortedPeers.slice(0,10).map((p,i) => {
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
