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
interface PeerRow { ticker: string; name: string; market_cap: number | null; mom_3m: number | null; mom_6m: number | null; pct_above_ma200: number | null; is_me: boolean; }
interface PeerData {
  available: boolean; reason?: string; ticker: string; name: string;
  bucket: string; peer_count: number; factors: FactorPct[]; peers: PeerRow[];
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

  const load = useCallback(async () => {
    if (!ticker) { setLoading(false); return; }
    setLoading(true); setErr(false);
    try {
      const res = await api.get(`/api/v6/peers/${ticker}`);
      setPd(res.data?.data || null);
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
