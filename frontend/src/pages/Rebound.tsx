// ============================================================
// QuantEdge v6.0 — Rebound Tracker (discounted-quality shortlist)
// Down hard from prior highs, strong financials. A research
// SCREENER, NOT a recovery predictor. Not investment advice.
// The recovery thesis was tested and did not validate on the
// available data window — this surfaces candidates, honestly.
// ============================================================
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../auth/authStore';

const GOLD='#daa520', DIM='#9d8b7a', BG='#1a0f0a', BORDER='#3a2920';
const GREEN='#6cae6c', RED='#c77';

interface Recovery { progress_pct:number|null; reached_high:boolean; upside_to_high_pct:number|null; }
interface Row {
  ticker:string; name:string; score:number; stage:string; tier:string;
  drawdown_from_high_pct:number|null; thesis:string;
  entry_price:number|null; prior_high:number|null;
  current_price?:number; recovery?:Recovery;
}
interface Artifact {
  as_of:string; generated:string; disclaimer:string;
  total_passed:number; n_universe?:number; n_prefilter?:number;
  tiers:Record<string,Row[]>; stage_counts?:Record<string,number>;
}

const stageColor=(s:string)=> s==='RECOVERING'?GREEN : s==='TURNING'?GOLD : s==='FALLING'?RED : DIM;

export default function Rebound(){
  const nav = useNavigate();
  const [data,setData] = useState<Artifact|null>(null);
  const [err,setErr] = useState<string|null>(null);
  const [tier,setTier] = useState<'small'|'mid'|'large'>('small');
  const [expanded,setExpanded] = useState<string|null>(null);

  useEffect(()=>{ api.get('/api/v6/rebound/list')
    .then(r=>{ if(r.data.error) setErr(r.data.error); else setData(r.data); })
    .catch(()=>setErr('Could not load rebound scan.')); },[]);

  const wrap:React.CSSProperties={minHeight:'100vh',background:'#0d0805',color:'#e8dcc8',
    fontFamily:"'Fira Code', monospace",padding:'24px'};
  const card:React.CSSProperties={background:BG,border:`1px solid ${BORDER}`,borderRadius:6,padding:16,marginBottom:16};
  const label:React.CSSProperties={color:GOLD,fontSize:11,letterSpacing:2,textTransform:'uppercase'};
  const tabBtn=(t:string,active:boolean):React.CSSProperties=>({
    background:active?'#2a1a0e':'transparent',border:`1px solid ${active?GOLD:BORDER}`,
    color:active?GOLD:DIM,padding:'8px 18px',borderRadius:4,cursor:'pointer',
    fontFamily:"'Fira Code', monospace",fontSize:11,letterSpacing:1,marginRight:8});

  const rows = data?.tiers[tier] ?? [];

  return (
    <div style={wrap}>
      <button onClick={()=>nav('/')} style={{background:'transparent',border:`1px solid ${BORDER}`,
        color:DIM,padding:'6px 14px',borderRadius:4,cursor:'pointer',fontSize:10,marginBottom:18}}>← HOME</button>
      <h1 style={{...label,fontSize:16}}>QuantEdge — Rebound Tracker</h1>

      <div style={{...card,borderColor:'#5a4020',background:'#23150a',color:'#e0b860',fontSize:12}}>
        <strong style={label}>RESEARCH SCREEN — NOT A PREDICTOR · NOT ADVICE</strong>
        <div style={{marginTop:8,color:DIM,lineHeight:1.6}}>
          Companies down significantly from their 3-year high that still show
          strong financials — growing revenue, healthy Piotroski scores, cheap
          versus their own history. This surfaces <strong>candidates</strong>.
          We tested whether good financials predict recovery to prior highs and
          it did <strong>not</strong> validate on the available data window, so
          nothing here is a buy signal. The "recovery" figures show how far each
          name has climbed back toward its prior high — as information, not a
          forecast. Research before acting.
        </div>
      </div>

      {data && (
        <div style={{...card,fontSize:12,color:DIM}}>
          <span style={label}>SCAN </span>
          {data.n_universe ?? '–'} universe → {data.n_prefilter ?? '–'} beaten-down →{' '}
          <span style={{color:GOLD}}>{data.total_passed}</span> passed quality gates
          <span style={{marginLeft:12,color:'#6a5a4a'}}>as of {data.as_of}</span>
        </div>
      )}

      <div style={{marginBottom:16}}>
        {(['small','mid','large'] as const).map(t=>(
          <button key={t} style={tabBtn(t,tier===t)} onClick={()=>setTier(t)}>
            {t.toUpperCase()} CAP</button>
        ))}
      </div>

      {err && <div style={card}>{err} — the nightly scan may not have run yet.</div>}

      {rows.map(r=>(
        <div key={r.ticker} style={{...card,padding:0,overflow:'hidden'}}>
          <div onClick={()=>setExpanded(expanded===r.ticker?null:r.ticker)}
            style={{display:'flex',alignItems:'center',gap:14,padding:'14px 16px',cursor:'pointer'}}>
            <span style={{color:GOLD,fontWeight:700,fontSize:15,minWidth:70}}>{r.ticker}</span>
            <span style={{color:'#e8dcc8',fontSize:12,minWidth:52,textAlign:'right'}}>{r.score.toFixed(1)}</span>
            <span style={{color:stageColor(r.stage),fontSize:10,letterSpacing:1,minWidth:96}}>{r.stage}</span>
            <span style={{color:RED,fontSize:12,minWidth:70}}>
              {r.drawdown_from_high_pct!=null?`-${r.drawdown_from_high_pct}%`:'–'}</span>
            {r.recovery && r.recovery.progress_pct!=null ? (
              <span style={{flex:1,minWidth:120}}>
                <span style={{fontSize:10,color:DIM}}>recovery </span>
                <span style={{display:'inline-block',width:90,height:8,background:'#2a1a0e',
                  borderRadius:4,verticalAlign:'middle',overflow:'hidden'}}>
                  <span style={{display:'block',height:'100%',width:`${r.recovery.progress_pct}%`,
                    background:r.recovery.reached_high?GREEN:GOLD}}/>
                </span>
                <span style={{fontSize:11,color:GOLD,marginLeft:6}}>{r.recovery.progress_pct}%</span>
              </span>
            ) : <span style={{flex:1,fontSize:10,color:'#6a5a4a'}}>recovery n/a</span>}
            <span style={{color:DIM,fontSize:12}}>{expanded===r.ticker?'▾':'▸'}</span>
          </div>
          {expanded===r.ticker && (
            <div style={{padding:'0 16px 16px',color:DIM,fontSize:12,lineHeight:1.7,borderTop:`1px solid ${BORDER}`}}>
              <div style={{marginTop:12,color:'#cbb896'}}>{r.thesis}</div>
              <div style={{marginTop:10,display:'flex',gap:24,flexWrap:'wrap',fontSize:11}}>
                <span>entry <span style={{color:GOLD}}>${r.entry_price?.toFixed(2)}</span></span>
                {r.current_price && <span>now <span style={{color:GOLD}}>${r.current_price.toFixed(2)}</span></span>}
                <span>prior high <span style={{color:GOLD}}>${r.prior_high?.toFixed(2)}</span></span>
                {r.recovery?.upside_to_high_pct!=null &&
                  <span>upside to high <span style={{color:GREEN}}>+{r.recovery.upside_to_high_pct}%</span></span>}
              </div>
            </div>
          )}
        </div>
      ))}

      {data && rows.length===0 && !err &&
        <div style={{...card,color:DIM}}>No names in this tier right now.</div>}
    </div>
  );
}
