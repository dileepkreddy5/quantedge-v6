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
  insights?:{
    required_return_to_high_pct?:number;
    historical_recovery?:{drawdown_bucket:string;recovered_within_1y_pct:number;
      median_days_when_recovered:number|null;sample_size:number;note:string;};
    days_since_low?:number; off_the_lows?:boolean;
    accumulation_signal?:boolean; up_day_volume_share_pct?:number;
    analysis_url?:string;
  };
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
  const [showInfo,setShowInfo] = useState(false);

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

      <div style={{...card,padding:0,overflow:'hidden'}}>
        <div onClick={()=>setShowInfo(!showInfo)}
          style={{padding:'12px 16px',cursor:'pointer',display:'flex',justifyContent:'space-between',alignItems:'center'}}>
          <span style={{...label,fontSize:10}}>What do these scores mean? {showInfo?'▾':'▸'}</span>
        </div>
        {showInfo && (
          <div style={{padding:'0 16px 16px',color:DIM,fontSize:12,lineHeight:1.7,borderTop:`1px solid ${BORDER}`}}>
            <div style={{marginTop:12,color:'#cbb896'}}>
              <strong style={{color:GOLD}}>F-score (0–9)</strong> — the Piotroski score, a
              9-point financial-health checklist from accounting research (Piotroski, 2000).
              A company earns one point for each test it passes. 7–9 is strong, 4–6 is
              middling, 0–3 is weak. The nine tests:
            </div>
            <ol style={{marginTop:10,paddingLeft:20,color:DIM}}>
              <li>Positive net income (the company is profitable)</li>
              <li>Positive operating cash flow (profits are backed by real cash)</li>
              <li>Return on assets rising vs last year (getting more efficient)</li>
              <li>Cash flow greater than net income (earnings quality, not accounting tricks)</li>
              <li>Falling leverage (less debt relative to assets)</li>
              <li>Rising current ratio (better able to cover short-term bills)</li>
              <li>No share dilution (not printing new stock and shrinking your slice)</li>
              <li>Rising gross margin (pricing power improving)</li>
              <li>Rising asset turnover (generating more revenue per dollar of assets)</li>
            </ol>
            <div style={{marginTop:12,color:'#cbb896'}}>
              <strong style={{color:GOLD}}>Score (0–100)</strong> — QuantEdge's blend of the
              discount (how far below its own history), the F-score above, revenue-growth
              streak, and volume confirmation. Higher = a cleaner discounted-quality profile.
              It is a ranking of candidates, not a prediction.
            </div>
            <div style={{marginTop:12,color:'#cbb896'}}>
              <strong style={{color:GOLD}}>Stage</strong> —
              <span style={{color:RED}}> FALLING</span> (still declining),
              <span style={{color:GOLD}}> TURNING</span> (decline slowing),
              <span style={{color:GREEN}}> RECOVERING</span> (climbing back). Based on recent
              price action and volume.
            </div>
          </div>
        )}
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
            <span onClick={(e)=>{e.stopPropagation(); nav(r.insights?.analysis_url || `/dashboard?ticker=${r.ticker}`);}}
              style={{color:GOLD,fontWeight:700,fontSize:15,minWidth:70,textDecoration:'underline',cursor:'pointer'}}
              title="Open full analysis">{r.ticker}</span>
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
              {r.insights && (
                <div style={{marginTop:14,paddingTop:12,borderTop:`1px solid ${BORDER}`,fontSize:11.5}}>
                  {r.insights.required_return_to_high_pct!=null && (
                    <div style={{marginBottom:8}}>
                      <span style={{color:DIM}}>To reclaim its prior high this needs </span>
                      <span style={{color:GOLD,fontWeight:700}}>+{r.insights.required_return_to_high_pct}%</span>
                      <span style={{color:'#6a5a4a'}}> from here.</span>
                    </div>
                  )}
                  {r.insights.historical_recovery && (
                    <div style={{marginBottom:8}}>
                      <span style={{color:DIM}}>Historically, stocks down </span>
                      <span style={{color:'#cbb896'}}>{r.insights.historical_recovery.drawdown_bucket}%</span>
                      <span style={{color:DIM}}> reclaimed their high within a year </span>
                      <span style={{color:r.insights.historical_recovery.recovered_within_1y_pct>=10?GREEN:RED,fontWeight:700}}>
                        {r.insights.historical_recovery.recovered_within_1y_pct}% </span>
                      <span style={{color:DIM}}>of the time</span>
                      {r.insights.historical_recovery.median_days_when_recovered &&
                        <span style={{color:DIM}}> (median {r.insights.historical_recovery.median_days_when_recovered}d when they did)</span>}
                      <span style={{color:'#6a5a4a'}}> · {r.insights.historical_recovery.note}</span>
                    </div>
                  )}
                  <div style={{display:'flex',gap:16,flexWrap:'wrap',color:DIM}}>
                    {r.insights.days_since_low!=null &&
                      <span>{r.insights.days_since_low}d since low {r.insights.off_the_lows?'· off the lows':'· still near lows'}</span>}
                    {r.insights.accumulation_signal &&
                      <span style={{color:GREEN}}>◆ accumulation ({r.insights.up_day_volume_share_pct}% up-day volume)</span>}
                  </div>
                  <div style={{marginTop:10}}>
                    <span onClick={()=>nav(r.insights!.analysis_url || `/dashboard?ticker=${r.ticker}`)}
                      style={{color:GOLD,cursor:'pointer',textDecoration:'underline',fontSize:11}}>
                      → Open full analysis for {r.ticker}</span>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      ))}

      {data && rows.length===0 && !err &&
        <div style={{...card,color:DIM}}>No names in this tier right now.</div>}
    </div>
  );
}
