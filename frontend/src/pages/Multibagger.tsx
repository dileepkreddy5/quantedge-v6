// ============================================================
// QuantEdge v6.0 — Multibagger Shortlist (cap-tier scanner)
// Ranks small/mid/large caps by fundamental score + price ladder.
// A FILTER/shortlist, NOT a predictor. Not investment advice.
// ============================================================
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../auth/authStore';

const GOLD='#daa520', DIM='#9d8b7a', BG='#1a0f0a', BORDER='#3a2920';
const LADDER = ['1d','3d','1w','2w','1m','2m','3m'];
const pct = (x:number|null) => x==null ? '–' : (x>=0?'+':'')+(x*100).toFixed(1)+'%';

interface Row {
  ticker:string; score:number; qtr_yoy_growth:number; piotroski:number;
  price_move_6mo:number|null; quiet_price:boolean;
  margin_trend:number|null; accruals:number|null; debt_trend:number|null;
  vol_change_pct:number|null; up_vol_ratio:number|null; market_cap?:number;
  price_ladder:Record<string,number|null>;
  beneficiaries?:{ticker:string;revenue_share:number;note:string}[];
}
interface Artifact { generated:string; disclaimer:string; tiers:Record<string,Row[]>; }

export default function Multibagger(){
  const nav = useNavigate();
  const [data,setData] = useState<Artifact|null>(null);
  const [err,setErr] = useState<string|null>(null);
  const [tier,setTier] = useState<'small'|'mid'|'large'>('small');
  const [expanded,setExpanded] = useState<string|null>(null);

  useEffect(()=>{ api.get('/api/v6/scan/tiers')
    .then(r=>setData(r.data)).catch(()=>setErr('Could not load scan.')); },[]);

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
      <h1 style={{...label,fontSize:16}}>QuantEdge — Multibagger Shortlist</h1>

      <div style={{...card,borderColor:'#5a4020',background:'#23150a',color:'#e0b860',fontSize:12}}>
        <strong style={label}>SHORTLIST — NOT A PREDICTOR · NOT ADVICE</strong>
        <div style={{marginTop:8,color:DIM,lineHeight:1.6}}>
          Companies ranked by quarterly revenue growth, quality (Piotroski), and a
          quiet price. This surfaces <strong>candidates</strong> with the multibagger
          profile — it does not predict winners. Backtests show the biggest movers are
          often unidentifiable from prior fundamentals. The price ladder (1d–3m) is
          timing context: a high score with a still-quiet price means the window may
          still be open. Research before acting.
        </div>
      </div>

      <div style={{marginBottom:16}}>
        {(['small','mid','large'] as const).map(t=>(
          <button key={t} style={tabBtn(t,tier===t)} onClick={()=>setTier(t)}>
            {t.toUpperCase()} CAP</button>
        ))}
      </div>

      {err && <div style={card}>{err}</div>}
      {!data && !err && <div style={{color:DIM}}>Loading…</div>}

      {data && (
        <div style={card}>
          <div style={{overflowX:'auto'}}>
            <table style={{width:'100%',borderCollapse:'collapse',fontSize:11,whiteSpace:'nowrap'}}>
              <thead><tr style={{color:GOLD,textAlign:'right'}}>
                <th style={{padding:6,textAlign:'left'}}>#</th>
                <th style={{textAlign:'left'}}>TICKER</th>
                <th>SCORE</th><th>QTR YOY</th><th>PIOT</th><th>MARGIN↑</th><th>ACCRUAL</th><th>DEBT↑</th><th>VOLΔ</th><th>UP-VOL</th><th>QUIET</th>
                {LADDER.map(l=><th key={l}>{l.toUpperCase()}</th>)}
              </tr></thead>
              <tbody>
                {rows.map((r,i)=>{
                  const hasKids = r.beneficiaries && r.beneficiaries.length>0;
                  const isOpen = expanded===r.ticker;
                  return (
                  <React.Fragment key={r.ticker}>
                  <tr style={{borderTop:`1px solid ${BORDER}`,color:'#cbb89a',textAlign:'right'}}>
                    <td style={{padding:6,textAlign:'left',color:DIM}}>{i+1}</td>
                    <td style={{textAlign:'left',color:GOLD,cursor:hasKids?'pointer':'default'}}
                        onClick={()=>hasKids&&setExpanded(isOpen?null:r.ticker)}>
                        {hasKids?(isOpen?'▾ ':'▸ '):''}{r.ticker}
                        <span style={{color:DIM,cursor:'pointer',marginLeft:6,fontSize:9}}
                          onClick={(e)=>{e.stopPropagation();nav('/dashboard?ticker='+r.ticker);}}>↗</span>
                    </td>
                    <td style={{color:'#e8dcc8'}}>{r.score.toFixed(1)}</td>
                    <td style={{color:r.qtr_yoy_growth>=0?'#6fbf73':'#cf6b5a'}}>{pct(r.qtr_yoy_growth)}</td>
                    <td>{r.piotroski}/9</td>
                    <td style={{color:(r.margin_trend??0)>=0?'#6fbf73':'#cf6b5a'}}>{pct(r.margin_trend)}</td>
                    <td style={{color:(r.accruals??0)<=0?'#6fbf73':'#cf6b5a'}}>{r.accruals==null?'–':r.accruals.toFixed(3)}</td>
                    <td style={{color:(r.debt_trend??0)<=0?'#6fbf73':'#cf6b5a'}}>{pct(r.debt_trend)}</td>
                    <td style={{color:(r.vol_change_pct??0)>=0?'#6fbf73':DIM}}>{pct(r.vol_change_pct)}</td>
                    <td style={{color:(r.up_vol_ratio??0)>=0.5?'#6fbf73':DIM}}>{r.up_vol_ratio==null?'–':(r.up_vol_ratio*100).toFixed(0)+'%'}</td>
                    <td style={{color:r.quiet_price?'#6fbf73':DIM}}>{r.quiet_price?'yes':'no'}</td>
                    {LADDER.map(l=>{const v=r.price_ladder?.[l];
                      return <td key={l} style={{color:v==null?DIM:(v>=0?'#6fbf73':'#cf6b5a')}}>{pct(v??null)}</td>;})}
                  </tr>
                  {isOpen && hasKids && (
                    <tr style={{background:'#140c06'}}>
                      <td colSpan={16} style={{padding:'8px 16px',textAlign:'left'}}>
                        <div style={{color:GOLD,fontSize:10,letterSpacing:1,marginBottom:6}}>
                          KNOWN BENEFICIARIES — if {r.ticker} grows, these tend to ride along
                          <span style={{color:DIM,marginLeft:8,letterSpacing:0}}>(curated from 10-K disclosures; not exhaustive)</span>
                        </div>
                        <div style={{display:'flex',flexWrap:'wrap',gap:8}}>
                          {r.beneficiaries!.map(b=>(
                            <span key={b.ticker} onClick={()=>nav('/dashboard?ticker='+b.ticker)}
                              style={{border:`1px solid ${BORDER}`,borderRadius:4,padding:'4px 10px',cursor:'pointer',fontSize:11,color:'#cbb89a'}}>
                              <span style={{color:GOLD}}>{b.ticker}</span>
                              {b.revenue_share>0 && <span style={{color:DIM}}> · {(b.revenue_share*100).toFixed(0)}% rev</span>}
                              <span style={{color:DIM}}> · {b.note}</span>
                            </span>
                          ))}
                        </div>
                      </td>
                    </tr>
                  )}
                  </React.Fragment>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div style={{color:DIM,fontSize:10,marginTop:10}}>Generated {data.generated?.slice(0,16)} · {data.disclaimer}</div>
        </div>
      )}
    </div>
  );
}
