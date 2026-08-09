import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface FData { ticker:string; available:boolean; score:number|null; forecast_rating:string; confidence?:number;
  coverage:{scored:number;total:number}; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string; }

// Shared score ramp — same gold/caramel/burnt/bear convention as Risk & News.
const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';
const ink =(s:number|null)=>s!=null&&s>=50?'#241510':'var(--cream)';   // legible text on a hot vs cold cell
const rc=(r:string)=>r==='Bullish Outlook'?'var(--gold)':r==='Constructive'?'var(--caramel)':r==='Neutral'?'var(--neutral)':r==='Cautious'?'#c9762f':'var(--bear)';

const pct=(v:number|null,d=1)=>v==null?'—':(v>=0?'+':'')+(v*100).toFixed(d)+'%';

// Format a raw signal value for the hover tooltip, mirroring the old fmt() rules.
const fmtVal=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id.includes('roiic')||id.includes('margin')||id.includes('growth')||id.includes('accel')||id.includes('slope')||id.includes('mom')||id.includes('traj')||id.includes('pull')||id.includes('dist')||id.includes('trend')||id.includes('yoy')||id.includes('stack')||id.includes('intrinsic')||id.includes('fcf')||id.includes('ocf')||id.includes('cash_traj')||(id.includes('rule_of_40')&&id!=='rule40_pass')||id.includes('vs_ma')||id.includes('golden')||id.includes('ext')||id.includes('wc_')) return (v*100).toFixed(1)+'%';
  if(id.includes('pass')||id.includes('align')||id.includes('agree')||id.includes('pos')||id.includes('consist')||id.includes('stab')||id.includes('persist')||id.includes('conf')||id.includes('comp')||id.includes('up_days')||id.includes('rsi')||id.includes('quality')||id.includes('anchor')||id.includes('conv_lvl')){ if(Math.abs(v)<=1.5) return (v*100).toFixed(0)+'%'; }
  if(id.includes('leverage')||id.includes('inc_margin')) return v.toFixed(2)+'x';
  return typeof v==='number'?v.toFixed(2):String(v);
};

// Gauge arc for the headline score.
const Gauge=({score}:{score:number|null})=>{
  const s=score??0; const r=38, circ=2*Math.PI*r; const dash=(s/100)*circ*0.75; const col=heat(score);
  return (
    <svg width="96" height="96" viewBox="0 0 96 96">
      <circle cx="48" cy="48" r={r} fill="none" stroke="var(--surface-3)" strokeWidth="7"
        strokeDasharray={`${circ*0.75} ${circ}`} transform="rotate(135 48 48)" strokeLinecap="round"/>
      <circle cx="48" cy="48" r={r} fill="none" stroke={col} strokeWidth="7"
        strokeDasharray={`${dash} ${circ}`} transform="rotate(135 48 48)" strokeLinecap="round"/>
      <text x="48" y="46" textAnchor="middle" fontSize="30" fontWeight="500" fill="var(--latte)">{score?.toFixed(0)??'—'}</text>
      <text x="48" y="62" textAnchor="middle" fontSize="9" fill="var(--cocoa)" letterSpacing="1">/ 100</text>
    </svg>
  );
};

export default function ForecastPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<FData|null>(null);
  const [loading,setLoading]=useState(false); const [err,setErr]=useState('');
  const [tip,setTip]=useState<string>(''); const [detail,setDetail]=useState(false);

  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/forecast/${ticker}`).then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed')).finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'var(--cocoa-dust)',padding:24}}>Enter a ticker for Forecast Intelligence.</div>;
  if(loading)return <div style={{color:'var(--gold)',padding:24}}>Projecting forward — earnings trajectory, compounding, momentum quality…</div>;
  if(err)return <div style={{color:'var(--bear)',padding:24}}>Forecast: {err}</div>;
  if(!d)return null;

  const cats=[...d.tree.categories];
  const scored=cats.filter(c=>c.score!=null);
  // Drivers: the two strongest and the single weakest scored categories, each
  // with its own strongest signal as the concrete evidence line.
  const byScore=[...scored].sort((a,b)=>(b.score!)-(a.score!));
  const drivers=[...byScore.slice(0,3), ...(byScore.length>3?[byScore[byScore.length-1]]:[])];
  const topSig=(c:Cat)=>{const s=[...c.signals].filter(x=>x.score!=null).sort((a,b)=>(b.score!)-(a.score!))[0]; return s;};
  const sortedForStrip=[...scored].sort((a,b)=>(b.score!)-(a.score!));

  return (
    <div style={{padding:'8px 4px',color:'var(--latte)'}}>

      {/* ── Verdict + drivers ─────────────────────────── */}
      <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:16,padding:'20px 22px',marginBottom:14}}>
        <div style={{display:'flex',alignItems:'flex-start',gap:24,flexWrap:'wrap'}}>
          <div style={{display:'flex',alignItems:'center',gap:16}}>
            <Gauge score={d.score}/>
            <div>
              <div style={{fontSize:22,fontWeight:500,color:rc(d.forecast_rating)}}>{d.forecast_rating}</div>
              <div style={{fontSize:12,color:'var(--cocoa)',marginTop:3}}>{d.coverage.scored} / {d.coverage.total} forward signals scored</div>
              <div style={{fontSize:11,color:'var(--cocoa)',marginTop:8,maxWidth:230,lineHeight:1.5}}>Model projection from historical trajectory — not analyst consensus</div>
            </div>
          </div>
          <div style={{flex:1,minWidth:280}}>
            <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,marginBottom:10}}>WHAT'S DRIVING IT</div>
            {drivers.map(c=>{const sg=topSig(c);return (
              <div key={c.id} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0'}}>
                <div style={{width:6,height:6,borderRadius:'50%',background:heat(c.score),flexShrink:0}}/>
                <div style={{fontSize:13,color:'var(--latte)',flex:1}}>{c.label}</div>
                {sg && <div style={{fontSize:11,color:'var(--cocoa-dust)'}}>{sg.label} {fmtVal(sg.id,sg.raw_value)}</div>}
                <div style={{fontSize:13,fontWeight:500,color:heat(c.score),width:26,textAlign:'right'}}>{c.score?.toFixed(0)}</div>
              </div>);})}
          </div>
        </div>
      </div>

      {/* ── Dimension strip ───────────────────────────── */}
      <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,margin:'18px 0 10px'}}>DIMENSION STRENGTH · {scored.length} categories</div>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(158px,1fr))',gap:10}}>
        {sortedForStrip.map(c=>(
          <div key={c.id} style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderLeft:`3px solid ${heat(c.score)}`,borderRadius:'0 10px 10px 0',padding:'11px 13px'}}>
            <span style={{fontSize:24,fontWeight:500,color:heat(c.score)}}>{c.score?.toFixed(0)}</span>
            <div style={{fontSize:11,color:'var(--cocoa-dust)',marginTop:3,lineHeight:1.3}}>{c.label}</div>
            <div style={{height:3,background:'var(--surface-3)',borderRadius:2,marginTop:8,overflow:'hidden'}}>
              <div style={{height:'100%',width:`${c.score}%`,background:heat(c.score)}}/>
            </div>
          </div>
        ))}
      </div>

      {/* ── Signal map (Option 4: number on the standouts only) ── */}
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',margin:'22px 0 10px'}}>
        <span style={{fontSize:10,color:'var(--gold)',letterSpacing:2}}>SIGNAL MAP · hover any cell</span>
        <button onClick={()=>setDetail(v=>!v)}
          style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',color:'var(--cocoa-dust)',borderRadius:8,padding:'4px 12px',fontSize:11,cursor:'pointer'}}>
          {detail?'Hide table':'Show full table'}</button>
      </div>
      <div style={{display:'flex',flexDirection:'column',gap:5}}>
        {cats.map(c=>(
          <div key={c.id} style={{display:'flex',alignItems:'center',gap:10}}>
            <div style={{fontSize:11,color:'var(--cocoa)',width:118,flexShrink:0,textAlign:'right',lineHeight:1.2}}>{c.label}</div>
            <div style={{display:'flex',gap:4,flexWrap:'wrap'}}>
              {c.signals.map(s=>{
                const pending=s.status==='needs_source'||s.score==null;
                const show=!pending && (s.score!>=80 || s.score!<=30);
                return (
                  <div key={s.id}
                    onMouseEnter={()=>setTip(`${s.label} · ${pending?'—':s.score!.toFixed(0)}${s.raw_value!=null?'  ('+fmtVal(s.id,s.raw_value)+')':''}`)}
                    onMouseLeave={()=>setTip('')}
                    style={{width:34,height:34,borderRadius:6,background:heat(s.score),opacity:pending?0.4:1,
                      display:'flex',alignItems:'center',justifyContent:'center',cursor:'pointer',
                      fontSize:11,fontWeight:500,color:ink(s.score),transition:'transform .1s'}}
                    onMouseOver={e=>{(e.currentTarget as HTMLDivElement).style.transform='scale(1.15)';}}
                    onMouseOut={e=>{(e.currentTarget as HTMLDivElement).style.transform='scale(1)';}}>
                    {show?s.score!.toFixed(0):''}
                  </div>);
              })}
            </div>
          </div>
        ))}
      </div>
      <div style={{height:20,marginTop:12,fontSize:12,textAlign:'center',color:tip?'var(--latte)':'var(--cocoa)'}}>{tip||'\u00A0'}</div>

      {/* ── Optional full table (kept for anyone who wants raw numbers) ── */}
      {detail && (
        <div style={{marginTop:10,display:'flex',flexDirection:'column',gap:8}}>
          {cats.map(cat=>(
            <div key={cat.id} style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:10,overflow:'hidden'}}>
              <div style={{display:'flex',alignItems:'center',gap:12,padding:'9px 14px',borderLeft:`4px solid ${heat(cat.score)}`}}>
                <span style={{fontSize:13,fontWeight:600,color:'var(--latte)',flex:1}}>{cat.label}</span>
                <span style={{fontSize:10,color:'var(--cocoa)'}}>wt {(cat.weight??0).toFixed(2)} · {cat.n_scored??0}/{cat.n_signals??0}</span>
                <span style={{fontSize:16,fontWeight:700,color:heat(cat.score),width:32,textAlign:'right'}}>{cat.score?.toFixed(0)??'—'}</span>
              </div>
              <div style={{padding:'4px 14px 12px 30px'}}>
                {cat.signals.map(s=>{const pending=s.status==='needs_source'||s.score==null;return (
                  <div key={s.id} title={s.evidence} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0',borderBottom:'1px solid var(--border-1)',opacity:pending?0.5:1}}>
                    <span style={{fontSize:12,color:'var(--latte)',flex:1}}>{s.label}</span>
                    <span style={{fontSize:12,color:'var(--cocoa-dust)',width:64,textAlign:'right'}}>{pending?'—':fmtVal(s.id,s.raw_value)}</span>
                    <div style={{width:80,height:6,background:'var(--surface-3)',borderRadius:3,overflow:'hidden'}}>{!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score)}}/>}</div>
                    <span style={{fontSize:11,fontWeight:600,color:pending?'var(--cocoa)':heat(s.score),width:26,textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
                  </div>);})}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
