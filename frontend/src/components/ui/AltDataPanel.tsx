import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface AData { ticker:string; available:boolean; score:number|null; altdata_rating:string;
  coverage:{scored:number;total:number}; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string; }

const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';
const rc=(r:string)=>/Strong|Positive/i.test(r)?'var(--gold)':/Neutral|Mixed/i.test(r)?'var(--neutral)':/Weak|Negative/i.test(r)?'var(--bear)':'var(--caramel)';

const fmtVal=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id.includes('ratio')||id.includes('velocity')||id.includes('trend')||id.includes('surge')||id.includes('divergence')||id.includes('sentiment')||id.includes('dispersion')||id.includes('expansion')||id.includes('attention')||id.includes('unusual')||id.includes('accum')) {
    if(Math.abs(v)<=2) return (v*100).toFixed(1)+'%';
  }
  if(id.includes('volume')||id.includes('breadth')||id.includes('filing')) return v.toFixed(0);
  if(id.includes('amihud')||id.includes('illiq')) return v.toFixed(4);
  return typeof v==='number'?v.toFixed(2):String(v);
};

// Signals whose sign has a directional meaning (inflow/bullish vs outflow/bearish).
// Score already encodes good/bad; we derive a left/right push from the score around 50.
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

export default function AltDataPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<AData|null>(null);
  const [loading,setLoading]=useState(false); const [err,setErr]=useState('');
  const [tip,setTip]=useState(''); const [showUnavail,setShowUnavail]=useState(false);

  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/altdata/${ticker}`).then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed')).finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'var(--cocoa-dust)',padding:24}}>Enter a ticker for Alt-Data Intelligence.</div>;
  if(loading)return <div style={{color:'var(--gold)',padding:24}}>Reading the flow — microstructure, news, insider filings, liquidity…</div>;
  if(err)return <div style={{color:'var(--bear)',padding:24}}>Alt-Data: {err}</div>;
  if(!d)return null;

  const cats=[...d.tree.categories];
  const live=cats.filter(c=>c.score!=null);
  const unavail=cats.filter(c=>c.score==null);
  const byScore=[...live].sort((a,b)=>(b.score!)-(a.score!));
  const drivers=[...byScore.slice(0,3), ...(byScore.length>3?[byScore[byScore.length-1]]:[])];
  const topSig=(c:Cat)=>[...c.signals].filter(x=>x.score!=null).sort((a,b)=>(b.score!)-(a.score!))[0];
  const stripCats=[...live].sort((a,b)=>(b.score!)-(a.score!));

  // Flatten live signals for the diverging flow view; push = score-50 (−50..+50).
  const flowSignals = live.flatMap(c=>c.signals.filter(s=>s.score!=null).map(s=>({...s, cat:c.label})));
  flowSignals.sort((a,b)=>(b.score!-50)-(a.score!-50));

  return (
    <div style={{padding:'8px 4px',color:'var(--latte)'}}>

      {/* ── Verdict + drivers ─────────────────────────── */}
      <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:16,padding:'20px 22px',marginBottom:14}}>
        <div style={{display:'flex',alignItems:'flex-start',gap:24,flexWrap:'wrap'}}>
          <div style={{display:'flex',alignItems:'center',gap:16}}>
            <Gauge score={d.score}/>
            <div>
              <div style={{fontSize:22,fontWeight:500,color:rc(d.altdata_rating)}}>{d.altdata_rating}</div>
              <div style={{fontSize:12,color:'var(--cocoa)',marginTop:3}}>{d.coverage.scored} / {d.coverage.total} flow signals · {unavail.reduce((a,c)=>a+c.n_signals,0)} need premium feeds</div>
              <div style={{fontSize:11,color:'var(--cocoa)',marginTop:8,maxWidth:230,lineHeight:1.5}}>Money-flow, news and insider signals from price, volume and SEC filings</div>
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
      <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,margin:'18px 0 10px'}}>FLOW DIMENSIONS · {live.length} live</div>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(158px,1fr))',gap:10}}>
        {stripCats.map(c=>(
          <div key={c.id} style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderLeft:`3px solid ${heat(c.score)}`,borderRadius:'0 10px 10px 0',padding:'11px 13px'}}>
            <span style={{fontSize:24,fontWeight:500,color:heat(c.score)}}>{c.score?.toFixed(0)}</span>
            <div style={{fontSize:11,color:'var(--cocoa-dust)',marginTop:3,lineHeight:1.3}}>{c.label}</div>
            <div style={{height:3,background:'var(--surface-3)',borderRadius:2,marginTop:8,overflow:'hidden'}}>
              <div style={{height:'100%',width:`${c.score}%`,background:heat(c.score)}}/>
            </div>
          </div>
        ))}
      </div>

      {/* ── Diverging flow bars ───────────────────────── */}
      <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,margin:'22px 0 4px'}}>SIGNAL FLOW · bullish pushes right, bearish left</div>
      <div style={{fontSize:10,color:'var(--cocoa)',marginBottom:12}}>{tip||'\u00A0'}</div>
      <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:12,padding:'14px 18px'}}>
        {flowSignals.map((s,i)=>{
          const push=(s.score!-50); const pctW=Math.abs(push)/50*50; // 0..50% of half-width
          const col=heat(s.score);
          return (
            <div key={s.id} onMouseEnter={()=>setTip(`${s.cat} — ${s.label} · score ${s.score!.toFixed(0)} · raw ${fmtVal(s.id,s.raw_value)}`)}
              onMouseLeave={()=>setTip('')}
              style={{display:'grid',gridTemplateColumns:'150px 1fr 34px',alignItems:'center',gap:12,padding:'6px 0',
                borderBottom:i<flowSignals.length-1?'1px solid var(--border-1)':'none',cursor:'pointer'}}>
              <div style={{fontSize:12,color:'var(--cocoa-dust)',textAlign:'right',whiteSpace:'nowrap',overflow:'hidden',textOverflow:'ellipsis'}}>{s.label}</div>
              <div style={{position:'relative',height:20}}>
                <div style={{position:'absolute',left:'50%',top:0,bottom:0,width:1,background:'var(--border-2)'}}/>
                <div style={{position:'absolute',top:3,bottom:3,background:col,borderRadius:3,
                  ...(push>=0?{left:'50%',width:`${pctW}%`}:{right:'50%',width:`${pctW}%`})}}/>
              </div>
              <div style={{fontSize:12,fontWeight:500,color:col,textAlign:'right'}}>{s.score!.toFixed(0)}</div>
            </div>
          );
        })}
      </div>

      {/* ── Unavailable / premium (honest) ───────────── */}
      {unavail.length>0 && (
        <div style={{marginTop:16}}>
          <div onClick={()=>setShowUnavail(v=>!v)} style={{display:'flex',alignItems:'center',gap:8,cursor:'pointer',marginBottom:8}}>
            <span style={{fontSize:10,color:'var(--cocoa)',letterSpacing:2}}>UNAVAILABLE ON THIS DATA TIER</span>
            <span style={{fontSize:10,color:'var(--cocoa)'}}>{unavail.reduce((a,c)=>a+c.n_signals,0)} signals · {showUnavail?'hide':'show'}</span>
          </div>
          {showUnavail && (
            <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(200px,1fr))',gap:10}}>
              {unavail.map(c=>(
                <div key={c.id} style={{background:'var(--surface-2)',border:'1px dashed var(--border-2)',borderRadius:10,padding:'11px 13px',opacity:0.7}}>
                  <div style={{fontSize:12,fontWeight:600,color:'var(--cocoa-dust)',marginBottom:6}}>{c.label}</div>
                  {c.signals.map(s=>(
                    <div key={s.id} style={{fontSize:11,color:'var(--cocoa)',padding:'2px 0',display:'flex',justifyContent:'space-between'}}>
                      <span>{s.label}</span><span style={{fontStyle:'italic'}}>premium</span>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
