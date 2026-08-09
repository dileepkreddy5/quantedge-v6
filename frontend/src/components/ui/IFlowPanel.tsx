import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface IData { ticker:string; available:boolean; score:number|null; iflow_rating:string;
  coverage:{scored:number;total:number}; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string; }

const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';
const rc=(r:string)=>/Strong Inflow|Net Inflow/i.test(r)?'var(--gold)':/Balanced/i.test(r)?'var(--neutral)':'var(--bear)';

// Honest relabels: some catalog labels overclaim. 13G is 5%+ disclosed holders, not full institutional ownership.
const RELABEL:Record<string,string>={
  holder_count:'5%+ disclosed holders (13G)',
  recent_13g:'New 13G filings (180d)',
};

const fmtVal=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id==='mfi'||id==='cmf'||id==='accum20'||id==='insider_net') return (v*100).toFixed(1)+'%';
  if(id==='adl'||id==='trade_size'||id==='dollar_flow') return (v>=0?'+':'')+(v*100).toFixed(1)+'%';
  if(id==='block_freq') return (v*100).toFixed(0)+'% of days';
  if(id==='insider_vel') return v.toFixed(0)+' filings';
  if(id==='recent_13g'||id==='holder_count') return v.toFixed(0);
  if(id==='insider_cluster') return v>0?'yes':'no';
  return typeof v==='number'?v.toFixed(2):String(v);
};

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

export default function IFlowPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<IData|null>(null);
  const [loading,setLoading]=useState(false); const [err,setErr]=useState('');
  const [tip,setTip]=useState('');

  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/iflow/${ticker}`).then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed')).finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'var(--cocoa-dust)',padding:24}}>Enter a ticker for Institutional Flow.</div>;
  if(loading)return <div style={{color:'var(--gold)',padding:24}}>Reading the flow — money-flow indicators, block trades, filings…</div>;
  if(err)return <div style={{color:'var(--bear)',padding:24}}>Inst-Flow: {err}</div>;
  if(!d)return null;

  const cats=[...d.tree.categories];
  const live=cats.filter(c=>c.score!=null);
  const byScore=[...live].sort((a,b)=>(b.score!)-(a.score!));
  const drivers=[...byScore.slice(0,3), ...(byScore.length>3?[byScore[byScore.length-1]]:[])];
  const lbl=(s:Sig)=>RELABEL[s.id]||s.label;
  const topSig=(c:Cat)=>[...c.signals].filter(x=>x.score!=null).sort((a,b)=>(b.score!)-(a.score!))[0];
  const stripCats=[...live].sort((a,b)=>(b.score!)-(a.score!));

  const flowSignals=live.flatMap(c=>c.signals.filter(s=>s.score!=null).map(s=>({...s,cat:c.label})));
  flowSignals.sort((a,b)=>(b.score!-50)-(a.score!-50));

  // Net-flow summary: share of live signals leaning inflow (score>50) vs outflow.
  const inflow=flowSignals.filter(s=>s.score!>52).length;
  const outflow=flowSignals.filter(s=>s.score!<48).length;
  const neutral=flowSignals.length-inflow-outflow;

  return (
    <div style={{padding:'8px 4px',color:'var(--latte)'}}>

      {/* ── Verdict + drivers ─────────────────────────── */}
      <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:16,padding:'20px 22px',marginBottom:14}}>
        <div style={{display:'flex',alignItems:'flex-start',gap:24,flexWrap:'wrap'}}>
          <div style={{display:'flex',alignItems:'center',gap:16}}>
            <Gauge score={d.score}/>
            <div>
              <div style={{fontSize:22,fontWeight:500,color:rc(d.iflow_rating)}}>{d.iflow_rating}</div>
              <div style={{fontSize:12,color:'var(--cocoa)',marginTop:3}}>{d.coverage.scored} flow signals · {inflow} inflow / {outflow} outflow</div>
              <div style={{fontSize:11,color:'var(--cocoa)',marginTop:8,maxWidth:230,lineHeight:1.5}}>Money-flow, block-trade and filing signals from price/volume and SEC data</div>
            </div>
          </div>
          <div style={{flex:1,minWidth:280}}>
            <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,marginBottom:10}}>WHAT'S DRIVING IT</div>
            {drivers.map(c=>{const sg=topSig(c);return (
              <div key={c.id} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0'}}>
                <div style={{width:6,height:6,borderRadius:'50%',background:heat(c.score),flexShrink:0}}/>
                <div style={{fontSize:13,color:'var(--latte)',flex:1}}>{c.label}</div>
                {sg && <div style={{fontSize:11,color:'var(--cocoa-dust)'}}>{lbl(sg)} {fmtVal(sg.id,sg.raw_value)}</div>}
                <div style={{fontSize:13,fontWeight:500,color:heat(c.score),width:26,textAlign:'right'}}>{c.score?.toFixed(0)}</div>
              </div>);})}
          </div>
        </div>
      </div>

      {/* ── Net-flow balance bar (Inst-Flow-specific) ── */}
      <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:12,padding:'12px 16px',marginBottom:14}}>
        <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,marginBottom:8}}>NET FLOW BALANCE</div>
        <div style={{display:'flex',height:14,borderRadius:7,overflow:'hidden',background:'var(--surface-3)'}}>
          <div style={{width:`${inflow/flowSignals.length*100}%`,background:'var(--bull)'}}/>
          <div style={{width:`${neutral/flowSignals.length*100}%`,background:'var(--neutral)'}}/>
          <div style={{width:`${outflow/flowSignals.length*100}%`,background:'var(--bear)'}}/>
        </div>
        <div style={{display:'flex',justifyContent:'space-between',marginTop:6,fontSize:11,color:'var(--cocoa)'}}>
          <span style={{color:'var(--bull)'}}>● {inflow} inflow</span>
          <span style={{color:'var(--neutral)'}}>● {neutral} neutral</span>
          <span style={{color:'var(--bear)'}}>● {outflow} outflow</span>
        </div>
      </div>

      {/* ── Dimension strip ───────────────────────────── */}
      <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,margin:'18px 0 10px'}}>FLOW DIMENSIONS · {live.length} categories</div>
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
      <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,margin:'22px 0 4px'}}>SIGNAL FLOW · inflow pushes right, outflow left</div>
      <div style={{fontSize:10,color:'var(--cocoa)',marginBottom:12,minHeight:14}}>{tip||'\u00A0'}</div>
      <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:12,padding:'14px 18px'}}>
        {flowSignals.map((s,i)=>{
          const push=(s.score!-50); const pctW=Math.abs(push)/50*50;
          const col=heat(s.score);
          return (
            <div key={s.id} onMouseEnter={()=>setTip(`${s.cat} — ${lbl(s)} · score ${s.score!.toFixed(0)} · ${fmtVal(s.id,s.raw_value)}`)}
              onMouseLeave={()=>setTip('')}
              style={{display:'grid',gridTemplateColumns:'160px 1fr 34px',alignItems:'center',gap:12,padding:'6px 0',
                borderBottom:i<flowSignals.length-1?'1px solid var(--border-1)':'none',cursor:'pointer'}}>
              <div style={{fontSize:12,color:'var(--cocoa-dust)',textAlign:'right',whiteSpace:'nowrap',overflow:'hidden',textOverflow:'ellipsis'}}>{lbl(s)}</div>
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

      <div style={{fontSize:10,color:'var(--cocoa)',marginTop:10,fontStyle:'italic'}}>
        13G reflects 5%+ disclosed holders only — full institutional ownership (13F) is not on this data tier.
      </div>
    </div>
  );
}
