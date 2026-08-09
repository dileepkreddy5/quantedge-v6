import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface IData { ticker:string; available:boolean; score:number|null; iflow_rating:string;
  coverage:{scored:number;total:number}; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>;
  series?:{adl:number[]|null; dollar_flow:number[]|null}; reason?:string; }

const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';
const rc=(r:string)=>/Strong Inflow|Net Inflow/i.test(r)?'var(--gold)':/Balanced/i.test(r)?'var(--neutral)':'var(--bear)';

const RELABEL:Record<string,string>={ holder_count:'5%+ disclosed holders (13G)', recent_13g:'New 13G filings (180d)' };

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

// Sparkline from a rebased 0..1 series.
const Spark=({data,color,w=120,h=34}:{data:number[];color:string;w?:number;h?:number})=>{
  if(!data||data.length<2) return <div style={{width:w,height:h}}/>;
  const pts=data.map((v,i)=>`${(i/(data.length-1))*w},${h-2-v*(h-4)}`).join(' ');
  const last=data[data.length-1];
  return (
    <svg width={w} height={h} style={{overflow:'visible'}}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" strokeLinejoin="round"/>
      <circle cx={w} cy={h-2-last*(h-4)} r="2.5" fill={color}/>
    </svg>
  );
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
  if(loading)return <div style={{color:'var(--gold)',padding:24}}>Weighing accumulation against distribution — money-flow, block trades, filings…</div>;
  if(err)return <div style={{color:'var(--bear)',padding:24}}>Inst-Flow: {err}</div>;
  if(!d)return null;

  const cats=[...d.tree.categories];
  const live=cats.filter(c=>c.score!=null);
  const byScore=[...live].sort((a,b)=>(b.score!)-(a.score!));
  const drivers=[...byScore.slice(0,3), ...(byScore.length>3?[byScore[byScore.length-1]]:[])];
  const lbl=(s:Sig)=>RELABEL[s.id]||s.label;
  const topSig=(c:Cat)=>[...c.signals].filter(x=>x.score!=null).sort((a,b)=>(b.score!)-(a.score!))[0];

  // All live signals as forces on the beam.
  const allSigs=live.flatMap(c=>c.signals.filter(s=>s.score!=null).map(s=>({...s,cat:c.label})));
  const net=allSigs.reduce((a,s)=>a+(s.score!-50),0)/allSigs.length;   // −50..+50
  const buySide=allSigs.filter(s=>s.score!>52).length;
  const sellSide=allSigs.filter(s=>s.score!<48).length;
  const neutral=allSigs.length-buySide-sellSide;
  const fulcrumPct=50+(net/50)*44;   // clamp within track
  const beamW=Math.min(Math.abs(net)/50*44,44);

  const adl=d.series?.adl||null;
  const dvf=d.series?.dollar_flow||null;

  // Money-flow signals that have a real trajectory to show.
  const mf=live.find(c=>c.id==='money_flow');
  const acc=live.find(c=>c.id==='accumulation');

  return (
    <div style={{padding:'8px 4px',color:'var(--latte)'}}>

      {/* ── Verdict + drivers (family DNA) ─────────────── */}
      <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:16,padding:'20px 22px',marginBottom:14}}>
        <div style={{display:'flex',alignItems:'flex-start',gap:24,flexWrap:'wrap'}}>
          <div style={{display:'flex',alignItems:'center',gap:16}}>
            <Gauge score={d.score}/>
            <div>
              <div style={{fontSize:22,fontWeight:500,color:rc(d.iflow_rating)}}>{d.iflow_rating}</div>
              <div style={{fontSize:12,color:'var(--cocoa)',marginTop:3}}>{d.coverage.scored} flow signals scored</div>
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

      {/* ── THE BEAM: accumulation vs distribution (hero) ── */}
      <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,margin:'20px 0 10px'}}>ACCUMULATION vs DISTRIBUTION</div>
      <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'22px 26px',marginBottom:14}}>
        <div style={{display:'flex',justifyContent:'space-between',fontSize:11,marginBottom:8}}>
          <span style={{color:'var(--bear)'}}>◄ DISTRIBUTION</span>
          <span style={{color:'var(--bull)'}}>ACCUMULATION ►</span>
        </div>
        <div style={{position:'relative',height:40}}>
          <div style={{position:'absolute',left:'50%',top:0,bottom:8,width:2,background:'var(--surface-4)',transform:'translateX(-1px)'}}/>
          <div style={{position:'absolute',top:'calc(50% - 4px)',height:8,borderRadius:4,transition:'all .4s',
            background:net>=0?'var(--bull)':'var(--bear)',
            ...(net>=0?{left:'50%',width:`${beamW}%`}:{right:'50%',width:`${beamW}%`})}}/>
          <div style={{position:'absolute',top:'calc(50% + 4px)',left:`${fulcrumPct}%`,width:0,height:0,transform:'translateX(-8px)',
            borderLeft:'8px solid transparent',borderRight:'8px solid transparent',borderTop:'12px solid var(--gold)'}}/>
        </div>
        <div style={{textAlign:'center',marginTop:12}}>
          <div style={{fontSize:34,fontWeight:500,color:net>=0?'var(--bull)':'var(--bear)',lineHeight:1}}>{net>=0?'+':''}{net.toFixed(0)}</div>
          <div style={{fontSize:12,color:'var(--cocoa)',marginTop:4}}>net {net>=0?'accumulation':'distribution'} pressure</div>
        </div>
        <div style={{display:'flex',justifyContent:'space-between',marginTop:18,paddingTop:14,borderTop:'1px solid var(--border-1)'}}>
          <div><div style={{fontSize:22,color:'var(--bear)',fontWeight:500}}>{sellSide}</div><div style={{fontSize:11,color:'var(--cocoa)'}}>pushing out</div></div>
          <div style={{textAlign:'center'}}><div style={{fontSize:22,color:'var(--neutral)',fontWeight:500}}>{neutral}</div><div style={{fontSize:11,color:'var(--cocoa)'}}>balanced</div></div>
          <div style={{textAlign:'right'}}><div style={{fontSize:22,color:'var(--bull)',fontWeight:500}}>{buySide}</div><div style={{fontSize:11,color:'var(--cocoa)'}}>pushing in</div></div>
        </div>
      </div>

      {/* ── Money-flow trajectory (real series sparklines) ── */}
      <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,margin:'20px 0 10px'}}>MONEY-FLOW TRAJECTORY · last 30 days</div>
      <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:12,padding:'8px 18px',marginBottom:14}}>
        {adl && (
          <div style={{display:'grid',gridTemplateColumns:'1fr 120px 44px',alignItems:'center',gap:16,padding:'11px 0',borderBottom:'1px solid var(--border-1)'}}>
            <div><div style={{fontSize:13,color:'var(--latte)'}}>Accumulation/Distribution line</div>
              <div style={{fontSize:11,color:'var(--cocoa)'}}>cumulative money-flow volume</div></div>
            <Spark data={adl} color={adl[adl.length-1]>=adl[0]?'var(--bull)':'var(--bear)'}/>
            <div style={{fontSize:12,fontWeight:500,color:heat((mf?.signals.find(x=>x.id==='adl')?.score)??null),textAlign:'right'}}>{mf?.signals.find(x=>x.id==='adl')?.score?.toFixed(0)??'—'}</div>
          </div>
        )}
        {dvf && (
          <div style={{display:'grid',gridTemplateColumns:'1fr 120px 44px',alignItems:'center',gap:16,padding:'11px 0'}}>
            <div><div style={{fontSize:13,color:'var(--latte)'}}>Dollar-flow</div>
              <div style={{fontSize:11,color:'var(--cocoa)'}}>daily dollar volume</div></div>
            <Spark data={dvf} color={dvf[dvf.length-1]>=dvf[0]?'var(--bull)':'var(--bear)'}/>
            <div style={{fontSize:12,fontWeight:500,color:heat((acc?.signals.find(x=>x.id==='dollar_flow')?.score)??null),textAlign:'right'}}>{acc?.signals.find(x=>x.id==='dollar_flow')?.score?.toFixed(0)??'—'}</div>
          </div>
        )}
      </div>

      {/* ── The forces (all signals, grouped by category) ── */}
      <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,margin:'20px 0 4px'}}>THE FORCES · every signal on the beam</div>
      <div style={{fontSize:10,color:'var(--cocoa)',marginBottom:10,minHeight:14}}>{tip||'\u00A0'}</div>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(230px,1fr))',gap:10}}>
        {live.map(c=>(
          <div key={c.id} style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderLeft:`3px solid ${heat(c.score)}`,borderRadius:'0 10px 10px 0',padding:'12px 14px'}}>
            <div style={{display:'flex',justifyContent:'space-between',alignItems:'baseline',marginBottom:8}}>
              <span style={{fontSize:13,fontWeight:600,color:'var(--latte)'}}>{c.label}</span>
              <span style={{fontSize:16,fontWeight:500,color:heat(c.score)}}>{c.score?.toFixed(0)}</span>
            </div>
            {c.signals.filter(s=>s.score!=null).map(s=>{
              const push=s.score!-50; const pct=Math.abs(push)/50*50;
              return (
                <div key={s.id} onMouseEnter={()=>setTip(`${lbl(s)} · score ${s.score!.toFixed(0)} · ${fmtVal(s.id,s.raw_value)} — ${s.evidence}`)}
                  onMouseLeave={()=>setTip('')}
                  style={{display:'grid',gridTemplateColumns:'1fr 60px 22px',alignItems:'center',gap:8,padding:'4px 0',cursor:'pointer'}}>
                  <span style={{fontSize:11,color:'var(--cocoa-dust)',whiteSpace:'nowrap',overflow:'hidden',textOverflow:'ellipsis'}}>{lbl(s)}</span>
                  <div style={{position:'relative',height:12}}>
                    <div style={{position:'absolute',left:'50%',top:0,bottom:0,width:1,background:'var(--border-2)'}}/>
                    <div style={{position:'absolute',top:2,bottom:2,background:heat(s.score),borderRadius:2,
                      ...(push>=0?{left:'50%',width:`${pct}%`}:{right:'50%',width:`${pct}%`})}}/>
                  </div>
                  <span style={{fontSize:11,fontWeight:500,color:heat(s.score),textAlign:'right'}}>{s.score!.toFixed(0)}</span>
                </div>
              );
            })}
          </div>
        ))}
      </div>

      <div style={{fontSize:10,color:'var(--cocoa)',marginTop:14,fontStyle:'italic'}}>
        13G reflects 5%+ disclosed holders only — full institutional ownership (13F) is not on this data tier.
      </div>
    </div>
  );
}
