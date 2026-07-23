import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface CData { ticker:string; available:boolean; score:number|null; competitive_rating:string; confidence:number;
  coverage:{scored:number;total:number}; bucket:string; peer_count:number; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string; }

const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';
const ratingColor=(r:string)=>r==='Dominant'?'var(--gold)':r==='Strong'?'var(--gold)':r==='Competitive'?'var(--caramel)':r==='Challenged'?'#c9762f':'var(--bear)';
const fmt=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id.includes('pctile')||id.includes('rank')||id.includes('_pct')||id.includes('proxy')||id.includes('premium')||id.includes('discount')||id.includes('cheap')||id.includes('dominance')||id.includes('durability')||id.includes('persistence')||id.includes('excess')||id.includes('efficiency')) { if(Math.abs(v)<=1.01) return (v*100).toFixed(0)+'th'; }
  if(id.includes('advantage')||id.includes('spread')||id.includes('level')||id.includes('growth')||id.includes('decel')) return (v*100).toFixed(1)+'%';
  if(id.includes('scale')&&id.includes('abs')) return '$'+v.toFixed(0)+'B';
  if(id.includes('employee')) return v.toLocaleString();
  if(id.includes('risk')||id.includes('loss')||id.includes('comp')) return v>0?'Yes':'No';
  if(id.includes('liquidity')||id.includes('rel')) return v.toFixed(2);
  return typeof v==='number'?v.toFixed(2):v;
};

// RadialBars removed — four dials showed a 100th percentile for being the
// largest of eight peers. The profile chart replaces it.

export default function CompetitivePanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<CData|null>(null);
  const [loading,setLoading]=useState(false); const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({}); const [allOpen,setAllOpen]=useState(false);

  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/competitive/${ticker}`).then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed')).finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for Competitive Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Analyzing competitive position — market share, profitability edge, moat vs 100+ peers…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>Competitive: {err}</div>;
  if(!d)return null;
  const km=d.key_metrics||{};

  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:14,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(0)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:ratingColor(d.competitive_rating),letterSpacing:0.5}}>{d.competitive_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>vs {d.peer_count} {d.bucket} peers · {d.coverage.scored}/{d.coverage.total} signals</div>
        </div>
      </div>

      {(() => {
        // Eight category scores on one axis beats eight collapsed cards: the
        // shape is the competitive position, and the distance between the best
        // and worst factor is the finding. Four radial dials showed a 100th
        // percentile for being the largest of eight peers, which says least.
        const cats=[...d.tree.categories].filter(c=>c.score!=null).sort((a,b)=>(b.score!)-(a.score!));
        if(cats.length<3) return null;
        const best=cats[0], worst=cats[cats.length-1];
        const W=1000,H=34*cats.length+34,L=210,R=54;
        const x=(v:number)=>L+(v/100)*(W-L-R);
        return (
          <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:6,padding:'16px 18px',marginBottom:14}}>
            <div style={{fontFamily:'var(--font-mono)',fontSize:9,letterSpacing:2,color:'var(--gold)',marginBottom:4}}>COMPETITIVE PROFILE</div>
            <div style={{fontFamily:'var(--font-body)',fontSize:11,color:'var(--cocoa-dust)',marginBottom:12}}>
              Each dimension scored against {d.peer_count} {d.bucket} peers. The centre line is the peer median.
            </div>
            <svg viewBox={`0 0 ${W} ${H}`} style={{width:'100%',display:'block'}}>
              <line x1={x(50)} y1={14} x2={x(50)} y2={H-16} stroke="var(--border-2)" strokeDasharray="3 4"/>
              <text x={x(50)} y={10} fill="var(--cocoa)" fontSize="9" fontFamily="monospace" textAnchor="middle">peer median</text>
              {cats.map((c,i)=>{
                const y=30+i*34, sc=c.score as number;
                return (
                  <g key={c.id}>
                    <text x={L-12} y={y+4} fill="var(--latte)" fontSize="12" fontFamily="var(--font-body)" textAnchor="end">{c.label}</text>
                    <line x1={L} y1={y} x2={W-R} y2={y} stroke="var(--surface-3)" strokeWidth={6} strokeLinecap="round"/>
                    <line x1={x(Math.min(50,sc))} y1={y} x2={x(Math.max(50,sc))} y2={y}
                      stroke={heat(sc)} strokeWidth={6} strokeLinecap="round" opacity={0.55}/>
                    <circle cx={x(sc)} cy={y} r={6} fill={heat(sc)}/>
                    <text x={W-R+10} y={y+4} fill={heat(sc)} fontSize="13" fontFamily="monospace" fontWeight="700">{sc.toFixed(0)}</text>
                  </g>
                );
              })}
            </svg>
            <div style={{fontFamily:'var(--font-body)',fontSize:12,color:'var(--latte)',marginTop:10,lineHeight:1.55}}>
              Strongest on <b style={{color:'var(--gold)'}}>{best.label}</b> at {best.score!.toFixed(0)},
              weakest on <b style={{color:heat(worst.score)}}>{worst.label}</b> at {worst.score!.toFixed(0)}
              {(best.score!-worst.score!)>45
                ? ` — a ${(best.score!-worst.score!).toFixed(0)}-point spread, so the composite of ${d.score?.toFixed(0)} averages over a real disagreement rather than describing a consistent position.`
                : '.'}
            </div>
          </div>
        );
      })()}

      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontFamily:'var(--font-mono)',fontSize:9,letterSpacing:2,color:'var(--gold)'}}>{d.tree.categories.length} DIMENSIONS \u00b7 {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} SIGNALS</span>
        <button onClick={()=>{const v=!allOpen;setAllOpen(v);const m:Record<string,boolean>={};d.tree.categories.forEach(c=>m[c.id]=v);setExpanded(m);}}
          style={{background:'transparent',border:'1px solid var(--border-1)',color:'var(--cocoa-dust)',borderRadius:3,padding:'4px 10px',fontFamily:'var(--font-mono)',fontSize:10,cursor:'pointer'}}>{allOpen?'Collapse all':'Expand all'}</button>
      </div>
      <div style={{display:'flex',flexDirection:'column',gap:8}}>
        {d.tree.categories.map(cat=>{const open=expanded[cat.id];return (
          <div key={cat.id} style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:4,overflow:'hidden'}}>
            <div onClick={()=>setExpanded(p=>({...p,[cat.id]:!p[cat.id]}))}
              style={{display:'flex',alignItems:'center',gap:12,padding:'10px 14px',cursor:'pointer',borderLeft:`3px solid ${heat(cat.score)}`}}>
              <span style={{fontSize:11,color:'#7a7266',width:12}}>{open?'▾':'▸'}</span>
              <span style={{fontSize:13,fontWeight:600,fontFamily:'var(--font-body)',color:'var(--latte)',flex:1}}>{cat.label}</span>
              <span style={{fontSize:10,color:'#7a7266'}}>wt {cat.weight.toFixed(2)} · {cat.n_scored}/{cat.n_signals}</span>
              <span style={{fontSize:18,fontWeight:700,color:heat(cat.score),width:36,textAlign:'right'}}>{cat.score?.toFixed(0)??'—'}</span>
            </div>
            {open && (<div style={{padding:'4px 14px 12px 30px'}}>
              {cat.signals.map(s=>{const pending=s.status==='needs_source'||s.score==null;return (
                <div key={s.id} title={s.evidence} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0',borderBottom:'1px solid var(--border-1)',opacity:pending?0.5:1}}>
                  <span style={{fontSize:12,fontFamily:'var(--font-body)',color:'var(--latte)',flex:1}}>{s.label}</span>
                  <span style={{fontSize:12,color:'#9d8b7a',width:70,textAlign:'right'}}>{pending?'—':fmt(s.id,s.raw_value)}</span>
                  <div style={{width:80,height:6,background:'var(--surface-3)',borderRadius:2,overflow:'hidden'}}>{!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score)}}/>}</div>
                  <span style={{fontSize:11,fontWeight:600,color:pending?'#555':heat(s.score),width:26,textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
                </div>);})}
            </div>)}
          </div>);})}
      </div>
    </div>
  );
}
