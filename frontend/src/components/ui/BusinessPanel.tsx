import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; method?:string; }
interface Cat { id:string; label:string; weight:number; score:number|null; confidence:number; n_signals:number; n_scored:number; signals:Sig[]; }
interface BizData {
  ticker:string; available:boolean; score:number|null; moat_rating:string; confidence:number;
  coverage:{scored:number;total:number}; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string;
}
const heat=(s:number|null)=>s==null?'#2a2a2a':s>=75?'#0f6e56':s>=58?'#1d9e75':s>=42?'#8a7519':s>=25?'#a35a1d':'#7a2320';
const moatColor=(r:string)=>r.includes('Wide')?'#0f9d6e':r.includes('Narrow')?'#1d9e75':r.includes('Emerging')?'#c9a227':'#c0705a';

const fmtVal=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id.includes('economic_profit')) return '$'+(v/1e9).toFixed(1)+'B';
  if(id.includes('persistence')||id.includes('cycle')) return v.toFixed(1);
  if(id.includes('cash_conversion_ratio')||id.includes('turnover')||id.includes('leverage')||id.includes('productivity')||id.includes('resilience')||id.includes('rule_of_40')||id.includes('roic')||id.includes('consistency')||id.includes('stability')||id.includes('predictability')||id.includes('direction')||id.includes('absorption')||id.includes('margin_proxy')||id.includes('incremental_margin')) return v.toFixed(2);
  return (v*100).toFixed(1)+'%';
};

export default function BusinessPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<BizData|null>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({});
  const [allOpen,setAllOpen]=useState(true);
  useEffect(()=>{
    if(!ticker)return;
    setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/business/${ticker}`)
      .then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No business data');else{setD(x);
        const init:Record<string,boolean>={}; (x.tree?.categories||[]).forEach((c:Cat)=>init[c.id]=true); setExpanded(init);}})
      .catch(e=>setErr(e?.message||'Request failed'))
      .finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for Business Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Computing Business Intelligence — 76 moat & durability signals…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>Business: {err}</div>;
  if(!d)return null;

  const km=d.key_metrics;
  const radarCats=d.tree.categories.filter(c=>c.score!=null).slice(0,10);
  const N=radarCats.length;
  const cx=150,cy=150,R=105;
  const pt=(i:number,r:number)=>{const ang=(-Math.PI/2)+(2*Math.PI*i/N);return [cx+r*Math.cos(ang),cy+r*Math.sin(ang)];};
  const rings=[0.25,0.5,0.75,1.0];
  const dataPoly=radarCats.map((c,i)=>pt(i,R*(c.score!/100))).map(p=>p.join(',')).join(' ');

  const evidence:string[]=[];
  if((km.excess_return_spread??0)>0.1||(km.roic_wacc_spread??0)>0.1) evidence.push(`Persistent excess returns over cost of capital`);
  if((km.gross_margin_level??0)>0.5) evidence.push(`Strong pricing power (${((km.gross_margin_level??0)*100).toFixed(0)}% gross margin)`);
  if((km.recurring_revenue_ratio??0)>0.1) evidence.push(`Recurring revenue base`);
  if((km.reinvestment_quality??0)>0.08) evidence.push('Value-creating reinvestment');
  if(evidence.length===0) evidence.push('Moderate competitive position');

  const toggleAll=()=>{const v=!allOpen;setAllOpen(v);const m:Record<string,boolean>={};d.tree.categories.forEach(c=>m[c.id]=v);setExpanded(m);};

  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:20,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(1)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:moatColor(d.moat_rating),letterSpacing:1}}>{d.moat_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>{d.coverage.scored}/{d.coverage.total} durability signals · {(d.confidence*100).toFixed(0)}% confidence</div>
        </div>
        <button onClick={toggleAll} style={{marginLeft:'auto',background:'#181818',border:'1px solid #2a2a2a',color:'#9d8b7a',borderRadius:8,padding:'6px 14px',fontSize:11,cursor:'pointer'}}>
          {allOpen?'Collapse all':'Expand all'}</button>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'320px 1fr',gap:18,marginBottom:18}}>
        <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:16}}>
          <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:8}}>MOAT PROFILE · 10 dimensions</div>
          <svg viewBox="0 0 300 310" style={{width:'100%'}}>
            {rings.map((rr,ri)=><polygon key={ri} points={radarCats.map((_,i)=>pt(i,R*rr).join(',')).join(' ')} fill="none" stroke="#2a2a2a" strokeWidth="1"/>)}
            {radarCats.map((_,i)=>{const [x,y]=pt(i,R);return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="#2a2a2a" strokeWidth="1"/>;})}
            <polygon points={dataPoly} fill="rgba(15,157,110,0.25)" stroke="#0f9d6e" strokeWidth="2"/>
            {radarCats.map((c,i)=>{const [x,y]=pt(i,R*(c.score!/100));return <circle key={i} cx={x} cy={y} r="2.5" fill="#0f9d6e"/>;})}
            {radarCats.map((c,i)=>{const [x,y]=pt(i,R+16);const s=c.label.split(' ')[0].slice(0,9);
              return <text key={i} x={x} y={y} fontSize="7.5" fill="#9d8b7a" textAnchor="middle" dominantBaseline="middle">{s}</text>;})}
          </svg>
        </div>
        <div>
          <div style={{background:'#1a1512',border:'1px solid #3a2a1a',borderRadius:12,padding:'14px 16px',marginBottom:12}}>
            <div style={{fontSize:12,color:'#c9a227',letterSpacing:1,marginBottom:8,fontWeight:600}}>MOAT EVIDENCE</div>
            {evidence.map((e,i)=><div key={i} style={{fontSize:12,color:'#cdbfae',marginBottom:4}}>▸ {e}</div>)}
          </div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(120px,1fr))',gap:8}}>
            {[['Economic Profit',km.economic_profit!=null?'$'+(km.economic_profit/1e9).toFixed(0)+'B':'—'],
              ['ROIC',km.roic_current!=null?(km.roic_current*100).toFixed(0)+'%':'—'],
              ['Excess Return',km.excess_return_spread!=null?'+'+(km.excess_return_spread*100).toFixed(0)+'%':'—'],
              ['Gross Margin',km.gross_margin_level!=null?(km.gross_margin_level*100).toFixed(0)+'%':'—'],
              ['Recurring Rev',km.recurring_revenue_ratio!=null?(km.recurring_revenue_ratio*100).toFixed(0)+'%':'—'],
              ['Reinvest Quality',km.reinvestment_quality!=null?(km.reinvestment_quality*100).toFixed(0)+'%':'—']].map(([k,v])=>(
              <div key={k as string} style={{background:'#181818',border:'1px solid #2a2a2a',borderRadius:8,padding:'8px 10px'}}>
                <div style={{fontSize:9,color:'#9d8b7a'}}>{k}</div>
                <div style={{fontSize:15,fontWeight:600,color:'#daa520'}}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:8}}>10 BUSINESS DIMENSIONS · {d.coverage.total} SIGNALS</div>
      <div style={{display:'flex',flexDirection:'column',gap:8}}>
        {d.tree.categories.map(cat=>{
          const open=expanded[cat.id];
          return (
            <div key={cat.id} style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:10,overflow:'hidden'}}>
              <div onClick={()=>setExpanded(p=>({...p,[cat.id]:!p[cat.id]}))}
                style={{display:'flex',alignItems:'center',gap:12,padding:'10px 14px',cursor:'pointer',borderLeft:`4px solid ${heat(cat.score)}`}}>
                <span style={{fontSize:11,color:'#7a7266',width:12}}>{open?'▾':'▸'}</span>
                <span style={{fontSize:13,fontWeight:600,color:'#e8ddd0',flex:1}}>{cat.label}</span>
                <span style={{fontSize:10,color:'#7a7266'}}>wt {cat.weight.toFixed(2)} · {cat.n_scored}/{cat.n_signals}</span>
                <span style={{fontSize:18,fontWeight:700,color:heat(cat.score),width:36,textAlign:'right'}}>{cat.score?.toFixed(0)??'—'}</span>
              </div>
              {open && (
                <div style={{padding:'4px 14px 12px 30px'}}>
                  {cat.signals.map(s=>{
                    const pending=s.status==='needs_source'||s.score==null;
                    return (
                      <div key={s.id} title={s.evidence} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0',borderBottom:'1px solid #1e1e1e',opacity:pending?0.5:1}}>
                        <span style={{fontSize:12,color:'#cdbfae',flex:1}}>{s.label}</span>
                        <span style={{fontSize:12,color:'#9d8b7a',width:80,textAlign:'right'}}>{pending?'pending':fmtVal(s.id,s.raw_value)}</span>
                        <div style={{width:90,height:6,background:'#242424',borderRadius:3,overflow:'hidden'}}>
                          {!pending && <div style={{height:'100%',width:`${s.score}%`,background:heat(s.score)}}/>}
                        </div>
                        <span style={{fontSize:11,fontWeight:600,color:pending?'#555':heat(s.score),width:26,textAlign:'right'}}>{pending?'—':s.score!.toFixed(0)}</span>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
