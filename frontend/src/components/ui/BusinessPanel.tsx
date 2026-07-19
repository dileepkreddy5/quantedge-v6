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

export default function BusinessPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<BizData|null>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState('');
  useEffect(()=>{
    if(!ticker)return;
    setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/business/${ticker}`)
      .then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No business data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed'))
      .finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for Business Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Computing Business Intelligence — moat strength, pricing power, durability…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>Business: {err}</div>;
  if(!d)return null;

  const km=d.key_metrics;
  const radarCats=d.tree.categories.filter(c=>c.score!=null).slice(0,6);
  const N=radarCats.length;
  const cx=150,cy=150,R=110;
  const pt=(i:number,r:number)=>{
    const ang=(-Math.PI/2)+(2*Math.PI*i/N);
    return [cx+r*Math.cos(ang), cy+r*Math.sin(ang)];
  };
  const rings=[0.25,0.5,0.75,1.0];
  const dataPoly=radarCats.map((c,i)=>pt(i,R*(c.score!/100))).map(p=>p.join(',')).join(' ');

  const evidence:string[]=[];
  if((km.excess_return_spread??0)>0.1) evidence.push(`Persistent excess returns (+${((km.excess_return_spread??0)*100).toFixed(0)}% over WACC)`);
  if((km.gross_margin_level??0)>0.5) evidence.push(`Strong pricing power (${((km.gross_margin_level??0)*100).toFixed(0)}% gross margin)`);
  if((km.gross_margin_stability??0)>0.8) evidence.push('Stable margins (durable advantage)');
  if((km.recurring_revenue_ratio??0)>0.1) evidence.push(`Recurring revenue base (${((km.recurring_revenue_ratio??0)*100).toFixed(0)}%)`);
  if((km.reinvestment_quality??0)>0.08) evidence.push('Value-creating reinvestment');
  if(evidence.length===0) evidence.push('Moderate competitive position');

  const fmtP=(v:number|null)=>v==null?'—':(v*100).toFixed(1)+'%';
  const fmtN=(v:number|null)=>v==null?'—':v.toFixed(2);

  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:20,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(1)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:moatColor(d.moat_rating),letterSpacing:1}}>{d.moat_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>Coverage {d.coverage.scored}/{d.coverage.total} durability signals</div>
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'320px 1fr',gap:18,marginBottom:18}}>
        <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:16}}>
          <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:8}}>MOAT PROFILE</div>
          <svg viewBox="0 0 300 320" style={{width:'100%'}}>
            {rings.map((rr,ri)=>(
              <polygon key={ri} points={radarCats.map((_,i)=>pt(i,R*rr).join(',')).join(' ')} fill="none" stroke="#2a2a2a" strokeWidth="1"/>
            ))}
            {radarCats.map((_,i)=>{const [x,y]=pt(i,R);return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="#2a2a2a" strokeWidth="1"/>;})}
            <polygon points={dataPoly} fill="rgba(29,158,117,0.25)" stroke="#1d9e75" strokeWidth="2"/>
            {radarCats.map((c,i)=>{const [x,y]=pt(i,R*(c.score!/100));return <circle key={i} cx={x} cy={y} r="3" fill="#1d9e75"/>;})}
            {radarCats.map((c,i)=>{
              const [x,y]=pt(i,R+22);
              const short=c.label.replace(' & Operating Leverage','').replace('Revenue Quality & Recurrence','Recurring Rev').replace(' Economics','').replace('Moat Strength','Moat').replace('Pricing Power','Pricing');
              return <text key={i} x={x} y={y} fontSize="8" fill="#9d8b7a" textAnchor="middle" dominantBaseline="middle">{short}</text>;
            })}
            {radarCats.map((c,i)=>{const [x,y]=pt(i,R*(c.score!/100));return <text key={'s'+i} x={x} y={y-7} fontSize="9" fill="#daa520" textAnchor="middle" fontWeight="700">{c.score!.toFixed(0)}</text>;})}
          </svg>
        </div>

        <div>
          <div style={{background:'#1a1512',border:'1px solid #3a2a1a',borderRadius:12,padding:'14px 16px',marginBottom:12}}>
            <div style={{fontSize:12,color:'#c9a227',letterSpacing:1,marginBottom:8,fontWeight:600}}>MOAT EVIDENCE</div>
            {evidence.map((e,i)=><div key={i} style={{fontSize:12,color:'#cdbfae',marginBottom:4}}>▸ {e}</div>)}
          </div>
          <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(140px,1fr))',gap:10}}>
            {[['Excess Return',fmtP(km.excess_return_spread)],['Gross Margin',fmtP(km.gross_margin_level)],
              ['Margin Stability',fmtN(km.gross_margin_stability)],['Recurring Rev',fmtP(km.recurring_revenue_ratio)],
              ['Operating Leverage',fmtN(km.operating_leverage)],['Reinvest Quality',fmtP(km.reinvestment_quality)]].map(([k,v])=>(
              <div key={k as string} style={{background:'#181818',border:'1px solid #2a2a2a',borderRadius:8,padding:'10px 12px'}}>
                <div style={{fontSize:10,color:'#9d8b7a',marginBottom:3}}>{k}</div>
                <div style={{fontSize:16,fontWeight:600,color:'#daa520'}}>{v}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:8}}>8 BUSINESS DIMENSIONS</div>
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(180px,1fr))',gap:8}}>
        {d.tree.categories.map(cat=>{
          const pending=cat.score==null;
          return (
            <div key={cat.id} style={{background:pending?'#181818':heat(cat.score),borderRadius:8,padding:'10px 12px',opacity:pending?0.6:1,border:pending?'1px dashed #3a3a3a':'none'}}>
              <div style={{display:'flex',justifyContent:'space-between',alignItems:'center'}}>
                <span style={{fontSize:11,fontWeight:600,color:'#fff'}}>{cat.label}</span>
                <span style={{fontSize:15,fontWeight:700,color:'#fff'}}>{pending?'—':cat.score!.toFixed(0)}</span>
              </div>
              <div style={{fontSize:9,color:'#fff',opacity:0.6,marginTop:2}}>
                {pending?'pending · AI Analyst layer':`wt ${cat.weight.toFixed(2)} · ${cat.n_scored}/${cat.n_signals}`}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
