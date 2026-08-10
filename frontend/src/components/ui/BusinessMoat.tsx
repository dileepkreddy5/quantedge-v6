import { useState } from 'react';

const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';

const PILLARS=[
  {id:'economic',label:'Economic moat',desc:'advantage visible in the economics',
   dims:['moat_strength','pricing_power','revenue_quality','unit_economics']},
  {id:'structural',label:'Structural moat',desc:'advantage from position, scale, IP',
   dims:['competitive_position','scale_leverage','intangible_moat']},
  {id:'durability',label:'Durability',desc:'how sustainable the advantage is',
   dims:['capital_allocation','growth_quality','business_risk']},
];

// one real driver metric per dimension
const DRIVER:Record<string,(km:any)=>string|null>={
  moat_strength:km=>km.roic_current!=null?`ROIC ${(km.roic_current*100).toFixed(0)}%`:null,
  pricing_power:km=>km.gross_margin_level!=null?`${(km.gross_margin_level*100).toFixed(0)}% gross margin`:null,
  revenue_quality:km=>km.recurring_revenue_ratio!=null?`${(km.recurring_revenue_ratio*100).toFixed(0)}% recurring`:null,
  unit_economics:km=>km.capital_intensity!=null?`capex ${(km.capital_intensity*100).toFixed(0)}% of rev`:null,
  competitive_position:km=>km.excess_return_spread!=null?`+${(km.excess_return_spread*100).toFixed(0)}% excess return`:null,
  scale_leverage:km=>km.operating_leverage!=null?`op leverage ${km.operating_leverage.toFixed(1)}×`:null,
  capital_allocation:km=>km.reinvestment_quality!=null?`reinvest quality ${(km.reinvestment_quality*100).toFixed(0)}%`:null,
  growth_quality:km=>km.revenue_consistency!=null?`consistency ${(km.revenue_consistency*100).toFixed(0)}%`:null,
  business_risk:km=>km.roe_stability!=null?`ROE stability ${(km.roe_stability*100).toFixed(0)}%`:null,
  intangible_moat:km=>km.gross_margin_stability!=null?`margin stability ${(km.gross_margin_stability*100).toFixed(0)}%`:null,
};

export default function BusinessMoat({ d }:{ d:any }){
  const [hover,setHover]=useState<string>('');
  const cats:Record<string,any>={};
  (d.tree?.categories||[]).forEach((c:any)=>cats[c.id]=c);
  const km=d.key_metrics||{};
  const rating=d.moat_rating||'';
  const ratingCol=/Wide/i.test(rating)?'var(--gold)':/Narrow/i.test(rating)?'var(--caramel)':/No Moat|None/i.test(rating)?'var(--bear)':'#c9762f';

  // radar over all 10 dims
  const order=['moat_strength','pricing_power','revenue_quality','unit_economics','competitive_position','capital_allocation','scale_leverage','growth_quality','business_risk','intangible_moat'];
  const rdims=order.map(id=>cats[id]).filter(Boolean);
  const N=rdims.length,R=108,cx=140,cy=140;
  const pt=(i:number,f:number)=>{const a=(Math.PI*2*i/N)-Math.PI/2;return [cx+Math.cos(a)*R*f,cy+Math.sin(a)*R*f];};
  const rings=[0.25,0.5,0.75,1].map(f=>rdims.map((_,i)=>pt(i,f).join(',')).join(' '));
  const poly=rdims.map((c,i)=>pt(i,(c.score||0)/100).join(',')).join(' ');

  // strongest / weakest for the driver readout
  const scored=[...rdims].filter(c=>c.score!=null).sort((a,b)=>b.score-a.score);
  const top1=scored[0], weak1=scored[scored.length-1];

  return (
    <div style={{marginBottom:18}}>
      {/* TOP: radar + verdict + drivers */}
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr',gap:16,marginBottom:16}}>
        <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:18,display:'flex',alignItems:'center',justifyContent:'center'}}>
          <svg width="100%" viewBox="0 0 280 280" style={{maxWidth:280,display:'block'}}>
            {rings.map((r,i)=><polygon key={i} points={r} fill="none" stroke="var(--surface-4)" strokeWidth="1"/>)}
            {rdims.map((_,i)=>{const[x,y]=pt(i,1);return <line key={i} x1={cx} y1={cy} x2={x} y2={y} stroke="var(--surface-4)" strokeWidth="0.5"/>;})}
            <polygon points={poly} fill="rgba(218,165,32,0.15)" stroke="var(--gold)" strokeWidth="2"/>
            {rdims.map((c,i)=>{const[x,y]=pt(i,(c.score||0)/100);return <circle key={i} cx={x} cy={y} r="3.5" fill={heat(c.score)}/>;})}
          </svg>
        </div>
        <div style={{display:'flex',flexDirection:'column',gap:12}}>
          <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'18px 20px',flex:1}}>
            <div style={{fontSize:11,letterSpacing:2,color:'#c9762f',marginBottom:6}}>MOAT VERDICT</div>
            <div style={{fontSize:26,fontWeight:500,color:ratingCol,lineHeight:1}}>{rating||'—'}</div>
            <div style={{fontSize:12,color:'var(--cocoa-dust)',marginTop:8,lineHeight:1.5}}>
              Durable advantage scored across {rdims.length} dimensions.
              {top1&&<> Strongest in <span style={{color:'var(--gold)'}}>{top1.label.replace(' & Durability','').replace(' & Recurrence','')}</span>.</>}
            </div>
          </div>
          <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'16px 20px'}}>
            <div style={{fontSize:10,letterSpacing:1,color:'var(--cocoa)',marginBottom:10}}>WHAT BUILDS THE MOAT</div>
            <div style={{display:'flex',flexDirection:'column',gap:7,fontSize:12}}>
              {top1&&<div style={{display:'flex',justifyContent:'space-between'}}><span style={{color:'var(--bull)'}}>▲ {top1.label.replace(' & Durability','').replace(' & Recurrence','').replace(' Allocation','')}</span><span style={{color:'var(--latte)'}}>{DRIVER[top1.id]?.(km)||top1.score.toFixed(0)}</span></div>}
              {scored[1]&&<div style={{display:'flex',justifyContent:'space-between'}}><span style={{color:'var(--bull)'}}>▲ {scored[1].label.replace(' & Durability','').replace(' & Recurrence','').replace(' Allocation','')}</span><span style={{color:'var(--latte)'}}>{DRIVER[scored[1].id]?.(km)||scored[1].score.toFixed(0)}</span></div>}
              {weak1&&<div style={{display:'flex',justifyContent:'space-between'}}><span style={{color:weak1.score<50?'var(--bear)':'#c9762f'}}>▼ {weak1.label.replace(' & Operating Leverage','').replace(' & Durability','')}</span><span style={{color:'var(--latte)'}}>weakest · {weak1.score.toFixed(0)}</span></div>}
            </div>
          </div>
        </div>
      </div>

      {/* MOAT PILLARS — grouped, weight-sized, strength-colored */}
      <div style={{fontSize:10,color:'var(--gold)',letterSpacing:2,margin:'6px 0 10px'}}>MOAT PILLARS · dimensions grouped by the source of advantage</div>
      <div style={{display:'grid',gridTemplateColumns:'1fr 1fr 1fr',gap:12}}>
        {PILLARS.map(p=>{
          const pd=p.dims.map(id=>cats[id]).filter(Boolean);
          const avg=pd.length?pd.reduce((a,c)=>a+(c.score||0),0)/pd.length:0;
          const pcol=heat(avg);
          return (
            <div key={p.id} style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderTop:`3px solid ${pcol}`,borderRadius:'0 0 12px 12px',padding:16}}>
              <div style={{fontSize:12,color:pcol,fontWeight:600}}>{p.label}</div>
              <div style={{fontSize:10,color:'var(--cocoa)',marginBottom:12}}>avg {avg.toFixed(0)} · {p.desc}</div>
              <div style={{display:'flex',flexDirection:'column',gap:8}}>
                {pd.sort((a,b)=>b.score-a.score).map(c=>{
                  const drv=DRIVER[c.id]?.(km);
                  const on=hover===c.id;
                  return (
                    <div key={c.id} onMouseEnter={()=>setHover(c.id)} onMouseLeave={()=>setHover('')}
                      style={{background:on?'var(--surface-3)':'transparent',borderRadius:8,padding:'6px 8px',cursor:'default',transition:'background .15s'}}>
                      <div style={{display:'flex',justifyContent:'space-between',alignItems:'baseline'}}>
                        <span style={{fontSize:11,color:'var(--latte)'}}>{c.label.replace(' & Durability','').replace(' & Recurrence','').replace(' & Operating Leverage','').replace(' & Capital Allocation','').replace(' Position','')}</span>
                        <span style={{fontSize:16,fontWeight:600,color:heat(c.score)}}>{c.score?.toFixed(0)}</span>
                      </div>
                      <div style={{height:4,background:'var(--surface-4)',borderRadius:2,marginTop:4,overflow:'hidden'}}>
                        <div style={{height:'100%',width:`${c.score}%`,background:heat(c.score)}}/>
                      </div>
                      {drv&&<div style={{fontSize:9,color:'var(--cocoa)',marginTop:3}}>▸ {drv}</div>}
                    </div>
                  );
                })}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
