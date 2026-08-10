import { useState } from 'react';

interface Bridge { d:any; }

const CATCOL=(fam:string)=>fam==='dcf'?'var(--gold)':fam==='intrinsic'?'var(--caramel)':'#c9762f';
const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';

export default function ValuationBridge({ d }:Bridge){
  const [hover,setHover]=useState<string>('');
  const price=d.current_price||0;
  const vr=d.value_range?.methods||{};
  const ma=d.model_agreement||{};
  const km=d.key_metrics||{};
  const rating=d.valuation_rating||'';
  const isOver=/Overvalued/i.test(rating);
  const ratingCol=/Undervalued/i.test(rating)?'var(--gold)':isOver?'var(--bear)':'var(--caramel)';

  // Build method list from real fair-values, sorted high→low.
  const methods=[
    {n:'DCF bull case',v:vr.dcf_bull,fam:'dcf'},
    {n:'Residual income',v:vr.residual_income,fam:'intrinsic'},
    {n:'EPV (earnings power)',v:vr.epv,fam:'intrinsic'},
    {n:'DCF base case',v:vr.dcf_base,fam:'dcf'},
    {n:'DDM (dividend)',v:vr.ddm,fam:'intrinsic'},
    {n:'Graham number',v:vr.graham,fam:'asset'},
    {n:'DCF bear case',v:vr.dcf_bear,fam:'dcf'},
    {n:'NAV (book value)',v:vr.nav,fam:'asset'},
  ].filter(m=>m.v!=null).sort((a,b)=>b.v-a.v);

  if(!methods.length||!price) return null;

  const maxV=Math.max(price*1.15, ...methods.map(m=>m.v))*1.05;
  const xof=(v:number)=>Math.max(2,Math.min(96,(v/maxV)*100));
  const pricePos=xof(price);

  const agree=ma.agreement_score!=null?Math.round(ma.agreement_score*100):null;
  const consensus=ma.consensus_overvalued!=null?Math.round(ma.consensus_overvalued*100):null;
  const disp=ma.dispersion_cv!=null?ma.dispersion_cv.toFixed(2):null;
  const impliedGrowth=km.reverse_dcf_implied_growth!=null?(km.reverse_dcf_implied_growth*100).toFixed(1):null;

  const bear=vr.dcf_bear, bull=vr.dcf_bull;

  // Category verdict spread — diverging from 50.
  const cats=[...d.tree.categories].filter((c:any)=>c.score!=null).sort((a:any,b:any)=>b.score-a.score);

  return (
    <div style={{marginBottom:18}}>

      {/* ── VERDICT + AGREEMENT ── */}
      <div style={{display:'flex',gap:14,alignItems:'stretch',marginBottom:14,flexWrap:'wrap'}}>
        <div style={{flex:2,minWidth:280,background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'18px 20px'}}>
          <div style={{fontSize:11,letterSpacing:2,color:'#c9762f',marginBottom:6}}>THE VERDICT</div>
          <div style={{fontSize:26,fontWeight:600,color:ratingCol,lineHeight:1}}>{rating}</div>
          <div style={{fontSize:12,color:'var(--cocoa-dust)',marginTop:8,lineHeight:1.5}}>
            {methods.length} intrinsic methods place fair value {isOver?'below':'around'} the ${price.toFixed(0)} price.
            {impliedGrowth&&<> The market is pricing in <span style={{color:'var(--gold)'}}>{impliedGrowth}% implied growth</span>{km.pe_vs_history!=null&&km.pe_vs_history>1?' — above its own history':''}.</>}
          </div>
        </div>
        {agree!=null&&(
          <div style={{flex:1,minWidth:150,background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'18px 20px',display:'flex',flexDirection:'column',justifyContent:'center'}}>
            <div style={{fontSize:11,letterSpacing:1,color:'var(--cocoa)',marginBottom:4}}>MODEL AGREEMENT</div>
            <div style={{fontSize:30,fontWeight:600,color:'var(--gold)',lineHeight:1}}>{agree}%</div>
            {disp&&<div style={{fontSize:11,color:'var(--cocoa-dust)',marginTop:6}}>{agree>=70?'tight cluster':'mixed'} · dispersion {disp}</div>}
            {consensus!=null&&<div style={{fontSize:11,color:consensus>=60?'var(--bear)':'var(--gold)',marginTop:6}}>{consensus}% say overvalued</div>}
          </div>
        )}
      </div>

      {/* ── FAIR-VALUE BRIDGE ── */}
      <div style={{background:'var(--surface-2)',border:'1px solid var(--border-2)',borderRadius:14,padding:'22px 22px 14px'}}>
        <div style={{display:'flex',justifyContent:'space-between',fontSize:10,letterSpacing:1,color:'var(--cocoa)',marginBottom:4}}>
          <span>FAIR-VALUE BRIDGE · {methods.length} methods vs price</span>
          <span style={{color:'#c9762f'}}>hover a method</span>
        </div>
        <div style={{position:'relative',marginTop:26}}>
          {/* DCF scenario band */}
          {bear!=null&&bull!=null&&(
            <div style={{position:'absolute',top:0,bottom:26,left:`${xof(bear)}%`,width:`${xof(bull)-xof(bear)}%`,
              background:'rgba(201,118,47,0.10)',borderLeft:'1px dashed rgba(201,118,47,0.4)',borderRight:'1px dashed rgba(201,118,47,0.4)',pointerEvents:'none'}}/>
          )}
          {/* price line */}
          <div style={{position:'absolute',left:`${pricePos}%`,top:-18,bottom:8,width:2,background:'var(--gold)',zIndex:3}}>
            <div style={{position:'absolute',top:-16,left:'50%',transform:'translateX(-50%)',background:'var(--gold)',color:'var(--surface-0)',fontSize:10,fontWeight:700,padding:'2px 7px',borderRadius:4,whiteSpace:'nowrap'}}>PRICE ${price.toFixed(0)}</div>
          </div>
          {/* method rows */}
          {methods.map(m=>{
            const x=xof(m.v); const col=CATCOL(m.fam); const gap=(m.v-price)/price*100;
            const on=hover===m.n;
            const barL=Math.min(x,pricePos), barR=Math.max(x,pricePos);
            return (
              <div key={m.n} onMouseEnter={()=>setHover(m.n)} onMouseLeave={()=>setHover('')}
                style={{position:'relative',height:30,cursor:'pointer'}}>
                <div style={{position:'absolute',left:`${barL}%`,width:`${barR-barL}%`,top:14,height:2,background:col,opacity:0.3}}/>
                <div style={{position:'absolute',left:`${x}%`,top:15,width:11,height:11,borderRadius:'50%',background:col,
                  transform:`translate(-50%,-50%) scale(${on?1.5:1})`,border:'2px solid var(--surface-2)',zIndex:2,transition:'transform .15s'}}/>
                <div style={{position:'absolute',left:`${x}%`,top:-2,transform:'translateX(-50%)',fontSize:10,color:col,fontWeight:600,whiteSpace:'nowrap'}}>${m.v.toFixed(0)}</div>
                <div style={{position:'absolute',left:`calc(${x}% + 12px)`,top:11,fontSize:11,color:'var(--latte)',whiteSpace:'nowrap'}}>
                  {m.n}{on&&<span style={{color:gap<0?'var(--bear)':'var(--gold)',marginLeft:8}}>{gap>=0?'+':''}{gap.toFixed(0)}% {gap<0?'overvalued':'undervalued'}</span>}
                </div>
              </div>
            );
          })}
        </div>
        <div style={{fontSize:10,color:'var(--cocoa)',marginTop:8,borderTop:'1px solid var(--border-1)',paddingTop:8}}>
          shaded band = DCF bear→bull range · each dot = one method's fair value · gold line = today's price
        </div>
      </div>

      {/* ── CATEGORY VERDICT SPREAD ── */}
      <div style={{fontSize:10,letterSpacing:2,color:'var(--gold)',margin:'18px 0 8px'}}>VALUATION LENSES · ranked cheap → expensive</div>
      <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:12,padding:'14px 20px'}}>
        {cats.map((c:any,i:number)=>{
          const dev=(c.score-50)/50; const pct=Math.abs(dev)*46;
          return (
            <div key={c.id} style={{display:'grid',gridTemplateColumns:'150px 1fr 34px',alignItems:'center',gap:12,padding:'6px 0',
              borderBottom:i<cats.length-1?'1px solid var(--border-1)':'none'}}>
              <span style={{fontSize:12,color:'var(--latte)',textAlign:'right'}}>{c.label.replace(' Valuation','').replace(' Value','')}</span>
              <div style={{position:'relative',height:14}}>
                <div style={{position:'absolute',left:'50%',top:0,bottom:0,width:1,background:'var(--border-2)'}}/>
                <div style={{position:'absolute',top:3,bottom:3,background:heat(c.score),borderRadius:2,
                  ...(dev>=0?{left:'50%',width:`${pct}%`}:{right:'50%',width:`${pct}%`})}}/>
              </div>
              <span style={{fontSize:13,fontWeight:600,color:heat(c.score),textAlign:'right'}}>{c.score.toFixed(0)}</span>
            </div>
          );
        })}
        <div style={{display:'flex',justifyContent:'space-between',fontSize:9,color:'var(--cocoa)',marginTop:8,paddingTop:6,borderTop:'1px solid var(--border-1)'}}>
          <span>◄ expensive</span><span>cheap ►</span>
        </div>
      </div>
    </div>
  );
}
