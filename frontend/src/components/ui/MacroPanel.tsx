import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';
interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface MData { ticker:string; available:boolean; score:number|null; macro_rating:string; coverage:{scored:number;total:number};
  tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string; }
const heat=(s:number|null)=>s==null?'var(--border-2)':s>=70?'var(--gold)':s>=50?'var(--caramel)':s>=30?'#c9762f':'var(--bear)';
const rc=(r:string)=>r==='Insulated'?'var(--gold)':r==='Resilient'?'var(--gold)':r==='Balanced'?'var(--caramel)':r==='Exposed'?'#c9762f':'var(--bear)';
const fmt=(v:number|null):string=>v==null?'—':(v>=0?'+':'')+v.toFixed(2);
const BetaBar=({label,val,desc}:{label:string;val:number|null;desc:string})=>{
  const v=val??0; const mag=Math.min(1,Math.abs(v)); const pos=v>=0;
  // Magnitude carries the meaning here, not direction: a beta of -0.3 to the
  // dollar is as much an exposure as +0.3. Gold for the strong ones, caramel for
  // moderate, muted surface for anything close to no exposure at all.
  const col=Math.abs(v)<0.15?'var(--border-2)':Math.abs(v)<0.4?'var(--caramel)':'var(--gold)';
  return (
    <div title={desc} style={{display:'flex',alignItems:'center',gap:10,padding:'5px 0'}}>
      <span style={{fontSize:11,color:'#cdbfae',width:120}}>{label}</span>
      <div style={{flex:1,height:18,position:'relative',background:'var(--surface-3)',borderRadius:2}}>
        <div style={{position:'absolute',left:'50%',top:0,bottom:0,width:1,background:'#3a3a3a'}}/>
        <div style={{position:'absolute',top:3,bottom:3,borderRadius:3,background:col,
          left:pos?'50%':`${50-mag*50}%`,width:`${mag*50}%`}}/>
      </div>
      <span style={{fontSize:11,fontWeight:600,color:col,width:44,textAlign:'right'}}>{fmt(val)}</span>
    </div>
  );
};
export default function MacroPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<MData|null>(null);
  const [loading,setLoading]=useState(false); const [err,setErr]=useState('');
  // No collapse state — the cards always show their signals, so nothing expands.
  useEffect(()=>{ if(!ticker)return; setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/macro/${ticker}`).then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No data');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed')).finally(()=>setLoading(false));
  },[ticker]);
  if(!ticker)return <div style={{fontFamily:'var(--font-body)',color:'var(--cocoa-dust)',padding:24}}>Enter a ticker for Macro Sensitivity.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Analyzing macro exposures — rates, dollar, inflation, cycle, factors…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>Macro: {err}</div>;
  if(!d)return null;
  const km=d.key_metrics||{};
  const betas=[['Rates (TLT)',km.rate_beta,'Sensitivity to long-bond/rate moves'],
    ['Dollar (UUP)',km.dollar_beta,'Strong-dollar impact (- = multinational)'],
    ['Inflation (Gold)',km.inflation_hedge,'Gold correlation (+ = inflation hedge)'],
    ['Oil (USO)',km.oil_beta,'Energy/oil-price sensitivity'],
    ['Market (SPY)',km.market_beta,'Systematic market risk'],
    ['Credit (HYG)',km.credit_beta,'Risk-on/off sensitivity'],
    ['Value tilt',km.value_tilt,'Value-factor loading'],
    ['Momentum tilt',km.momentum_tilt,'Momentum-factor loading']] as [string,number|null,string][];
  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:14,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(0)??'—'}</span>
          <span style={{fontFamily:'var(--font-mono)',fontSize:16,color:'var(--cocoa)'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:rc(d.macro_rating),letterSpacing:0.5}}>{d.macro_rating}</div>
          <div style={{fontFamily:'var(--font-body)',fontSize:11,color:'var(--cocoa-dust)',marginTop:2}}>{d.coverage.scored}/{d.coverage.total} exposures measured</div>
        </div>
      </div>
      <div style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',borderRadius:4,padding:'12px 16px',marginBottom:14}}>
        <div style={{display:'flex',justifyContent:'space-between',fontFamily:'var(--font-mono)',fontSize:9,color:'var(--gold)',letterSpacing:2,marginBottom:8}}>
          <span>MACRO FACTOR EXPOSURE (β)</span><span style={{fontSize:9}}>← negative · positive →</span></div>
        {betas.map(([l,v,desc])=><BetaBar key={l} label={l} val={v} desc={desc}/>)}
        <div style={{marginTop:8,paddingTop:8,borderTop:'1px solid var(--border-1)',fontFamily:'var(--font-body)',fontSize:10.5,color:'var(--cocoa)'}}>
          {/* "Defensiveness -86%" is a double negative a reader has to unpick. The
              number is market beta minus one, so state it as sensitivity in the
              direction it actually runs. */}
          Macro resilience {km.macro_resilience!=null?(km.macro_resilience*100).toFixed(0)+'%':'—'}
          {km.defensiveness!=null && (km.defensiveness < 0
            ? ` · ${Math.abs(km.defensiveness*100).toFixed(0)}% more market-sensitive than the index`
            : ` · ${(km.defensiveness*100).toFixed(0)}% less market-sensitive than the index`)}
        </div>
      </div>
      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontFamily:'var(--font-mono)',fontSize:9,letterSpacing:2,color:'var(--gold)'}}>{d.tree.categories.length} REGIMES · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} SIGNALS</span>
      </div>
      {/* Eight regimes are peers rather than a ranked list — you compare them, so
          they sit as cards in a grid instead of stacked full-width rows with the
          label at one edge and the number at the other. Each card carries its own
          signals at their natural width and the tab fits on one screen. */}
      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fit,minmax(420px,1fr))',gap:10,alignItems:'start'}}>
        {d.tree.categories.map(cat=>(
          <div key={cat.id} style={{background:'var(--surface-2)',border:'1px solid var(--border-1)',
            borderTop:`2px solid ${heat(cat.score)}`,borderRadius:4,padding:'12px 14px'}}>
            <div style={{display:'flex',alignItems:'baseline',justifyContent:'space-between',gap:8}}>
              <span style={{fontFamily:'var(--font-body)',fontSize:12.5,fontWeight:600,color:'var(--latte)'}}>{cat.label}</span>
              <span style={{fontFamily:'var(--font-mono)',fontSize:22,fontWeight:700,color:heat(cat.score),lineHeight:1}}>{cat.score?.toFixed(0)??'—'}</span>
            </div>
            <div style={{fontFamily:'var(--font-mono)',fontSize:9,color:'var(--cocoa)',marginBottom:10}}>
              weight {cat.weight.toFixed(2)} · {cat.n_scored} of {cat.n_signals} measured
            </div>
            {cat.signals.map(s=>{const pending=s.status==='needs_source'||s.score==null;return (
              <div key={s.id} title={s.evidence} style={{display:'flex',alignItems:'baseline',gap:8,padding:'4px 0',borderTop:'1px solid var(--border-1)',opacity:pending?0.45:1}}>
                <span style={{fontFamily:'var(--font-body)',fontSize:11.5,color:'var(--latte)',flex:1}}>{s.label}</span>
                <span style={{fontFamily:'var(--font-mono)',fontSize:11.5,color:'var(--cocoa-dust)'}}>{pending?'—':fmt(s.raw_value)}</span>
                <span style={{fontFamily:'var(--font-mono)',fontSize:11,fontWeight:600,width:24,textAlign:'right',color:pending?'var(--cocoa)':heat(s.score)}}>{pending?'—':s.score!.toFixed(0)}</span>
              </div>);})}
          </div>))}
      </div>
    </div>
  );
}
