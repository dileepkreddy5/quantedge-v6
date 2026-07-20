import { useEffect, useState } from 'react';
import { api } from '../../auth/authStore';

interface Sig { id:string; label:string; weight:number; status:string; evidence:string; raw_value:number|null; score:number|null; }
interface Cat { id:string; label:string; weight:number; score:number|null; n_signals:number; n_scored:number; signals:Sig[]; }
interface BriefItem { headline:string; sentiment:string; publisher:string; date:string; url:string; }
interface Headline { title:string; sentiment:string; reason:string; publisher:string; date:string; url:string; }
interface NewsData {
  ticker:string; available:boolean; score:number|null; news_rating:string; confidence:number;
  coverage:{scored:number;total:number}; article_count:number;
  sentiment_dist:{positive:number;neutral:number;negative:number};
  brief:BriefItem[]; recent_headlines:Headline[]; tree:{categories:Cat[]}; key_metrics:Record<string,number|null>; reason?:string;
}
const heat=(s:number|null)=>s==null?'#2a2a2a':s>=75?'#0f6e56':s>=58?'#1d9e75':s>=42?'#8a7519':s>=25?'#a35a1d':'#7a2320';
const ratingColor=(r:string)=>r.includes('Very Positive')?'#0f9d6e':r.includes('Positive')?'#1d9e75':r.includes('Mixed')?'#c9a227':r.includes('Negative')?'#c0705a':'#7a2320';
const sentDot=(s:string)=>s==='positive'?'#0f9d6e':s==='negative'?'#c0705a':'#9d8b7a';

const fmtVal=(id:string,v:number|null):string=>{
  if(v==null) return '—';
  if(id.includes('count')||id.includes('publishers')||id.includes('topics')||id.includes('diversity')||id.includes('reasoning')||id.includes('flag')||id.includes('spike')) return Number.isInteger(v)?v.toString():v.toFixed(1);
  if(id.includes('hours')) return v.toFixed(0)+'h';
  if(id.includes('ratio')||id.includes('velocity')||id.includes('balance')||id.includes('severity')||id.includes('sentiment')||id.includes('mean')) return v.toFixed(2);
  return (v*100).toFixed(0)+'%';
};

export default function NewsPanel({ ticker }:{ ticker:string }){
  const [d,setD]=useState<NewsData|null>(null);
  const [loading,setLoading]=useState(false);
  const [err,setErr]=useState('');
  const [expanded,setExpanded]=useState<Record<string,boolean>>({});
  const [showAllSignals,setShowAllSignals]=useState(false);

  useEffect(()=>{
    if(!ticker)return;
    setLoading(true);setErr('');setD(null);
    api.get(`/api/v6/news/${ticker}`)
      .then(r=>{const x=r.data?.data;if(!x?.available)setErr(x?.reason||'No news coverage');else setD(x);})
      .catch(e=>setErr(e?.message||'Request failed'))
      .finally(()=>setLoading(false));
  },[ticker]);

  if(!ticker)return <div style={{color:'#9d8b7a',padding:24}}>Enter a ticker for News Intelligence.</div>;
  if(loading)return <div style={{color:'#daa520',padding:24}}>Analyzing news — sentiment, events, and 47 signals across recent coverage…</div>;
  if(err)return <div style={{color:'#c0705a',padding:24}}>News: {err}</div>;
  if(!d)return null;

  const sd=d.sentiment_dist||{positive:0,neutral:0,negative:0};
  const total=sd.positive+sd.neutral+sd.negative||1;
  const km=d.key_metrics||{};

  return (
    <div style={{padding:'8px 4px',color:'#e8ddd0'}}>
      <div style={{display:'flex',alignItems:'center',gap:24,marginBottom:18,flexWrap:'wrap'}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10}}>
          <span style={{fontSize:46,fontWeight:700,color:heat(d.score),lineHeight:1}}>{d.score?.toFixed(0)??'—'}</span>
          <span style={{fontSize:16,color:'#9d8b7a'}}>/100</span>
        </div>
        <div>
          <div style={{fontSize:22,fontWeight:700,color:ratingColor(d.news_rating),letterSpacing:0.5}}>{d.news_rating}</div>
          <div style={{fontSize:11,color:'#9d8b7a',marginTop:2}}>{d.article_count} articles analyzed · {d.coverage.scored}/{d.coverage.total} signals</div>
        </div>
        <div style={{marginLeft:'auto',minWidth:220}}>
          <div style={{fontSize:10,color:'#9d8b7a',marginBottom:4,letterSpacing:1}}>SENTIMENT DISTRIBUTION</div>
          <div style={{display:'flex',height:22,borderRadius:6,overflow:'hidden',border:'1px solid #2a2a2a'}}>
            <div style={{width:`${sd.positive/total*100}%`,background:'#0f9d6e',display:'flex',alignItems:'center',justifyContent:'center'}}>
              {sd.positive>0 && <span style={{fontSize:10,color:'#fff',fontWeight:600}}>{sd.positive}</span>}</div>
            <div style={{width:`${sd.neutral/total*100}%`,background:'#3a3a3a',display:'flex',alignItems:'center',justifyContent:'center'}}>
              {sd.neutral>0 && <span style={{fontSize:10,color:'#cdbfae'}}>{sd.neutral}</span>}</div>
            <div style={{width:`${sd.negative/total*100}%`,background:'#c0705a',display:'flex',alignItems:'center',justifyContent:'center'}}>
              {sd.negative>0 && <span style={{fontSize:10,color:'#fff',fontWeight:600}}>{sd.negative}</span>}</div>
          </div>
          <div style={{display:'flex',justifyContent:'space-between',fontSize:9,color:'#9d8b7a',marginTop:3}}>
            <span>{sd.positive} positive</span><span>{sd.neutral} neutral</span><span>{sd.negative} negative</span>
          </div>
        </div>
      </div>

      <div style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:12,padding:'16px 18px',marginBottom:16}}>
        <div style={{display:'flex',alignItems:'baseline',gap:10,marginBottom:12}}>
          <span style={{fontSize:14,fontWeight:700,color:'#daa520',letterSpacing:1}}>KEY HEADLINES</span>
          <span style={{fontSize:10,color:'#7a7266'}}>most important recent coverage</span>
        </div>
        <div style={{display:'flex',flexDirection:'column',gap:9}}>
          {(d.brief||[]).map((b,i)=>(
            <div key={i} style={{display:'flex',gap:10,alignItems:'flex-start'}}>
              <span style={{fontSize:12,color:'#7a7266',minWidth:18,textAlign:'right'}}>{i+1}.</span>
              <span style={{width:8,height:8,borderRadius:4,background:sentDot(b.sentiment),marginTop:5,flexShrink:0}}/>
              <div style={{flex:1}}>
                {b.url?<a href={b.url} target="_blank" rel="noopener noreferrer" style={{fontSize:13,color:'#e8ddd0',textDecoration:'none',lineHeight:1.4}}>{b.headline}</a>
                  :<span style={{fontSize:13,color:'#e8ddd0',lineHeight:1.4}}>{b.headline}</span>}
                <span style={{fontSize:10,color:'#7a7266',marginLeft:8}}>{b.publisher} · {b.date}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      <div style={{display:'grid',gridTemplateColumns:'repeat(auto-fill,minmax(130px,1fr))',gap:8,marginBottom:16}}>
        {[['Net Sentiment',km.net_sentiment!=null?(km.net_sentiment>0?'+':'')+km.net_sentiment.toFixed(2):'—'],
          ['7-Day Trend',km.sentiment_trend!=null?(km.sentiment_trend>0?'↑ ':'↓ ')+Math.abs(km.sentiment_trend).toFixed(2):'—'],
          ['Articles (7d)',km.article_count_7d!=null?km.article_count_7d.toString():'—'],
          ['Tier-1 Sources',km.tier1_source_share!=null?(km.tier1_source_share*100).toFixed(0)+'%':'—'],
          ['Contrarian',km.contrarian_signal!=null?(km.contrarian_signal>0?'Bullish':km.contrarian_signal<0?'Bearish':'None'):'—'],
          ['Risk Flag',km.fraud_litigation_flag!=null?(km.fraud_litigation_flag>0?'⚠ Yes':'Clean'):'—']].map(([k,v])=>(
          <div key={k as string} style={{background:'#181818',border:'1px solid #2a2a2a',borderRadius:8,padding:'8px 10px'}}>
            <div style={{fontSize:9,color:'#9d8b7a'}}>{k}</div>
            <div style={{fontSize:14,fontWeight:600,color:'#daa520'}}>{v}</div>
          </div>
        ))}
      </div>

      <div style={{fontSize:12,color:'#9d8b7a',letterSpacing:1,marginBottom:8}}>RECENT COVERAGE · sentiment + reasoning</div>
      <div style={{display:'flex',flexDirection:'column',gap:6,marginBottom:18}}>
        {(d.recent_headlines||[]).slice(0,10).map((h,i)=>(
          <div key={i} style={{background:'#141414',border:'1px solid #2a2a2a',borderRadius:8,padding:'9px 12px',borderLeft:`3px solid ${sentDot(h.sentiment)}`}}>
            <div style={{display:'flex',justifyContent:'space-between',gap:10}}>
              {h.url?<a href={h.url} target="_blank" rel="noopener noreferrer" style={{fontSize:12.5,color:'#e8ddd0',textDecoration:'none',fontWeight:500,flex:1}}>{h.title}</a>
                :<span style={{fontSize:12.5,color:'#e8ddd0',fontWeight:500,flex:1}}>{h.title}</span>}
              <span style={{fontSize:9,color:sentDot(h.sentiment),textTransform:'uppercase',fontWeight:600,flexShrink:0}}>{h.sentiment}</span>
            </div>
            {h.reason && <div style={{fontSize:11,color:'#9d8b7a',marginTop:4,lineHeight:1.4}}>{h.reason}</div>}
            <div style={{fontSize:9,color:'#7a7266',marginTop:3}}>{h.publisher} · {h.date}</div>
          </div>
        ))}
      </div>

      <div style={{display:'flex',justifyContent:'space-between',alignItems:'center',marginBottom:8}}>
        <span style={{fontSize:12,color:'#9d8b7a',letterSpacing:1}}>NEWS SIGNALS · {d.tree.categories.reduce((a,c)=>a+c.n_signals,0)} scored</span>
        <button onClick={()=>{setShowAllSignals(!showAllSignals);const m:Record<string,boolean>={};d.tree.categories.forEach(c=>m[c.id]=!showAllSignals);setExpanded(m);}}
          style={{background:'#181818',border:'1px solid #2a2a2a',color:'#9d8b7a',borderRadius:8,padding:'5px 12px',fontSize:11,cursor:'pointer'}}>
          {showAllSignals?'Collapse all':'Expand all'}</button>
      </div>
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
                        <span style={{fontSize:12,color:'#9d8b7a',width:64,textAlign:'right'}}>{pending?'—':fmtVal(s.id,s.raw_value)}</span>
                        <div style={{width:80,height:6,background:'#242424',borderRadius:3,overflow:'hidden'}}>
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
