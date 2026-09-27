// Canvas chart (TradingView lightweight-charts v4): candles, volume, SMAs,
// 52w price lines, formations drawn through their extrema, candlestick and
// earnings markers, RS vs SPY on its own scale, and the analog envelope
// extended past today. Zoom/pan/crosshair; legend shows the bar and any
// pattern on it with its measured odds.
import React, { useEffect, useRef, useState } from 'react';
import { createChart, IChartApi, ISeriesApi, LineStyle, UTCTimestamp } from 'lightweight-charts';
const C={gold:'#daa520',caramel:'#d4956c',cocoa:'#8a7560',dust:'#9d8b7a',latte:'#d4c4b0',cream:'#f4e8d8',bull:'#22c55e',bear:'#ef4444',blue:'#60a5fa',b1:'#3a2920'};
const mono="'Fira Code',monospace";
const BEAR_F=new Set(['head_shoulders','double_top','triple_top','rising_wedge','descending_triangle']);
const LABEL:Record<string,string>={head_shoulders:'Head & Shoulders',inv_head_shoulders:'Inv. H&S',double_top:'Double Top',double_bottom:'Double Bottom',
  triple_top:'Triple Top',triple_bottom:'Triple Bottom',ascending_triangle:'Asc. Triangle',descending_triangle:'Desc. Triangle',symmetrical_triangle:'Sym. Triangle',
  rectangle:'Rectangle',rising_wedge:'Rising Wedge',falling_wedge:'Falling Wedge',doji:'Doji',hammer:'Hammer',hanging_man:'Hanging Man',bullish_engulfing:'Bull Engulfing',
  bearish_engulfing:'Bear Engulfing',bullish_harami:'Bull Harami',bearish_harami:'Bear Harami',morning_star:'Morning Star',evening_star:'Evening Star',
  three_white_soldiers:'3 White Soldiers',three_black_crows:'3 Black Crows'};
const pf=(v:any,d=1)=>v==null?'—':`${v>=0?'+':''}${Number(v).toFixed(d)}%`;
const nextBiz=(iso:string,k:number)=>{const d=new Date(iso+'T00:00:00Z');let n=0;while(n<k){d.setUTCDate(d.getUTCDate()+1);if(d.getUTCDay()!==0&&d.getUTCDay()!==6)n++;}return d.toISOString().slice(0,10);};

const PatternChartPro:React.FC<{d:any;show:{f:boolean;c:boolean;doji:boolean};height?:number}>=({d,show,height=520})=>{
  const ref=useRef<HTMLDivElement>(null); const chartRef=useRef<IChartApi|null>(null);
  const [legend,setLegend]=useState<any>(null);
  useEffect(()=>{ if(!ref.current||!d) return;
    const chart=createChart(ref.current,{height,layout:{background:{color:'rgba(0,0,0,0)'},textColor:C.dust,fontFamily:mono,fontSize:10},
      grid:{vertLines:{color:'rgba(58,41,32,0.35)'},horzLines:{color:'rgba(58,41,32,0.35)'}},
      rightPriceScale:{borderColor:C.b1,scaleMargins:{top:0.12,bottom:0.22}},timeScale:{borderColor:C.b1,rightOffset:4,barSpacing:7,fixLeftEdge:true,fixRightEdge:true,lockVisibleTimeRangeOnResize:true},
      handleScale:{axisPressedMouseMove:{time:true,price:false}},
      crosshair:{mode:0,vertLine:{color:C.gold,labelBackgroundColor:'#3a2920'},horzLine:{color:C.gold,labelBackgroundColor:'#3a2920'}}});
    chartRef.current=chart;
    const cs=d.candles; const N=cs.length; const t=(i:number)=>cs[Math.max(0,Math.min(N-1,i))].d as any;
    const inWin=(i:number)=>i>=0&&i<N;
    const candles=chart.addCandlestickSeries({upColor:C.bull,downColor:C.bear,borderUpColor:C.bull,borderDownColor:C.bear,wickUpColor:C.bull,wickDownColor:C.bear});
    candles.setData(cs.map((c:any)=>({time:c.d,open:c.o,high:c.h,low:c.l,close:c.c})));
    const vol=chart.addHistogramSeries({priceScaleId:'vol',priceFormat:{type:'volume'},lastValueVisible:false,priceLineVisible:false});
    chart.priceScale('vol').applyOptions({scaleMargins:{top:0.82,bottom:0}});
    vol.setData(cs.map((c:any,i:number)=>{const avg=i>=20?cs.slice(i-20,i).reduce((a:number,z:any)=>a+z.v,0)/20:c.v;
      return {time:c.d,value:c.v,color:c.v>=2.5*avg?'rgba(218,165,32,0.85)':c.c>=c.o?'rgba(34,197,94,0.35)':'rgba(239,68,68,0.35)'};}));
    const vol5=chart.addLineSeries({priceScaleId:'vol',color:C.gold,lineWidth:1,lastValueVisible:false,priceLineVisible:false,crosshairMarkerVisible:false});
    vol5.setData(cs.map((c:any,i:number)=>i>=4?{time:c.d,value:cs.slice(i-4,i+1).reduce((a:number,z:any)=>a+z.v,0)/5}:null).filter(Boolean) as any);
    const sma=(arr:any[],color:string,style:LineStyle)=>{if(!arr)return;const s=chart.addLineSeries({color,lineWidth:1,lineStyle:style,lastValueVisible:false,priceLineVisible:false,crosshairMarkerVisible:false});
      s.setData(arr.map((v:any,i:number)=>v==null?null:{time:t(i),value:v}).filter(Boolean) as any);};
    sma(d.sma20,C.gold,LineStyle.Solid); sma(d.sma50,C.caramel,LineStyle.Dashed); sma(d.sma200,C.dust,LineStyle.Dotted);
    candles.createPriceLine({price:d.high_52w,color:C.caramel,lineWidth:1,lineStyle:LineStyle.Dashed,axisLabelVisible:true,title:'52W HIGH'});
    candles.createPriceLine({price:d.low_52w,color:C.blue,lineWidth:1,lineStyle:LineStyle.Dashed,axisLabelVisible:true,title:'52W LOW'});
    // RS vs SPY on its own top strip
    if(d.relative_strength_vs_spy){const rs=chart.addLineSeries({priceScaleId:'rs',color:'rgba(96,165,250,0.9)',lineWidth:1,lastValueVisible:true,priceLineVisible:false,crosshairMarkerVisible:false,title:'RS vs SPY'});
      chart.priceScale('rs').applyOptions({scaleMargins:{top:0.0,bottom:0.9},visible:false});
      rs.setData(d.relative_strength_vs_spy.map((v:any,i:number)=>v==null?null:{time:t(i),value:v}).filter(Boolean) as any);}
    // formations drawn through their five extrema
    if(show.f) d.formations.forEach((f:any)=>{const col=BEAR_F.has(f.name)?C.bear:C.bull;
      const pts=f.points.filter((p:any)=>inWin(p.i)); if(pts.length<2) return;   // extrema before the window can't be drawn
      const s=chart.addLineSeries({color:col,lineWidth:2,lastValueVisible:false,priceLineVisible:false,crosshairMarkerVisible:false});
      s.setData(pts.map((p:any)=>({time:t(p.i),value:p.price})));});
    // markers: candlestick patterns + earnings + formation completions
    const marks:any[]=[];
    if(show.c) d.candlesticks.filter((o:any)=>inWin(o.i)&&(show.doji||o.name!=='doji')).forEach((o:any)=>{const up=o.direction==='bullish',neu=o.direction==='neutral';
      marks.push({time:t(o.i),position:up?'belowBar':'aboveBar',color:neu?C.gold:up?C.bull:C.bear,shape:neu?'circle':up?'arrowUp':'arrowDown',size:0.8});});
    if(show.f) d.formations.forEach((f:any)=>marks.push({time:t(Math.min(f.confirm_i??f.i,cs.length-1)),position:BEAR_F.has(f.name)?'aboveBar':'belowBar',color:BEAR_F.has(f.name)?C.bear:C.bull,shape:'square',text:LABEL[f.name],size:1}));
    (d.earnings||[]).filter((e:any)=>inWin(e.i)).forEach((e:any)=>marks.push({time:t(e.i),position:'aboveBar',color:C.blue,shape:'circle',text:'EARNINGS',size:1}));
    marks.sort((a,b)=>String(a.time)<String(b.time)?-1:String(a.time)>String(b.time)?1:0);
    try{ candles.setMarkers(marks); }catch(e){ console.warn('markers skipped',e); }
    // analog envelope extended past today (real forward paths' median / p25 / p75)
    const fan=d.analog?.forward_fan; const last=cs[cs.length-1];
    if(fan){const F=Math.min(fan.sessions,d.outcome_sessions); const fp=(p:number)=>last.c*(1+p/100);
      const mk=(arr:number[],color:string,width:any,style:LineStyle,title:string)=>{const s=chart.addLineSeries({color,lineWidth:width,lineStyle:style,lastValueVisible:true,priceLineVisible:false,crosshairMarkerVisible:false,title});
        s.setData(Array.from({length:F},(_,j)=>({time:nextBiz(last.d,j+1),value:fp(arr[j])})));};
      mk(fan.p75.slice(0,F),'rgba(218,165,32,0.55)',1,LineStyle.Dotted,'p75'); mk(fan.p25.slice(0,F),'rgba(218,165,32,0.55)',1,LineStyle.Dotted,'p25'); mk(fan.median.slice(0,F),C.gold,2,LineStyle.Solid,'median');}
    // pattern cone from the latest measured pattern
    const measured=[...d.formations,...d.candlesticks].filter((o:any)=>{const st=o.scorecard?.all||o.scorecard;return st&&st.median_pct!=null&&st.p25_pct!=null;}).sort((a:any,b:any)=>b.i-a.i)[0];
    if(measured){const st=measured.scorecard?.all||measured.scorecard; const i0=Math.min(measured.confirm_i??measured.i,cs.length-1); const p0=cs[i0].c; const K=d.outcome_sessions; const col=st.positive_pct>=50?C.bull:C.bear;
      const endT=i0+K<cs.length?t(i0+K):nextBiz(last.d,i0+K-(cs.length-1));
      [[st.median_pct,2,LineStyle.Dashed],[st.p25_pct,1,LineStyle.Dotted],[st.p75_pct,1,LineStyle.Dotted]].forEach(([pct,w,ls]:any)=>{
        const s=chart.addLineSeries({color:col,lineWidth:w,lineStyle:ls,lastValueVisible:false,priceLineVisible:false,crosshairMarkerVisible:false});
        s.setData([{time:t(i0),value:p0},{time:endT,value:p0*(1+pct/100)}]);});}
    // legend
    const byTime:Record<string,any[]>={}; [...d.formations,...d.candlesticks].forEach((o:any)=>{const k=cs[o.i]?.d; if(k)(byTime[k]=byTime[k]||[]).push(o);});
    chart.subscribeCrosshairMove((p:any)=>{if(!p.time){setLegend(null);return;} const bar:any=p.seriesData.get(candles); if(!bar){setLegend(null);return;}
      setLegend({time:p.time,o:bar.open,h:bar.high,l:bar.low,c:bar.close,pats:byTime[p.time as string]||[]});});
    chart.timeScale().fitContent();
    const ro=new ResizeObserver(()=>chart.applyOptions({width:ref.current?.clientWidth||800})); ro.observe(ref.current);
    return ()=>{ro.disconnect();chart.remove();chartRef.current=null;};
  },[d,show,height]);
  return (<div style={{position:'relative'}}>
    <div ref={ref} style={{width:'100%'}}/>
    <div style={{position:'absolute',left:10,top:8,fontFamily:mono,fontSize:10,color:C.latte,pointerEvents:'none',background:'rgba(16,10,7,0.75)',padding:'4px 8px',borderRadius:4}}>
      {legend?(<><span style={{color:C.cream}}>{legend.time}</span> · O {legend.o} H {legend.h} L {legend.l} C <span style={{color:legend.c>=legend.o?C.bull:C.bear}}>{legend.c}</span>
        {legend.pats.map((o:any,i:number)=>{const st=o.scorecard?.all||o.scorecard;return <span key={i} style={{color:C.gold}}> · {LABEL[o.name]||o.name}{st?.positive_pct!=null?` ${st.positive_pct}% up · med ${pf(st.median_pct,1)} (n=${st.n})`:' (not yet measured)'}</span>;})}</>)
        :<span style={{color:C.cocoa}}>scroll to zoom · drag to pan · hover a bar</span>}
    </div>
  </div>);
};
export default PatternChartPro;
