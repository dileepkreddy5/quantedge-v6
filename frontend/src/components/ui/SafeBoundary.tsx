// Catches a render crash in one panel so it reports itself instead of blanking the dashboard.
import React from 'react';
export default class SafeBoundary extends React.Component<{label:string;children:React.ReactNode},{err:string|null}> {
  state={err:null as string|null};
  static getDerivedStateFromError(e:any){return {err:String(e?.message||e)};}
  componentDidCatch(e:any){console.error('[SafeBoundary]',this.props.label,e);}
  render(){ if(this.state.err) return (
    <div style={{fontFamily:"'Fira Code',monospace",fontSize:11,color:'#f59e0b',border:'1px solid #3a2920',borderRadius:8,padding:14}}>
      {this.props.label} failed to render: {this.state.err}. This panel is disabled; the rest of the page works.</div>);
    return this.props.children; }
}
