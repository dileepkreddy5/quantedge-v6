// Watchlist as its own page: it isn't about one stock, so it doesn't belong among a stock's tabs.
import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Watchlist } from '../components/ui';
const WatchlistPage: React.FC = () => {
  const nav = useNavigate();
  const link = { cursor: 'pointer' } as React.CSSProperties;
  return (<div style={{ minHeight: '100vh', background: '#0d0806', color: '#d9c9b4', fontFamily: "'IBM Plex Sans',system-ui,sans-serif" }}>
    <div style={{ borderBottom: '1px solid #33241b' }}>
      <div style={{ maxWidth: 1720, margin: '0 auto', padding: '0 clamp(20px,3vw,56px)', height: 64, display: 'flex', alignItems: 'center', gap: 30 }}>
        <a onClick={() => nav('/')} style={{ fontFamily: "'IBM Plex Mono',monospace", fontWeight: 600, letterSpacing: '.4em', color: '#e0ad3a', fontSize: 15, cursor: 'pointer' }}>QUANTEDGE</a>
        <div style={{ marginLeft: 'auto', display: 'flex', gap: 26, fontSize: 14.5, color: '#b09c86' }}>
          <a onClick={() => nav('/')} style={link}>Markets</a><a onClick={() => nav('/trackers')} style={link}>Trackers</a>
          <a style={{ color: '#f6ecdd' }}>Watchlist</a><a onClick={() => nav('/methodology')} style={link}>How it works</a></div></div></div>
    <div style={{ maxWidth: 1720, margin: '0 auto', padding: '32px clamp(20px,3vw,56px) 60px' }}>
      <div style={{ fontFamily: "'IBM Plex Mono',monospace", fontSize: 11.5, letterSpacing: '.26em', color: '#e0ad3a' }}>WATCHLIST</div>
      <h1 style={{ fontFamily: "'Fraunces',Georgia,serif", fontWeight: 300, fontSize: 40, color: '#f6ecdd', margin: '10px 0 24px' }}>The companies you're following.</h1>
      <Watchlist onAnalyze={(t: string) => nav(`/dashboard?ticker=${t}`)} />
    </div></div>);
};
export default WatchlistPage;
