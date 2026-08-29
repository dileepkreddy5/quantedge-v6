// ============================================================
// QuantEdge v6.0 — Command Search (hero)
// Inline ticker entry: type, Enter, analyze. No modal, no page hop.
// ============================================================

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const C = {
  s2: '#241610', b1: '#3a2920', gold: '#daa520', cocoa: '#8a7560',
  dust: '#9d8b7a', cream: '#f4e8d8',
};
const mono = "'Fira Code',monospace";

const CommandSearch: React.FC = () => {
  const navigate = useNavigate();
  const [q, setQ] = useState('');
  const go = () => {
    const t = q.trim().toUpperCase().replace(/[^A-Z0-9.\-]/g, '');
    if (t) navigate(`/dashboard?ticker=${t}`);
  };
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 0, maxWidth: 520,
      background: C.s2, border: `1px solid ${C.b1}`, borderRadius: 6,
      overflow: 'hidden', marginBottom: 22,
    }}>
      <span style={{ fontFamily: mono, fontSize: 13, color: C.gold, padding: '0 4px 0 16px' }}>❯</span>
      <input
        value={q}
        onChange={e => setQ(e.target.value)}
        onKeyDown={e => e.key === 'Enter' && go()}
        placeholder="TICKER — e.g. NVDA, then Enter"
        spellCheck={false}
        style={{
          flex: 1, background: 'none', border: 'none', outline: 'none',
          fontFamily: mono, fontSize: 14, letterSpacing: 2, color: C.cream,
          padding: '15px 10px', textTransform: 'uppercase',
        }}
      />
      <button onClick={go} style={{
        background: 'linear-gradient(135deg,#daa520,#b8860b)', border: 'none',
        color: '#1a0f0a', fontFamily: mono, fontWeight: 700, fontSize: 11,
        letterSpacing: 2, padding: '15px 22px', cursor: 'pointer',
      }}>RUN</button>
    </div>
  );
};

export default CommandSearch;
