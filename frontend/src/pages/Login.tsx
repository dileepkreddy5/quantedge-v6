// ============================================================
// QuantEdge v5.0 — Login Page
// Matches dileepkapu.com chocolate/espresso aesthetic exactly
// ============================================================

import React, { useState, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuthStore } from '../auth/authStore';
import toast from 'react-hot-toast';

// ── Grain texture SVG (same as dileepkapu.com) ───────────────
const GRAIN_SVG = `url("data:image/svg+xml,%3Csvg viewBox='0 0 400 400' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='3' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E")`;

export default function Login() {
  const navigate = useNavigate();
  const { login, verifyMfa, isAuthenticated } = useAuthStore();

  const [step, setStep] = useState<'credentials' | 'mfa'>('credentials');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [mfaCode, setMfaCode] = useState('');
  const [loading, setLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [attempts, setAttempts] = useState(0);
  const mfaInputs = useRef<(HTMLInputElement | null)[]>([]);
  const [mfaDigits, setMfaDigits] = useState(['', '', '', '', '', '']);

  useEffect(() => {
    if (isAuthenticated) navigate('/dashboard');
  }, [isAuthenticated, navigate]);

  // Auto-focus first MFA input when step changes
  useEffect(() => {
    if (step === 'mfa') setTimeout(() => mfaInputs.current[0]?.focus(), 100);
  }, [step]);

  const handleMfaDigit = (idx: number, value: string) => {
    if (!/^\d*$/.test(value)) return;
    const newDigits = [...mfaDigits];
    newDigits[idx] = value.slice(-1);
    setMfaDigits(newDigits);
    setMfaCode(newDigits.join(''));
    if (value && idx < 5) mfaInputs.current[idx + 1]?.focus();
    if (newDigits.every(d => d) && newDigits.join('').length === 6) {
      handleMfaSubmit(newDigits.join(''));
    }
  };

  const handleMfaKeyDown = (idx: number, e: React.KeyboardEvent) => {
    if (e.key === 'Backspace' && !mfaDigits[idx] && idx > 0) {
      mfaInputs.current[idx - 1]?.focus();
    }
  };

  const handleCredentials = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username || !password) return;
    setLoading(true);
    try {
      const result = await login(username, password);
      if (result.requires_mfa) {
        setStep('mfa');
        toast.success('Enter your authenticator code', { icon: '🔐' });
      }
    } catch (err: any) {
      setAttempts(a => a + 1);
      const msg = err?.response?.data?.detail || 'Invalid credentials';
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleMfaSubmit = async (code?: string) => {
    const finalCode = code || mfaCode;
    if (finalCode.length !== 6) return;
    setLoading(true);
    try {
      await verifyMfa(finalCode);
      toast.success('Welcome back, Dileep 👋');
      navigate('/dashboard');
    } catch (err: any) {
      const msg = err?.response?.data?.detail || 'Invalid MFA code';
      toast.error(msg);
      setMfaDigits(['', '', '', '', '', '']);
      setMfaCode('');
      setTimeout(() => mfaInputs.current[0]?.focus(), 50);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{
      minHeight: '100vh',
      background: '#1a0f0a',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      fontFamily: "'Outfit', sans-serif",
      position: 'relative',
      overflow: 'hidden',
    }}>

      {/* Grain texture */}
      <div style={{
        position: 'fixed', inset: 0,
        backgroundImage: GRAIN_SVG,
        opacity: 0.025, pointerEvents: 'none', zIndex: 0,
      }} />

      {/* Ambient gradient glow */}
      <div style={{
        position: 'fixed', inset: 0, zIndex: 0,
        background: `
          radial-gradient(ellipse at 20% 30%, rgba(212,149,108,0.08) 0%, transparent 60%),
          radial-gradient(ellipse at 80% 70%, rgba(218,165,32,0.06) 0%, transparent 60%)
        `,
      }} />

      {/* Login card */}
      <div style={{
        position: 'relative', zIndex: 1,
        width: '100%', maxWidth: 420,
        margin: '0 16px',
        background: '#2d1e18',
        border: '1px solid rgba(212,149,108,0.15)',
        borderRadius: 12,
        padding: '40px 36px',
        boxShadow: '0 24px 80px rgba(0,0,0,0.7), 0 2px 8px rgba(0,0,0,0.4)',
        animation: 'fadeIn 0.5s ease',
      }}>

        {/* Header */}
        <div style={{ textAlign: 'center', marginBottom: 36 }}>
          <div style={{
            fontFamily: "'Bebas Neue', sans-serif",
            fontSize: 32, letterSpacing: 8,
            color: '#daa520',
            marginBottom: 6,
          }}>
            QUANTEDGE
          </div>
          <div style={{
            fontFamily: "'Fira Code', monospace",
            fontSize: 9, letterSpacing: 3,
            color: '#9d8b7a', textTransform: 'uppercase',
          }}>
            INSTITUTIONAL · QUANTITATIVE · ANALYTICS
          </div>
          <div style={{
            width: 60, height: 1,
            background: 'linear-gradient(90deg, transparent, #daa520, transparent)',
            margin: '16px auto 0',
          }} />
        </div>

        {/* Greeting */}
        <div style={{
          fontFamily: "'Fira Code', monospace",
          fontSize: 10, color: '#9d8b7a',
          letterSpacing: 1, marginBottom: 24, textAlign: 'center',
        }}>
          {step === 'credentials'
            ? '🔒 Save watchlist & portfolio'
            : '📱 Google Authenticator · Enter 6-digit code'
          }
        </div>

        {/* Step indicator */}
        <div style={{ display: 'flex', gap: 6, marginBottom: 24 }}>
          {['Credentials', 'Authenticator'].map((s, i) => (
            <div key={s} style={{
              flex: 1, height: 2, borderRadius: 1,
              background: (step === 'credentials' ? i === 0 : i <= 1)
                ? '#daa520' : 'rgba(218,165,32,0.2)',
              transition: 'background 0.4s',
            }} />
          ))}
        </div>

        {/* ── Step 1: Username + Password ── */}
        {step === 'credentials' && (
          <form onSubmit={handleCredentials} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
            <div>
              <label style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: '#9d8b7a', letterSpacing: 2, display: 'block', marginBottom: 6 }}>
                USERNAME
              </label>
              <input
                value={username}
                onChange={e => setUsername(e.target.value)}
                autoFocus
                autoComplete="username"
                placeholder="dileep"
                style={inputStyle}
              />
            </div>
            <div>
              <label style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: '#9d8b7a', letterSpacing: 2, display: 'block', marginBottom: 6 }}>
                PASSWORD
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  type={showPassword ? 'text' : 'password'}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  autoComplete="current-password"
                  placeholder="••••••••••••••••"
                  style={{ ...inputStyle, paddingRight: 44 }}
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(v => !v)}
                  style={{
                    position: 'absolute', right: 12, top: '50%',
                    transform: 'translateY(-50%)', background: 'none',
                    border: 'none', color: '#9d8b7a', cursor: 'pointer', fontSize: 14,
                  }}
                >
                  {showPassword ? '🙈' : '👁'}
                </button>
              </div>
            </div>

            {attempts > 2 && (
              <div style={{ fontFamily: "'Fira Code',monospace", fontSize: 9, color: '#ef4444', textAlign: 'center' }}>
                ⚠ Multiple failed attempts — account may be locked after {5 - attempts} more
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !username || !password}
              style={{
                ...btnStyle,
                marginTop: 8,
                opacity: (loading || !username || !password) ? 0.5 : 1,
                cursor: (loading || !username || !password) ? 'not-allowed' : 'pointer',
              }}
            >
              {loading ? (
                <span style={{ display: 'flex', alignItems: 'center', gap: 8, justifyContent: 'center' }}>
                  <span style={{ width: 14, height: 14, border: '2px solid #1a0f0a', borderTopColor: 'transparent', borderRadius: '50%', display: 'inline-block', animation: 'spin 0.7s linear infinite' }} />
                  AUTHENTICATING...
                </span>
              ) : 'CONTINUE →'}
            </button>
          </form>
        )}

        {/* ── Step 2: MFA Code ── */}
        {step === 'mfa' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div style={{ textAlign: 'center' }}>
              <div style={{ fontSize: 40, marginBottom: 8 }}>📱</div>
              <div style={{ color: '#d4c4b0', fontSize: 13 }}>
                Open Google Authenticator<br />
                and enter the 6-digit code
              </div>
            </div>

            {/* 6-digit OTP input */}
            <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
              {mfaDigits.map((digit, i) => (
                <input
                  key={i}
                  ref={el => { mfaInputs.current[i] = el; }}
                  value={digit}
                  onChange={e => handleMfaDigit(i, e.target.value)}
                  onKeyDown={e => handleMfaKeyDown(i, e)}
                  type="text"
                  inputMode="numeric"
                  maxLength={1}
                  style={{
                    width: 44, height: 52,
                    background: '#1a0f0a',
                    border: `2px solid ${digit ? '#daa520' : 'rgba(212,149,108,0.2)'}`,
                    borderRadius: 6,
                    color: '#f4e8d8',
                    fontFamily: "'Fira Code', monospace",
                    fontSize: 20, fontWeight: 700,
                    textAlign: 'center',
                    outline: 'none',
                    transition: 'border-color 0.15s',
                  }}
                />
              ))}
            </div>

            <button
              onClick={() => handleMfaSubmit()}
              disabled={loading || mfaCode.length !== 6}
              style={{
                ...btnStyle,
                opacity: (loading || mfaCode.length !== 6) ? 0.5 : 1,
                cursor: (loading || mfaCode.length !== 6) ? 'not-allowed' : 'pointer',
              }}
            >
              {loading ? 'VERIFYING...' : 'VERIFY & ACCESS →'}
            </button>

            <button
              onClick={() => { setStep('credentials'); setMfaDigits(['','','','','','']); }}
              style={{ background: 'none', border: 'none', color: '#9d8b7a', fontFamily: "'Fira Code',monospace", fontSize: 10, cursor: 'pointer', textDecoration: 'underline' }}
            >
              ← Back to credentials
            </button>
          </div>
        )}

        {/* Footer */}
        <div style={{
          marginTop: 28, paddingTop: 20,
          borderTop: '1px solid rgba(212,149,108,0.08)',
          textAlign: 'center',
          fontFamily: "'Fira Code', monospace",
          fontSize: 9, color: '#4a3428', letterSpacing: 1,
        }}>
          QUANTEDGE v6.0 · DILEEP KUMAR REDDY KAPU
          <br />8 ML MODELS · 200+ SIGNALS · PRIVATE ACCESS
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;600;800&family=Fira+Code:wght@400;600&display=swap');
        @keyframes fadeIn { from{opacity:0;transform:translateY(16px);}to{opacity:1;transform:none;} }
        @keyframes spin { to{transform:rotate(360deg);} }
        input::placeholder { color: #4a3428; }
        input:focus { border-color: #daa520 !important; box-shadow: 0 0 0 3px rgba(218,165,32,0.1); }
      `}</style>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  width: '100%',
  background: '#1a0f0a',
  border: '1px solid rgba(212,149,108,0.2)',
  borderRadius: 6,
  color: '#f4e8d8',
  fontFamily: "'Fira Code', monospace",
  fontSize: 13,
  padding: '11px 14px',
  outline: 'none',
  transition: 'border-color 0.15s, box-shadow 0.15s',
};

const btnStyle: React.CSSProperties = {
  width: '100%',
  background: 'linear-gradient(135deg, #daa520, #b8860b)',
  color: '#1a0f0a',
  fontFamily: "'Fira Code', monospace",
  fontWeight: 700,
  fontSize: 11,
  letterSpacing: 3,
  padding: '13px 20px',
  border: 'none',
  borderRadius: 6,
  cursor: 'pointer',
  textTransform: 'uppercase' as const,
  transition: 'all 0.15s',
  boxShadow: '0 4px 16px rgba(218,165,32,0.25)',
};
