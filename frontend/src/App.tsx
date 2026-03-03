// QuantEdge v6.0 — App Root
// Public: landing + dashboard (no auth required)
// Protected: save watchlist/portfolio only

import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Landing from './pages/Landing';
import Dashboard from './pages/Dashboard';
import Login from './pages/Login';

export default function App() {
  return (
    <BrowserRouter>
      <Toaster
        position="bottom-right"
        toastOptions={{
          style: {
            background: '#2d1e18',
            color: '#f4e8d8',
            border: '1px solid rgba(212,149,108,0.3)',
            fontFamily: "'Fira Code', monospace",
            fontSize: '11px',
            letterSpacing: '0.5px',
          },
        }}
      />
      <Routes>
        {/* Public */}
        <Route path="/"          element={<Landing />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/login"     element={<Login />} />
        {/* Catch-all → landing */}
        <Route path="*"          element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
