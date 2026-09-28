// QuantEdge v6.0 — App Root
// Public: landing + dashboard (no auth required)
// Protected: save watchlist/portfolio only

import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Toaster } from 'react-hot-toast';
import Landing from './pages/Landing';
import Home from './pages/Home';
import Trackers from './pages/Trackers';
import TrackersV2 from './pages/TrackersV2';
import Legal from './pages/Legal';
import LandingWire from './pages/LandingWire';
import Dashboard from './pages/Dashboard';
import Screener from './pages/Screener';
import AscentRadar from './pages/AscentRadar';
import Methodology from './pages/Methodology';
import Login from './pages/Login';
import Research from './pages/Research';
import Multibagger from './pages/Multibagger';
import Rebound from './pages/Rebound';

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
        <Route path="/"          element={<Home />} />
        <Route path="/trackers-new" element={<TrackersV2 />} />
        <Route path="/trackers" element={<Trackers />} />
        <Route path="/trackers/:tab" element={<Trackers />} />
        <Route path="/terms"     element={<Legal page="terms" />} />
        <Route path="/privacy"   element={<Legal page="privacy" />} />
        <Route path="/disclaimer" element={<Legal page="disclaimer" />} />
        <Route path="/data-sources" element={<Legal page="data-sources" />} />
        <Route path="/classic"   element={<Landing />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/screener" element={<Navigate to="/trackers/filter" replace />} />
        <Route path="/ascent" element={<Navigate to="/trackers/climbers" replace />} />
        <Route path="/methodology" element={<Methodology />} />
        <Route path="/login"     element={<Login />} />
        <Route path="/research"  element={<Research />} />
        <Route path="/multibagger" element={<Navigate to="/trackers/fast-growers" replace />} />
        <Route path="/rebound" element={<Navigate to="/trackers/comebacks" replace />} />
        {/* Catch-all → landing */}
        <Route path="*"          element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}
