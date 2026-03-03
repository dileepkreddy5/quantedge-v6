// ============================================================
// QuantEdge v6.0 — Auth Store (Zustand) + API Client
// Public by default; auth only needed to save watchlist/portfolio
// ============================================================

import axios, { AxiosInstance } from 'axios';
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

const API_BASE =
  process.env.REACT_APP_API_URL ||
  (process.env.NODE_ENV === 'production'
    ? 'https://quant.dileepkapu.com'
    : 'http://localhost:8000');

interface AuthState {
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  mfaSession: string | null;
  mfaUsername: string | null;
  login: (username: string, password: string) => Promise<{ requires_mfa: boolean; session?: string }>;
  verifyMfa: (code: string) => Promise<void>;
  logout: () => Promise<void>;
  refresh: () => Promise<boolean>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      accessToken:     null,
      refreshToken:    null,
      isAuthenticated: false,
      mfaSession:      null,
      mfaUsername:     null,

      login: async (username, password) => {
        const res = await axios.post(`${API_BASE}/auth/login`, { username, password });
        const data = res.data;
        if (data.requires_mfa) {
          set({ mfaSession: data.session, mfaUsername: username });
          return { requires_mfa: true, session: data.session };
        }
        return { requires_mfa: false };
      },

      verifyMfa: async (code) => {
        const { mfaSession, mfaUsername } = get();
        if (!mfaSession || !mfaUsername) throw new Error('No MFA session');
        const res = await axios.post(`${API_BASE}/auth/mfa`, {
          session: mfaSession,
          username: mfaUsername,
          mfa_code: code,
        });
        const { access_token, refresh_token } = res.data;
        set({
          accessToken:     access_token,
          refreshToken:    refresh_token,
          isAuthenticated: true,
          mfaSession:      null,
          mfaUsername:     null,
        });
      },

      logout: async () => {
        try {
          const { accessToken } = get();
          if (accessToken) {
            await axios.delete(`${API_BASE}/auth/logout`, {
              headers: { Authorization: `Bearer ${accessToken}` },
            });
          }
        } catch { /* ignore */ }
        set({ accessToken: null, refreshToken: null, isAuthenticated: false });
      },

      refresh: async () => {
        const { refreshToken } = get();
        if (!refreshToken) return false;
        try {
          const res = await axios.post(`${API_BASE}/auth/refresh`, { refresh_token: refreshToken });
          set({ accessToken: res.data.access_token });
          return true;
        } catch {
          set({ accessToken: null, refreshToken: null, isAuthenticated: false });
          return false;
        }
      },
    }),
    {
      name: 'qe-auth',
      partialize: (s) => ({
        accessToken:     s.accessToken,
        refreshToken:    s.refreshToken,
        isAuthenticated: s.isAuthenticated,
      }),
    }
  )
);

// ── API Client — attaches token if present, never blocks without one ──
export const createApiClient = (): AxiosInstance => {
  const client = axios.create({
    baseURL:         API_BASE,
    timeout:         120_000,  // 2 min for ML analysis
    withCredentials: true,
  });

  client.interceptors.request.use((config) => {
    const token = useAuthStore.getState().accessToken;
    if (token) config.headers.Authorization = `Bearer ${token}`;
    return config;
  });

  client.interceptors.response.use(
    (res) => res,
    async (error) => {
      const original = error.config;
      // Only try refresh on 401 for auth-required endpoints (watchlist, portfolio)
      if (error.response?.status === 401 && !original._retry) {
        original._retry = true;
        const refreshed = await useAuthStore.getState().refresh();
        if (refreshed) {
          const token = useAuthStore.getState().accessToken;
          original.headers.Authorization = `Bearer ${token}`;
          return client(original);
        }
        // Refresh failed — clear auth but DON'T redirect (page stays public)
        useAuthStore.getState().logout();
      }
      return Promise.reject(error);
    }
  );

  return client;
};

export const api = createApiClient();
