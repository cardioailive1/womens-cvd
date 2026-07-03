import { createContext, useContext, useEffect, useState, type ReactNode } from 'react';
import { api, setToken, getToken, type User } from './api';

interface AuthState {
  user: User | null; loading: boolean;
  login: (e: string, p: string) => Promise<void>;
  bootstrap: (e: string, n: string, p: string) => Promise<void>;
  register: (e: string, n: string, p: string) => Promise<void>;
  logout: () => void; refreshUser: () => Promise<void>;
}
const AuthCtx = createContext<AuthState>(null as unknown as AuthState);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!getToken()) { setLoading(false); return; }
    api.me().then((r) => setUser(r.user)).catch(() => setToken(null)).finally(() => setLoading(false));
  }, []);

  async function login(email: string, password: string) {
    const r = await api.login(email, password);
    setToken(r.token);
    setUser(r.user);
  }
  async function bootstrap(email: string, fullName: string, password: string) {
    const r = await api.bootstrap(email, fullName, password);
    setToken(r.token);
    setUser(r.user);
  }
  async function register(email: string, fullName: string, password: string) {
    const r = await api.register(email, fullName, password);
    setToken(r.token);
    setUser(r.user);
  }
  function logout() { setToken(null); setUser(null); }
  async function refreshUser() { const r = await api.me(); setUser(r.user); }

  return <AuthCtx.Provider value={{ user, loading, login, bootstrap, register, logout, refreshUser }}>{children}</AuthCtx.Provider>;
}

export const useAuth = () => useContext(AuthCtx);
