import { useState } from 'react';
import { useAuth } from '../auth';

export default function Login() {
  const { login } = useAuth();
  const [email, setEmail] = useState('admin@cardioai.demo');
  const [password, setPassword] = useState('');
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(''); setBusy(true);
    try { await login(email, password); }
    catch (e) { setErr((e as Error).message || 'Login failed'); }
    finally { setBusy(false); }
  }

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <div style={{ textAlign: 'center', fontSize: '2.5rem' }}>❤️</div>
        <h1>CardioAI Platform</h1>
        <p className="sub">Women's CVD Clinical Decision Support</p>
        {err && <div className="err">⚠️ {err}</div>}
        <div className="field">
          <label>Email</label>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="username" required />
        </div>
        <div className="field">
          <label>Password</label>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete="current-password" required />
        </div>
        <button className="btn dark" style={{ width: '100%' }} disabled={busy}>{busy ? 'Signing in…' : '🔒 Sign In'}</button>
        <div className="hint">
          <strong>Demo credentials</strong><br />
          admin@cardioai.demo · physician@cardioai.demo<br />
          password: <code>ChangeMe!2026</code><br />
          <em>Synthetic data only. Change credentials before any real use.</em>
        </div>
      </form>
    </div>
  );
}
