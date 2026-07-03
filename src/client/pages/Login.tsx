import { useEffect, useState } from 'react';
import { api } from '../api';
import { useAuth } from '../auth';

type Mode = 'login' | 'register' | 'bootstrap';

export default function Login() {
  const { login, register, bootstrap } = useAuth();
  const [mode, setMode] = useState<Mode>('login');
  const [openSignup, setOpenSignup] = useState(false);
  const [email, setEmail] = useState('');
  const [fullName, setFullName] = useState('');
  const [password, setPassword] = useState('');
  const [err, setErr] = useState('');
  const [busy, setBusy] = useState(false);

  // Ask the server what to show: first-run bootstrap, and whether open signup is on.
  useEffect(() => {
    api.authConfig()
      .then((c) => {
        setOpenSignup(c.openSignup);
        if (c.needsBootstrap) setMode('bootstrap');
      })
      .catch(() => {/* fall back to plain login */});
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(''); setBusy(true);
    try {
      if (mode === 'login') await login(email, password);
      else if (mode === 'register') await register(email, fullName, password);
      else await bootstrap(email, fullName, password);
    } catch (e) {
      setErr((e as Error).message || 'Something went wrong');
    } finally {
      setBusy(false);
    }
  }

  const isLogin = mode === 'login';
  const title = mode === 'bootstrap' ? 'Create the first administrator' : mode === 'register' ? 'Create your account' : 'CardioAI Platform';
  const cta = mode === 'bootstrap' ? 'Create admin & continue' : mode === 'register' ? 'Create account' : '🔒 Sign In';

  return (
    <div className="login-wrap">
      <form className="login-card" onSubmit={submit}>
        <div style={{ textAlign: 'center', fontSize: '2.5rem' }}>❤️</div>
        <h1>{title}</h1>
        <p className="sub">Women's CVD Clinical Decision Support</p>

        {mode === 'bootstrap' && (
          <div className="hint" style={{ marginTop: 0, marginBottom: '1rem' }}>
            No accounts exist yet. Create the first administrator to get started — this option disappears once an account exists.
          </div>
        )}
        {err && <div className="err">⚠️ {err}</div>}

        {!isLogin && (
          <div className="field">
            <label>Full name</label>
            <input value={fullName} onChange={(e) => setFullName(e.target.value)} required />
          </div>
        )}
        <div className="field">
          <label>Email</label>
          <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} autoComplete="username" required />
        </div>
        <div className="field">
          <label>Password{!isLogin && ' (min 10 characters)'}</label>
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} autoComplete={isLogin ? 'current-password' : 'new-password'} required />
        </div>

        <button className="btn dark" style={{ width: '100%' }} disabled={busy}>{busy ? 'Please wait…' : cta}</button>

        {/* Mode switches (hidden during first-run bootstrap) */}
        {mode !== 'bootstrap' && (
          <div style={{ textAlign: 'center', marginTop: '1rem', fontSize: '.85rem' }}>
            {isLogin && openSignup && (
              <span>New here? <a onClick={() => { setErr(''); setMode('register'); }} style={{ cursor: 'pointer' }}>Create an account</a></span>
            )}
            {mode === 'register' && (
              <span>Already have an account? <a onClick={() => { setErr(''); setMode('login'); }} style={{ cursor: 'pointer' }}>Sign in</a></span>
            )}
          </div>
        )}

        {isLogin && (
          <div className="hint">
            <strong>Demo credentials</strong><br />
            admin@cardioai.demo · physician@cardioai.demo<br />
            password: <code>ChangeMe!2026</code><br />
            <em>Synthetic data only. Change credentials before any real use.</em>
          </div>
        )}
        {mode === 'register' && (
          <div className="hint">New accounts are created with <strong>read-only</strong> access. An administrator can elevate your role.</div>
        )}
      </form>
    </div>
  );
}
