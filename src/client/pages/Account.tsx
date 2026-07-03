import { useState } from 'react';
import { api } from '../api';
import { useAuth } from '../auth';

export default function Account({ forced = false }: { forced?: boolean }) {
  const { refreshUser, logout } = useAuth();
  const [current, setCurrent] = useState('');
  const [next, setNext] = useState('');
  const [confirm, setConfirm] = useState('');
  const [err, setErr] = useState('');
  const [ok, setOk] = useState(false);
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setErr(''); setOk(false);
    if (next.length < 10) return setErr('New password must be at least 10 characters.');
    if (next !== confirm) return setErr('New password and confirmation do not match.');
    setBusy(true);
    try {
      await api.changePassword(current, next);
      setOk(true); setCurrent(''); setNext(''); setConfirm('');
      await refreshUser();
    } catch (e) { setErr((e as Error).message); }
    finally { setBusy(false); }
  }

  const body = (
    <form className="card" onSubmit={submit} style={{ maxWidth: 460 }}>
      <h3>{forced ? 'Set a new password' : 'Change password'}</h3>
      {forced && <div className="disclaimer">Your account uses a temporary password issued by an administrator. Please set your own password to continue.</div>}
      {err && <div className="err">{err}</div>}
      {ok && <div className="badge success" style={{ marginBottom: '1rem' }}>Password updated.</div>}
      <div className="field"><label>Current password</label><input type="password" value={current} onChange={(e) => setCurrent(e.target.value)} required /></div>
      <div className="field"><label>New password (min 10 chars)</label><input type="password" value={next} onChange={(e) => setNext(e.target.value)} required /></div>
      <div className="field"><label>Confirm new password</label><input type="password" value={confirm} onChange={(e) => setConfirm(e.target.value)} required /></div>
      <div className="row">
        <button className="btn dark" disabled={busy}>{busy ? 'Saving…' : 'Update password'}</button>
        {forced && <button type="button" className="btn ghost" onClick={logout}>Sign out</button>}
      </div>
    </form>
  );

  if (forced) return <div className="login-wrap" style={{ background: 'var(--bg)' }}>{body}</div>;
  return (<><div className="page-head"><h2>Account</h2><p>Manage your credentials</p></div>{body}</>);
}
