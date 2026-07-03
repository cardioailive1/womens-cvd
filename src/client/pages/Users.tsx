import { useEffect, useState } from 'react';
import { api, type ManagedUser } from '../api';

const ROLES = ['ADMIN', 'PHYSICIAN', 'NURSE', 'READONLY'];

export default function Users() {
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ email: '', fullName: '', role: 'PHYSICIAN', specialty: '', tempPassword: '' });
  const [flash, setFlash] = useState('');

  function load() { api.users().then((r) => setUsers(r.users)).catch((e) => setErr((e as Error).message)).finally(() => setLoading(false)); }
  useEffect(load, []);

  async function create(e: React.FormEvent) {
    e.preventDefault(); setErr(''); setFlash('');
    try {
      await api.createUser(form);
      setFlash(`Created ${form.email}. Share the temporary password securely; they must change it on first login.`);
      setForm({ email: '', fullName: '', role: 'PHYSICIAN', specialty: '', tempPassword: '' });
      setShowForm(false); load();
    } catch (e) { setErr((e as Error).message); }
  }

  async function toggleActive(u: ManagedUser) {
    setErr('');
    try { await api.updateUser(u.id, { active: !u.active }); load(); }
    catch (e) { setErr((e as Error).message); }
  }

  async function changeRole(u: ManagedUser, role: string) {
    setErr('');
    try { await api.updateUser(u.id, { role }); load(); }
    catch (e) { setErr((e as Error).message); }
  }

  async function resetPw(u: ManagedUser) {
    setErr(''); setFlash('');
    const temp = prompt(`New temporary password for ${u.email} (min 10 chars):`);
    if (!temp) return;
    try { await api.resetUserPassword(u.id, temp); setFlash(`Temporary password set for ${u.email}. They must change it on next login.`); load(); }
    catch (e) { setErr((e as Error).message); }
  }

  if (loading) return <div className="spinner" />;

  return (
    <>
      <div className="page-head"><h2>User Management</h2><p>Provision accounts and assign roles (admin only)</p></div>

      <div className="disclaimer">
        Accounts are provisioned by an administrator — there is no open self-registration. New users receive a
        temporary password and are required to set their own on first login. SSO/IdP federation is a documented
        future integration (see <code>backend/src/auth/sso.ts</code>).
      </div>

      {err && <div className="err">{err}</div>}
      {flash && <div className="badge success" style={{ display: 'block', marginBottom: '1rem', padding: '.6rem .8rem' }}>{flash}</div>}

      <div className="card" style={{ marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <h3 style={{ margin: 0 }}>Users ({users.length})</h3>
          <button className="btn" onClick={() => setShowForm((s) => !s)}>{showForm ? 'Cancel' : '+ Provision user'}</button>
        </div>

        {showForm && (
          <form onSubmit={create} style={{ marginTop: '1.25rem', borderTop: '1px solid var(--border)', paddingTop: '1.25rem' }}>
            <div className="row">
              <div className="field"><label>Email</label><input type="email" value={form.email} onChange={(e) => setForm({ ...form, email: e.target.value })} required /></div>
              <div className="field"><label>Full name</label><input value={form.fullName} onChange={(e) => setForm({ ...form, fullName: e.target.value })} required /></div>
            </div>
            <div className="row">
              <div className="field">
                <label>Role</label>
                <select value={form.role} onChange={(e) => setForm({ ...form, role: e.target.value })}>
                  {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
                </select>
              </div>
              <div className="field"><label>Specialty (optional)</label><input value={form.specialty} onChange={(e) => setForm({ ...form, specialty: e.target.value })} /></div>
              <div className="field"><label>Temporary password (min 10)</label><input value={form.tempPassword} onChange={(e) => setForm({ ...form, tempPassword: e.target.value })} required /></div>
            </div>
            <button className="btn dark">Create account</button>
          </form>
        )}
      </div>

      <div className="card">
        <table>
          <thead><tr><th>Name / Email</th><th>Role</th><th>Status</th><th>Last login</th><th>Actions</th></tr></thead>
          <tbody>
            {users.map((u) => (
              <tr key={u.id}>
                <td><strong>{u.fullName}</strong><br /><span style={{ color: 'var(--muted)', fontSize: '.82rem' }}>{u.email}</span></td>
                <td>
                  <select value={u.role} onChange={(e) => changeRole(u, e.target.value)} style={{ padding: '.3rem .4rem', fontSize: '.82rem' }}>
                    {ROLES.map((r) => <option key={r} value={r}>{r}</option>)}
                  </select>
                </td>
                <td>
                  <span className={`badge ${u.active ? 'success' : 'danger'}`}>{u.active ? 'Active' : 'Disabled'}</span>
                  {u.mustChangePassword && <span className="badge warning" style={{ marginLeft: 4 }}>temp pw</span>}
                </td>
                <td style={{ fontSize: '.82rem', color: 'var(--muted)' }}>{u.lastLoginAt ? new Date(u.lastLoginAt).toLocaleDateString() : '—'}</td>
                <td>
                  <div className="pill-row">
                    <button className="btn ghost" style={{ padding: '.3rem .6rem', fontSize: '.78rem' }} onClick={() => resetPw(u)}>Reset pw</button>
                    <button className="btn ghost" style={{ padding: '.3rem .6rem', fontSize: '.78rem' }} onClick={() => toggleActive(u)}>{u.active ? 'Disable' : 'Enable'}</button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
