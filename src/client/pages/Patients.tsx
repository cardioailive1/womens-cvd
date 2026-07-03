import { useEffect, useState } from 'react';
import { api, type Patient } from '../api';

export default function Patients() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [q, setQ] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState({ mrn: '', firstName: '', lastName: '', birthDate: '', sex: 'FEMALE', pregnant: false, preeclampsiaHx: false, hypertension: false });
  const [err, setErr] = useState('');

  function load() { api.patients().then((r) => setPatients(r.patients)).finally(() => setLoading(false)); }
  useEffect(load, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault(); setErr('');
    try {
      await api.createPatient(form);
      setShowForm(false);
      setForm({ mrn: '', firstName: '', lastName: '', birthDate: '', sex: 'FEMALE', pregnant: false, preeclampsiaHx: false, hypertension: false });
      load();
    } catch (e) { setErr((e as Error).message); }
  }

  const filtered = patients.filter((p) =>
    `${p.mrn} ${p.firstName} ${p.lastName}`.toLowerCase().includes(q.toLowerCase()));

  if (loading) return <div className="spinner" />;

  return (
    <>
      <div className="page-head"><h2>Patients</h2><p>Registry — PHI is encrypted at rest (AES-256-GCM)</p></div>

      <div className="card" style={{ marginBottom: '1.25rem' }}>
        <div className="row" style={{ alignItems: 'flex-end' }}>
          <div style={{ flex: 3 }}>
            <label>Search</label>
            <input placeholder="Name or MRN…" value={q} onChange={(e) => setQ(e.target.value)} />
          </div>
          <button className="btn" style={{ flex: 0, whiteSpace: 'nowrap' }} onClick={() => setShowForm((s) => !s)}>
            {showForm ? 'Cancel' : '+ Add Patient'}
          </button>
        </div>

        {showForm && (
          <form onSubmit={submit} style={{ marginTop: '1.25rem', borderTop: '1px solid var(--border)', paddingTop: '1.25rem' }}>
            {err && <div className="err">{err}</div>}
            <div className="row">
              <div className="field"><label>MRN</label><input value={form.mrn} onChange={(e) => setForm({ ...form, mrn: e.target.value })} required /></div>
              <div className="field"><label>First name</label><input value={form.firstName} onChange={(e) => setForm({ ...form, firstName: e.target.value })} required /></div>
              <div className="field"><label>Last name</label><input value={form.lastName} onChange={(e) => setForm({ ...form, lastName: e.target.value })} required /></div>
            </div>
            <div className="row">
              <div className="field"><label>Birth date</label><input type="date" value={form.birthDate} onChange={(e) => setForm({ ...form, birthDate: e.target.value })} /></div>
              <div className="field">
                <label>Flags</label>
                <div className="pill-row" style={{ paddingTop: '.4rem' }}>
                  <label style={{ fontWeight: 400 }}><input type="checkbox" style={{ width: 'auto', marginRight: 6 }} checked={form.pregnant} onChange={(e) => setForm({ ...form, pregnant: e.target.checked })} /> Pregnant</label>
                  <label style={{ fontWeight: 400 }}><input type="checkbox" style={{ width: 'auto', marginRight: 6 }} checked={form.preeclampsiaHx} onChange={(e) => setForm({ ...form, preeclampsiaHx: e.target.checked })} /> Preeclampsia Hx</label>
                  <label style={{ fontWeight: 400 }}><input type="checkbox" style={{ width: 'auto', marginRight: 6 }} checked={form.hypertension} onChange={(e) => setForm({ ...form, hypertension: e.target.checked })} /> Hypertension</label>
                </div>
              </div>
            </div>
            <button className="btn dark">Save Patient</button>
          </form>
        )}
      </div>

      <div className="card">
        <table>
          <thead><tr><th>MRN</th><th>Name</th><th>Sex</th><th>Status</th><th>Risk</th></tr></thead>
          <tbody>
            {filtered.map((p) => (
              <tr key={p.id}>
                <td>{p.mrn}</td>
                <td>{p.firstName} {p.lastName}</td>
                <td>{p.sex}</td>
                <td>{p.pregnant ? `Pregnant (${p.gestationalWeeks ?? '?'}w)` : 'Active'}</td>
                <td><span className={`badge ${p.riskLevel}`}>{p.riskLevel}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
