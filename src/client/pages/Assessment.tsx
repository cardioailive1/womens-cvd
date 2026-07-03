import { useEffect, useState } from 'react';
import { api, type Patient, type AssessmentResult } from '../api';

export default function Assessment() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [patientId, setPatientId] = useState('');
  const [sbp, setSbp] = useState(145);
  const [bmi, setBmi] = useState(29.5);
  const [age, setAge] = useState(35);
  const [result, setResult] = useState<AssessmentResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState('');
  const [saved, setSaved] = useState('');

  useEffect(() => { api.patients().then((r) => { setPatients(r.patients); setPatientId(r.patients[0]?.id ?? ''); }); }, []);

  async function run() {
    if (!patientId) return;
    setBusy(true); setResult(null); setSaved('');
    try { const r = await api.runAssessment(patientId, { age, systolicBp: sbp, bmi }); setResult(r.assessment); }
    finally { setBusy(false); }
  }

  async function confirm(confirmIt: boolean) {
    if (!result) return;
    await api.reviewAssessment(result.id, note || '(no note)', confirmIt);
    setSaved(confirmIt ? 'Assessment confirmed and recorded.' : 'Physician note saved (draft).');
  }

  return (
    <>
      <div className="page-head"><h2>Risk Assessment</h2><p>Transparent, explainable rules-based scoring — clinician-in-the-loop</p></div>

      <div className="grid cols-2">
        <div className="card">
          <h3>Inputs</h3>
          <div className="field">
            <label>Patient</label>
            <select value={patientId} onChange={(e) => setPatientId(e.target.value)}>
              {patients.map((p) => <option key={p.id} value={p.id}>{p.mrn} — {p.firstName} {p.lastName}</option>)}
            </select>
          </div>
          <div className="row">
            <div className="field"><label>Age: {age}</label><input type="range" min={18} max={80} value={age} onChange={(e) => setAge(+e.target.value)} /></div>
            <div className="field"><label>Systolic BP: {sbp}</label><input type="range" min={90} max={200} value={sbp} onChange={(e) => setSbp(+e.target.value)} /></div>
            <div className="field"><label>BMI: {bmi}</label><input type="range" min={16} max={45} step={0.5} value={bmi} onChange={(e) => setBmi(+e.target.value)} /></div>
          </div>
          <button className="btn dark" onClick={run} disabled={busy || !patientId}>{busy ? 'Analyzing…' : '🔬 Run Assessment'}</button>
        </div>

        <div className="card">
          <h3>Result</h3>
          {!result && <p style={{ color: 'var(--muted)' }}>Run an assessment to see the explainable output.</p>}
          {result && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '.6rem' }}>
                <strong style={{ color: 'var(--primary-dark)' }}>{result.primaryDx}</strong>
                <span className={`badge ${result.riskLevel}`}>{result.riskLevel}</span>
              </div>
              <div style={{ fontSize: '.85rem', color: 'var(--muted)', marginBottom: '.4rem' }}>
                Risk score {result.riskScore}/100 · confidence {result.confidence}% · {result.automationTier.replace('_', ' ')}
              </div>
              <div className="meter" style={{ marginBottom: '1rem' }}><span style={{ width: `${result.riskScore}%` }} /></div>

              <div className="grid cols-4" style={{ gap: '.5rem', marginBottom: '1rem' }}>
                {(['oneYear', 'fiveYear', 'tenYear', 'lifetime'] as const).map((k, i) => (
                  <div key={k} className="card" style={{ padding: '.6rem', textAlign: 'center', boxShadow: 'none' }}>
                    <div style={{ fontSize: '.7rem', color: 'var(--muted)' }}>{['1-yr', '5-yr', '10-yr', 'Lifetime'][i]}</div>
                    <div style={{ fontWeight: 700, color: 'var(--danger)' }}>{result.prognosis[k]}%</div>
                  </div>
                ))}
              </div>

              <details open>
                <summary style={{ cursor: 'pointer', fontWeight: 600, marginBottom: '.5rem' }}>Contributing factors (fully attributable)</summary>
                {result.factors.critical.map((f) => <div key={f} className="factor crit">🔴 {f}</div>)}
                {result.factors.moderate.map((f) => <div key={f} className="factor mod">🟠 {f}</div>)}
                {result.factors.protective.map((f) => <div key={f} className="factor prot">🟢 {f}</div>)}
              </details>

              <h3 style={{ marginTop: '1rem' }}>Recommendations</h3>
              <ul style={{ paddingLeft: '1.1rem', fontSize: '.88rem' }}>
                {result.recommendations.map((r) => <li key={r} style={{ marginBottom: '.3rem' }}>{r}</li>)}
              </ul>

              <div style={{ marginTop: '1rem', borderTop: '1px solid var(--border)', paddingTop: '1rem' }}>
                <label>Physician note</label>
                <textarea rows={2} value={note} onChange={(e) => setNote(e.target.value)} placeholder="Clinical judgment / modifications…" />
                <div className="row" style={{ marginTop: '.6rem' }}>
                  <button className="btn ghost" onClick={() => confirm(false)}>Save note</button>
                  <button className="btn success" onClick={() => confirm(true)}>✓ Confirm assessment</button>
                </div>
                {saved && <div className="badge success" style={{ marginTop: '.6rem' }}>{saved}</div>}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
