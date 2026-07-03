import { useEffect, useState } from 'react';
import { api, type Patient } from '../api';

export default function Dashboard() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => { api.patients().then((r) => setPatients(r.patients)).finally(() => setLoading(false)); }, []);

  if (loading) return <div className="spinner" />;

  const total = patients.length;
  const pregnant = patients.filter((p) => p.pregnant).length;
  const highRisk = patients.filter((p) => p.riskLevel === 'HIGH' || p.riskLevel === 'CRITICAL').length;
  const critical = patients.filter((p) => p.riskLevel === 'CRITICAL').length;

  return (
    <>
      <div className="page-head">
        <h2>Dashboard</h2>
        <p>Population overview and recent activity</p>
      </div>

      <div className="disclaimer">
        ⚠️ Demonstration platform with synthetic data and a transparent rules-based risk engine. Not a validated
        diagnostic device; not FDA-cleared. For clinical decision support research and education only.
      </div>

      <div className="grid cols-4" style={{ marginBottom: '1.25rem' }}>
        <div className="card stat"><div className="label">Total Patients</div><div className="value">{total}</div></div>
        <div className="card stat warning"><div className="label">Pregnant</div><div className="value">{pregnant}</div></div>
        <div className="card stat danger"><div className="label">High / Critical Risk</div><div className="value">{highRisk}</div></div>
        <div className="card stat danger"><div className="label">Critical Alerts</div><div className="value">{critical}</div></div>
      </div>

      <div className="card">
        <h3>Patient Registry</h3>
        <table>
          <thead><tr><th>MRN</th><th>Name</th><th>Status</th><th>Risk</th></tr></thead>
          <tbody>
            {patients.map((p) => (
              <tr key={p.id}>
                <td>{p.mrn}</td>
                <td>{p.firstName} {p.lastName}</td>
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
