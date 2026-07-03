import { useEffect, useState } from 'react';
import { api, type AlertDto } from '../api';

type Severity = 'CRITICAL' | 'HIGH' | 'INFO';

const META: Record<Severity, { label: string; tone: 'danger' | 'warning' | 'info'; color: string }> = {
  CRITICAL: { label: 'Critical', tone: 'danger', color: 'var(--danger)' },
  HIGH: { label: 'High Priority', tone: 'warning', color: 'var(--warning)' },
  INFO: { label: 'Information', tone: 'info', color: 'var(--info)' },
};

export default function Alerts() {
  const [items, setItems] = useState<AlertDto[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');
  const [done, setDone] = useState<Record<string, string>>({});

  useEffect(() => {
    api.alerts()
      .then((r) => setItems(r.alerts))
      .catch((e) => setErr((e as Error).message))
      .finally(() => setLoading(false));
  }, []);

  function act(a: AlertDto) {
    const verb = a.action === 'Acknowledge' ? 'Acknowledged' : a.action === 'Schedule' ? 'Scheduled' : a.action === 'Review' ? 'Reviewed' : 'Viewed';
    setDone((d) => ({ ...d, [a.id]: `${verb} · ${new Date().toLocaleTimeString()}` }));
    if (a.action === 'Acknowledge') setTimeout(() => setItems((list) => list.filter((x) => x.id !== a.id)), 900);
  }

  if (loading) return <div className="spinner" />;

  const groups: Severity[] = ['CRITICAL', 'HIGH', 'INFO'];

  return (
    <>
      <div className="page-head">
        <h2>Clinical Alerts</h2>
        <p>Derived in real time from current patient risk levels and flags</p>
      </div>

      {err && <div className="section-alert warning">Couldn't load alerts: {err}</div>}

      {groups.map((g) => {
        const list = items.filter((i) => i.severity === g);
        if (!list.length) return null;
        const m = META[g];
        return (
          <div key={g} className="card" style={{ marginBottom: '1.25rem' }}>
            <h3 style={{ color: m.color }}>{m.label} ({list.length})</h3>
            {list.map((a) => (
              <div key={a.id} className={`section-alert ${m.tone}`}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                  <div style={{ flex: 1, minWidth: 240 }}>
                    <strong>{g}: {a.patient} ({a.mrn})</strong><br />
                    {a.message}<br />
                    <small>{a.when}</small>
                  </div>
                  <div style={{ textAlign: 'right' }}>
                    {done[a.id]
                      ? <span className="pillbadge success">{done[a.id]}</span>
                      : <button className={a.action === 'Acknowledge' ? 'btn' : 'btn ghost'}
                          style={a.action === 'Acknowledge' ? { background: 'var(--danger)' } : undefined}
                          onClick={() => act(a)}>{a.action}</button>}
                  </div>
                </div>
              </div>
            ))}
          </div>
        );
      })}

      {!err && items.length === 0 && <div className="card"><p style={{ color: 'var(--muted)' }}>No active alerts — all patients are within expected risk ranges.</p></div>}
    </>
  );
}
