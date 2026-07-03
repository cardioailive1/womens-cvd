import { useEffect, useState } from 'react';
import { api, type ReportSummary } from '../api';

export default function Reports() {
  const [type, setType] = useState('Patient Summary');
  const [days, setDays] = useState(30);
  const [summary, setSummary] = useState<ReportSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState('');
  const [flash, setFlash] = useState('');

  function load(d: number) {
    setLoading(true); setErr('');
    api.reportSummary(d)
      .then(setSummary)
      .catch((e) => setErr((e as Error).message))
      .finally(() => setLoading(false));
  }
  useEffect(() => { load(days); /* eslint-disable-next-line */ }, []);

  function onPeriodChange(d: number) { setDays(d); load(d); }
  function generate() {
    setFlash(`Prepared "${type}" report over the last ${days} days from ${summary?.stats.totalAssessments ?? 0} assessment record(s). In a full deployment this would export a PDF/CSV.`);
  }

  const maxCount = Math.max(1, ...(summary?.distribution.map((d) => d.count) ?? [1]));

  return (
    <>
      <div className="page-head">
        <h2>Reports</h2>
        <p>Computed from live assessment records</p>
      </div>

      {err && <div className="section-alert warning">Couldn't load report data: {err}</div>}

      <div className="card" style={{ marginBottom: '1.25rem' }}>
        <h3>Generate Report</h3>
        <div className="row">
          <div className="field">
            <label>Report type</label>
            <select value={type} onChange={(e) => setType(e.target.value)}>
              {['Patient Summary', 'Risk Assessment', 'Quality Metrics', 'Pregnancy Outcomes', 'Population Health'].map((o) => <option key={o}>{o}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Time period</label>
            <select value={days} onChange={(e) => onPeriodChange(Number(e.target.value))}>
              <option value={7}>Last 7 Days</option>
              <option value={30}>Last 30 Days</option>
              <option value={90}>Last 3 Months</option>
              <option value={365}>Last Year</option>
            </select>
          </div>
        </div>
        <button className="btn dark" onClick={generate}>📊 Generate Report</button>
        {flash && <div className="section-alert info" style={{ marginTop: '1rem' }}>{flash}</div>}
      </div>

      {loading && <div className="spinner" />}

      {summary && (
        <>
          <div className="card" style={{ marginBottom: '1.25rem' }}>
            <h3>Platform Statistics (last {summary.period.days} days)</h3>
            <div className="grid cols-4">
              <div className="card stat"><div className="label">Total Assessments</div><div className="value">{summary.stats.totalAssessments}</div></div>
              <div className="card stat danger"><div className="label">High-Risk Identified</div><div className="value">{summary.stats.highRiskIdentified}</div></div>
              <div className="card stat warning"><div className="label">Critical</div><div className="value">{summary.stats.criticalCount}</div></div>
              <div className="card stat success"><div className="label">Physician-Confirmed</div><div className="value">{summary.stats.confirmedCount}</div></div>
            </div>
          </div>

          <div className="card">
            <h3>Assessments by Primary Impression</h3>
            {summary.distribution.length === 0 ? (
              <p style={{ color: 'var(--muted)' }}>No assessments recorded in this period yet. Run an assessment to populate this chart.</p>
            ) : (
              <>
                <div className="chart-bars">
                  {summary.distribution.map((d) => (
                    <div key={d.label} className="bar" style={{ height: `${Math.round((d.count / maxCount) * 100)}%` }} title={`${d.label}: ${d.count}`}>
                      <span className="val">{d.count}</span>
                      <span className="lbl">{d.label}</span>
                    </div>
                  ))}
                </div>
                <p style={{ textAlign: 'center', color: 'var(--muted)', fontSize: '.85rem', marginTop: '2.25rem' }}>
                  Distribution of assessment primary impressions across {summary.stats.totalPatients} active patient(s)
                </p>
              </>
            )}
          </div>
        </>
      )}
    </>
  );
}
