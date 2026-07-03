import { useEffect, useState } from 'react';
import { api, type Patient } from '../api';

const SAMPLE_HL7 = [
  'MSH|^~\\&|EPIC|HOSP|CARDIOAI|CARDIOAI|20260703120000||ORU^R01|MSG00001|P|2.5',
  'PID|1||WC001^^^HOSP^MR||Johnson^Sarah||19880515|F',
  'OBR|1||ORD123|CVDPANEL^CVD Panel',
  'OBX|1|NM|8480-6^Systolic BP^LN||165|mm[Hg]|||||F',
  'OBX|2|NM|8462-4^Diastolic BP^LN||105|mm[Hg]|||||F',
  'OBX|3|NM|39156-5^BMI^LN||29.5|kg/m2|||||F',
].join('\n');

export default function Interop() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [patientId, setPatientId] = useState('');
  const [fhir, setFhir] = useState('');
  const [hl7, setHl7] = useState(SAMPLE_HL7);
  const [hl7Out, setHl7Out] = useState('');

  useEffect(() => { api.patients().then((r) => { setPatients(r.patients); setPatientId(r.patients[0]?.id ?? ''); }); }, []);

  async function loadFhir() {
    if (!patientId) return;
    const data = await api.fhirEverything(patientId);
    setFhir(JSON.stringify(data, null, 2));
  }
  async function parseHl7() {
    const data = await api.hl7Parse(hl7);
    setHl7Out(JSON.stringify(data, null, 2));
  }

  return (
    <>
      <div className="page-head"><h2>Interoperability</h2><p>FHIR R4 resources & HL7 v2 message parsing</p></div>

      <div className="card" style={{ marginBottom: '1.25rem' }}>
        <h3>FHIR R4 — Patient $everything</h3>
        <div className="row" style={{ alignItems: 'flex-end', marginBottom: '1rem' }}>
          <div style={{ flex: 3 }}>
            <label>Patient</label>
            <select value={patientId} onChange={(e) => setPatientId(e.target.value)}>
              {patients.map((p) => <option key={p.id} value={p.id}>{p.mrn} — {p.firstName} {p.lastName}</option>)}
            </select>
          </div>
          <button className="btn" style={{ flex: 0, whiteSpace: 'nowrap' }} onClick={loadFhir}>Fetch Bundle</button>
        </div>
        {fhir && <pre className="code">{fhir}</pre>}
      </div>

      <div className="card">
        <h3>HL7 v2 — parse ORU/ADT message</h3>
        <textarea rows={7} value={hl7} onChange={(e) => setHl7(e.target.value)} style={{ fontFamily: 'monospace', fontSize: '.8rem' }} />
        <button className="btn" style={{ marginTop: '.75rem' }} onClick={parseHl7}>Parse Message</button>
        {hl7Out && <pre className="code" style={{ marginTop: '1rem' }}>{hl7Out}</pre>}
      </div>
    </>
  );
}
