import { useMemo, useState } from 'react';

type Sev = 'danger' | 'warning' | 'info' | 'success';
interface Section { h: string; note?: Sev; body: React.ReactNode }
interface Condition {
  id: string; title: string; accent: string;
  badges: { text: string; kind: 'danger' | 'warning' | 'info' | 'primary' }[];
  sections: Section[];
}

const CONDITIONS: Condition[] = [
  {
    id: 'scad', title: 'SCAD — Spontaneous Coronary Artery Dissection', accent: 'var(--danger)',
    badges: [{ text: 'High Risk', kind: 'danger' }, { text: 'Women-Specific', kind: 'primary' }, { text: 'Age 30–60', kind: 'info' }],
    sections: [
      { h: '⚠️ Overview', note: 'danger', body: <p>A tear in the wall of a coronary artery that primarily affects young, otherwise healthy women. Accounts for 25–35% of acute coronary syndromes in women under 50.</p> },
      { h: '📋 Clinical Presentation', body: <ul><li><strong>Typical symptoms:</strong> Chest pain, shortness of breath, arm/jaw pain</li><li><strong>Atypical:</strong> Fatigue, nausea, back pain, epigastric discomfort</li><li><strong>Timing:</strong> Often postpartum (peripartum SCAD) or during emotional/physical stress</li><li><strong>Risk factors:</strong> Recent pregnancy, fibromuscular dysplasia, connective tissue disorders</li></ul> },
      { h: '🔬 Diagnostic Criteria', note: 'info', body: <><strong>Gold standard:</strong> coronary angiography showing contrast staining of the arterial wall, multiple radiolucent lumens, long smooth stenosis (&gt;20mm), abrupt change in vessel caliber. <strong>Alternative:</strong> OCT or IVUS for confirmation.</> },
      { h: '🧪 Laboratory Findings', body: <ul><li><strong>Troponin:</strong> elevated (may be lower than atherosclerotic MI)</li><li><strong>ECG:</strong> ST elevation/depression, T-wave abnormalities</li><li><strong>Echo:</strong> regional wall motion abnormalities</li></ul> },
      { h: '💊 Management', note: 'success', body: <><strong>Conservative approach preferred:</strong><ul><li>Medical management: beta-blockers, aspirin, consider ACE inhibitors</li><li>Avoid thrombolytics (can extend dissection)</li><li>PCI only if ongoing ischemia or hemodynamic instability</li><li>Cardiac rehab with modified protocols; annual imaging follow-up</li></ul></> },
      { h: '🚨 Clinical Pearls', note: 'warning', body: <ul><li>Consider SCAD in any woman &lt;60 with ACS and minimal risk factors</li><li>Screen for fibromuscular dysplasia (30–50% association)</li><li>Pregnancy counseling essential — increased recurrence risk</li><li>Psychological support important — high rates of PTSD/anxiety</li></ul> },
    ],
  },
  {
    id: 'takotsubo', title: 'Takotsubo (Stress) Cardiomyopathy', accent: 'var(--warning)',
    badges: [{ text: 'Moderate Risk', kind: 'warning' }, { text: 'Women-Specific', kind: 'primary' }, { text: 'Postmenopausal', kind: 'info' }],
    sections: [
      { h: '⚠️ Overview', note: 'warning', body: <p>"Broken heart syndrome" — temporary left ventricular dysfunction following emotional or physical stress. 90% of cases occur in postmenopausal women.</p> },
      { h: '📋 Clinical Presentation', body: <ul><li><strong>Trigger:</strong> emotional (grief, fear, anger) or physical stress (surgery, acute illness)</li><li><strong>Symptoms:</strong> chest pain, dyspnea, syncope — mimics acute MI</li><li><strong>Timing:</strong> acute onset, usually within hours of the stressor</li></ul> },
      { h: '🔬 Diagnostic Criteria (Mayo Clinic)', note: 'info', body: <><strong>All four must be present:</strong><ol><li>Transient LV wall motion abnormalities beyond a single coronary distribution</li><li>Absence of obstructive coronary disease or acute plaque rupture</li><li>New ECG abnormalities or modest troponin elevation</li><li>Absence of pheochromocytoma or myocarditis</li></ol></> },
      { h: '🧪 Laboratory & Imaging', body: <ul><li><strong>Troponin:</strong> mildly–moderately elevated (lower than STEMI)</li><li><strong>BNP/NT-proBNP:</strong> markedly elevated, disproportionate to troponin</li><li><strong>Echo:</strong> apical ballooning (classic), or mid-ventricular/basal variants</li><li><strong>Cardiac MRI:</strong> myocardial edema without late gadolinium enhancement</li></ul> },
      { h: '💊 Management', note: 'success', body: <><strong>Acute:</strong> supportive care, ACE-I/ARB for LV dysfunction, cautious beta-blockers, anticoagulation if LV thrombus. <strong>Long-term:</strong> most recover fully in weeks–months; recurrence ~10% at 5 years.</> },
      { h: '🚨 Clinical Pearls', note: 'warning', body: <ul><li>Disproportionately high BNP relative to troponin suggests Takotsubo</li><li>Monitor for cardiogenic shock (5–10%), LV thrombus, arrhythmias</li><li>Distinguish from myocarditis via cardiac MRI</li></ul> },
    ],
  },
  {
    id: 'preeclampsia', title: 'Preeclampsia-Related Cardiovascular Disease', accent: 'var(--danger)',
    badges: [{ text: 'Critical Risk', kind: 'danger' }, { text: 'Pregnancy', kind: 'primary' }, { text: 'Age 20–45', kind: 'info' }],
    sections: [
      { h: '⚠️ Overview', note: 'danger', body: <p>New-onset hypertension and proteinuria or end-organ dysfunction after 20 weeks gestation. Increases lifetime CVD risk 2–4 fold.</p> },
      { h: '📋 Clinical Presentation', body: <ul><li><strong>Hypertension:</strong> BP ≥140/90 on two occasions, 4+ hours apart</li><li><strong>Severe features:</strong> BP ≥160/110, severe headache, visual changes, RUQ pain</li><li><strong>Risk factors:</strong> first pregnancy, multiple gestation, obesity, chronic hypertension, diabetes</li></ul> },
      { h: '🔬 Diagnostic Criteria (ACOG 2019)', note: 'info', body: <><strong>Hypertension PLUS one or more:</strong><ul><li>Proteinuria ≥300 mg/24hr or protein/creatinine ratio ≥0.3</li><li>Thrombocytopenia (platelets &lt;100,000/µL)</li><li>Renal insufficiency (creatinine &gt;1.1 mg/dL or doubling)</li><li>Elevated transaminases (2× normal)</li><li>Pulmonary edema; cerebral or visual symptoms</li></ul></> },
      { h: '🧪 Laboratory Monitoring', body: <ul><li><strong>CBC:</strong> platelet count, hemolysis markers</li><li><strong>LFTs:</strong> AST, ALT, LDH (HELLP screening)</li><li><strong>Renal:</strong> creatinine, BUN, uric acid</li><li><strong>Urinalysis:</strong> protein/creatinine ratio</li></ul> },
      { h: '💊 Management', note: 'success', body: <><strong>Antepartum:</strong> labetalol, nifedipine, methyldopa (avoid ACE-I/ARBs); MgSO4 for seizure prophylaxis if severe; delivery timing 34–37 weeks. <strong>Postpartum:</strong> continue MgSO4 24–48h if severe; BP monitoring up to 2 weeks; long-term CVD risk counseling.</> },
      { h: '🚨 Clinical Pearls', note: 'danger', body: <ul><li><strong>HELLP:</strong> hemolysis, elevated liver enzymes, low platelets — emergency</li><li><strong>Eclampsia:</strong> new-onset seizures — give MgSO4 immediately</li><li>2–4× increased lifetime risk of MI, stroke, heart failure</li><li>Low-dose aspirin (81–162mg) from 12 weeks reduces recurrence</li><li>Cardiology referral within 6–12 weeks postpartum</li></ul> },
    ],
  },
  {
    id: 'ppcm', title: 'Peripartum Cardiomyopathy (PPCM)', accent: 'var(--danger)',
    badges: [{ text: 'Critical Risk', kind: 'danger' }, { text: 'Pregnancy/Postpartum', kind: 'primary' }, { text: 'Age 25–40', kind: 'info' }],
    sections: [
      { h: '⚠️ Overview', note: 'danger', body: <p>Heart failure in the last month of pregnancy or within 5 months postpartum, without other identifiable cause. Incidence: 1 in 1,000–4,000 live births in the US.</p> },
      { h: '📋 Clinical Presentation', body: <ul><li><strong>Symptoms:</strong> dyspnea, orthopnea, PND, fatigue, peripheral edema, palpitations</li><li><strong>Timing:</strong> most in the first month postpartum</li><li><strong>Risk factors:</strong> multiparity, advanced maternal age (&gt;30), African descent, multiple gestation, preeclampsia</li></ul> },
      { h: '🔬 Diagnostic Criteria (ESC 2018)', note: 'info', body: <><strong>All four must be met:</strong><ol><li>Heart failure in last month of pregnancy or within 5 months postpartum</li><li>LVEF &lt;45%</li><li>LV end-diastolic dimension &gt;2.7 cm/m² BSA</li><li>No other identifiable cause of heart failure</li></ol></> },
      { h: '🧪 Laboratory & Imaging', body: <ul><li><strong>BNP/NT-proBNP:</strong> markedly elevated</li><li><strong>Echo:</strong> reduced LVEF (&lt;45%), LV dilation, possible LV thrombus</li><li><strong>Cardiac MRI:</strong> consider to assess for myocarditis</li></ul> },
      { h: '💊 Management', note: 'success', body: <><strong>Antepartum:</strong> furosemide, beta-blockers (metoprolol/carvedilol), hydralazine+nitrates; avoid ACE-I/ARBs/aldosterone antagonists. <strong>Postpartum:</strong> standard HF therapy; consider bromocriptine; anticoagulation if LVEF &lt;30% or thrombus; avoid future pregnancy until recovery.</> },
      { h: '📊 Prognosis & Recovery', note: 'warning', body: <ul><li>Full recovery (LVEF &gt;50%): 50–60% within 6 months</li><li>Persistent dysfunction: 25–30%; mortality 5–15%</li><li>Recurrence risk very high (30–50%) in subsequent pregnancies</li></ul> },
    ],
  },
  {
    id: 'cmd', title: 'Coronary Microvascular Disease (CMD)', accent: 'var(--primary)',
    badges: [{ text: 'Moderate Risk', kind: 'warning' }, { text: 'More Common in Women', kind: 'primary' }, { text: 'Age 40–70', kind: 'info' }],
    sections: [
      { h: '⚠️ Overview', note: 'info', body: <p>Chest pain with evidence of ischemia but no obstructive CAD on angiography. Accounts for up to 50% of women with angina and non-obstructive coronary arteries.</p> },
      { h: '📋 Clinical Presentation', body: <ul><li><strong>Symptoms:</strong> typical angina, but may be prolonged or occur at rest</li><li><strong>Pattern:</strong> often triggered by emotional stress, not just exertion</li><li><strong>Risk factors:</strong> diabetes, metabolic syndrome, estrogen deficiency, inflammation</li></ul> },
      { h: '🔬 Diagnostic Approach', note: 'info', body: <><strong>Diagnosis of exclusion:</strong> angina + evidence of ischemia + no obstructive CAD (&lt;50% stenosis) + objective CMD on invasive testing (CFR &lt;2.0, IMR &gt;25) or PET showing impaired flow reserve.</> },
      { h: '💊 Management', note: 'success', body: <><strong>Pharmacologic:</strong> beta-blockers, calcium channel blockers, ACE-I/ARBs, statins (even with normal cholesterol), ranolazine, aspirin. <strong>Lifestyle:</strong> supervised cardiac rehab, weight loss, diabetes control, stress management.</> },
      { h: '🚨 Clinical Pearls', note: 'info', body: <ul><li>CMD is not benign — associated with increased MI, HF, and mortality</li><li>Often coexists with vasospasm — consider acetylcholine testing</li><li>Mental health screening important (high depression/anxiety rates)</li></ul> },
    ],
  },
  {
    id: 'fmd', title: 'Fibromuscular Dysplasia (FMD)', accent: 'var(--primary)',
    badges: [{ text: 'Moderate Risk', kind: 'warning' }, { text: '90% Women', kind: 'primary' }, { text: 'Age 40–60', kind: 'info' }],
    sections: [
      { h: '⚠️ Overview', note: 'info', body: <p>Non-atherosclerotic, non-inflammatory vascular disease affecting medium arteries (commonly renal and cerebrovascular). 90% of patients are women; strong association with SCAD (30–50% of SCAD patients have FMD).</p> },
      { h: '📋 Clinical Presentation', body: <ul><li><strong>Renal FMD:</strong> hypertension (young onset, resistant), renal artery stenosis</li><li><strong>Cerebrovascular:</strong> headache, pulsatile tinnitus, TIA/stroke, dissection</li><li><strong>Coronary:</strong> often asymptomatic, may present with SCAD or angina</li></ul> },
      { h: '🔬 Diagnostic Imaging', note: 'info', body: <><strong>"String of beads"</strong> multifocal pattern (85–90%) or focal stenosis, on CT/MR/catheter angiography. Screen all SCAD patients head-to-pelvis; evaluate young women with hypertension or stroke.</> },
      { h: '💊 Management', note: 'success', body: <><strong>Medical:</strong> antiplatelet (aspirin), ACE-I/ARB or CCB for hypertension, consider statin. <strong>Interventional:</strong> angioplasty for symptomatic renal FMD; avoid stenting; surgery rarely needed.</> },
      { h: '🚨 Clinical Pearls', note: 'info', body: <ul><li>Screen all SCAD patients for systemic FMD</li><li>If found in one vascular bed, image others (systemic disease)</li><li>Pregnancy: increased dissection risk, monitor closely</li></ul> },
    ],
  },
];

export default function Diagnostics() {
  const [open, setOpen] = useState<Record<string, boolean>>({});
  const [query, setQuery] = useState('');

  const filtered = useMemo(() => {
    const q = query.toLowerCase();
    if (!q) return CONDITIONS;
    return CONDITIONS.filter((c) =>
      (c.title + ' ' + c.badges.map((b) => b.text).join(' ')).toLowerCase().includes(q));
  }, [query]);

  const toggle = (id: string) => setOpen((o) => ({ ...o, [id]: !o[id] }));
  const setAll = (v: boolean) => setOpen(Object.fromEntries(CONDITIONS.map((c) => [c.id, v])));

  return (
    <>
      <div className="page-head">
        <h2>Diagnostics Guide</h2>
        <p>Diagnostic criteria and clinical guidance for women-specific cardiovascular conditions</p>
      </div>

      <div className="disclaimer">
        Educational reference summarizing published criteria (ACOG / ESC / Mayo). Not a substitute for clinical
        judgment or primary sources.
      </div>

      <div className="card" style={{ marginBottom: '1.25rem' }}>
        <div style={{ display: 'flex', gap: '.5rem', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
          <button className="btn ghost" onClick={() => setAll(true)}>Expand all</button>
          <button className="btn ghost" onClick={() => setAll(false)}>Collapse all</button>
        </div>
        <div className="quicknav">
          <strong style={{ color: 'var(--primary-dark)' }}>Quick navigation:</strong>
          <div style={{ marginTop: '.5rem' }}>
            {CONDITIONS.map((c) => (
              <button key={c.id} className="btn ghost" style={{ fontSize: '.8rem', padding: '.35rem .7rem' }}
                onClick={() => { setOpen((o) => ({ ...o, [c.id]: true })); document.getElementById(`cond-${c.id}`)?.scrollIntoView({ behavior: 'smooth', block: 'start' }); }}>
                {c.id.toUpperCase()}
              </button>
            ))}
          </div>
        </div>
        <div style={{ marginTop: '1rem' }}>
          <input placeholder="🔍 Search conditions…" value={query} onChange={(e) => setQuery(e.target.value)} style={{ maxWidth: 560 }} />
        </div>
      </div>

      {filtered.map((c) => (
        <div key={c.id} id={`cond-${c.id}`} className="diag-card" style={{ borderLeftColor: c.accent }}>
          <div className="diag-head">
            <div>
              <h3>{c.title}</h3>
              <div>{c.badges.map((b) => <span key={b.text} className={`pillbadge ${b.kind}`}>{b.text}</span>)}</div>
            </div>
            <button className="btn ghost" onClick={() => toggle(c.id)} style={{ whiteSpace: 'nowrap' }}>
              {open[c.id] ? 'Hide ▲' : 'Details ▼'}
            </button>
          </div>
          {open[c.id] && (
            <div className="diag-body">
              {c.sections.map((s, i) => (
                <div key={i}>
                  <h4>{s.h}</h4>
                  {s.note
                    ? <div className={`section-alert ${s.note}`}>{s.body}</div>
                    : <div style={{ lineHeight: 1.9 }}>{s.body}</div>}
                </div>
              ))}
            </div>
          )}
        </div>
      ))}
      {filtered.length === 0 && <div className="card"><p style={{ color: 'var(--muted)' }}>No conditions match "{query}".</p></div>}
    </>
  );
}
