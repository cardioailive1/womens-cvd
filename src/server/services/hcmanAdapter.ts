/**
 * HCMAN adapter — connects the app to the HCMAN model trained on Mayo Clinic Platform.
 *
 * The model runs in a separate Python inference service (hcman/serve.py, ONNX Runtime) because
 * its preprocessing (EHR vocabulary, lab bins, ELM weights) ships with the exported model.
 * This file:
 *   1. builds the request from a Patient + Observations (FHIR-aligned, LOINC-coded),
 *   2. calls the service with a timeout,
 *   3. maps the six output heads onto the app's RiskResult so the UI, FHIR RiskAssessment
 *      export and human-in-the-loop review keep working unchanged,
 *   4. falls back to the transparent rules engine if the service is unset or fails.
 *
 * Explainability: HCMAN is not attributable point-by-point, so `factors` still come from the
 * rules engine and are labelled as such; the model contributes probabilities.
 */
import { env } from '../config/env.js';
import { assessRisk, type RiskInput, type RiskLevel, type RiskResult } from './riskEngine.js';

// Output order of the condition head — must match contract.json from hcman/export.py.
export const HCMAN_CONDITIONS = ['SCAD', 'PPCM', 'PREECLAMPSIA_CVD', 'TAKOTSUBO', 'CMD_INOCA', 'FMD'] as const;
const CONDITION_LABELS: Record<(typeof HCMAN_CONDITIONS)[number], string> = {
  SCAD: 'Spontaneous coronary artery dissection (SCAD)',
  PPCM: 'Peripartum cardiomyopathy (PPCM)',
  PREECLAMPSIA_CVD: 'Preeclampsia-related cardiovascular disease',
  TAKOTSUBO: 'Takotsubo (stress) cardiomyopathy',
  CMD_INOCA: 'Coronary microvascular dysfunction (INOCA)',
  FMD: 'Fibromuscular dysplasia (FMD)',
};
const TIERS: RiskLevel[] = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL'];

export interface PatientForModel {
  pregnant: boolean; gestationalWeeks: number | null; postpartumMonths: number | null;
  preeclampsiaHx: boolean; gestationalDmHx: boolean; hellpHx: boolean;
  pcos: boolean; autoimmune: boolean; migraineWithAura: boolean; fmd: boolean;
  // added by this integration (nullable / defaulted, so existing rows stay valid)
  menopausalStatus: string | null; ageAtMenopause: number | null; endometriosis: boolean;
  parity: number | null; breastCancerCardiotoxicTx: boolean; recentEmotionalStressor: boolean;
}
export interface ObservationForModel { loincCode: string; value: number; unit: string; effectiveAt: Date }

/** The 15 women-specific features of stream 4, keyed by the names the model was trained on.
 *  The service encodes them with the training-time ELM, so the app sends raw values. */
export function buildReproFeatures(p: PatientForModel) {
  const status = p.pregnant ? 'pregnant' : p.postpartumMonths != null && p.postpartumMonths < 6 ? 'postpartum' : 'not_pregnant';
  return {
    pregnancy_status: status,
    gestational_age_weeks: p.pregnant ? p.gestationalWeeks : null,
    gravida_para: p.parity,
    preeclampsia_history: p.preeclampsiaHx,
    gestational_diabetes: p.gestationalDmHx,
    hellp_history: p.hellpHx,
    menopausal_status: p.menopausalStatus,            // pre | peri | post | surgical | null (inferred from age)
    age_at_menopause: p.ageAtMenopause,
    pcos_diagnosis: p.pcos,
    endometriosis: p.endometriosis,
    autoimmune_disease_sle_ra: p.autoimmune,
    migraine_with_aura: p.migraineWithAura,
    fmd_history: p.fmd,
    breast_cancer_cardiotoxic_treatment: p.breastCancerCardiotoxicTx,
    emotional_stress_trigger_takotsubo: p.recentEmotionalStressor,
  };
}

export interface HcmanRequest {
  age: number | null;
  repro: ReturnType<typeof buildReproFeatures>;
  observations: { loinc: string; value: number; unit: string; daysBeforeIndex: number }[];
}

export interface HcmanPrediction {
  modelVersion: string;
  conditions: Record<string, number>;       // per-condition probability
  mace12m: number;
  postpartumCvd: number | null;             // null when the patient is not postpartum
  riskTier: number[];                       // [LOW, MODERATE, HIGH, CRITICAL]
  atypicalPresentation: number;
  fmdScreeningTrigger: number;
  thresholds: Record<string, number>;       // operating points chosen on MCP validation data
  missingModalities: string[];              // e.g. ["ecg", "echo"] — shown to the clinician
}

export function buildRequest(age: number | undefined, p: PatientForModel, obs: ObservationForModel[], now = new Date()): HcmanRequest {
  const dayMs = 86_400_000;
  return {
    age: age ?? null,
    repro: buildReproFeatures(p),
    observations: obs
      .map((o) => ({ loinc: o.loincCode, value: o.value, unit: o.unit,
                     daysBeforeIndex: Math.floor((now.getTime() - o.effectiveAt.getTime()) / dayMs) }))
      .filter((o) => o.daysBeforeIndex >= 0 && o.daysBeforeIndex <= 365),
  };
}

/** Map HCMAN outputs onto the app's RiskResult contract. */
export function toRiskResult(pred: HcmanPrediction, rulesInput: RiskInput): RiskResult {
  const rules = assessRisk(rulesInput);                      // explainability factors only
  const tierIdx = pred.riskTier.indexOf(Math.max(...pred.riskTier));
  const riskLevel = TIERS[tierIdx] ?? 'LOW';
  const ranked = HCMAN_CONDITIONS
    .map((c) => ({ c, p: pred.conditions[c] ?? 0, thr: pred.thresholds[c] ?? 0.5 }))
    .sort((a, b) => b.p - a.p);
  const flagged = ranked.filter((r) => r.p >= r.thr);
  const top = flagged[0] ?? ranked[0];

  const recommendations: string[] = [];
  if (pred.fmdScreeningTrigger >= (pred.thresholds.fmd_trigger ?? 0.5))
    recommendations.push('Model flags SCAD pattern: consider head-to-pelvis CTA/MRA screening for systemic FMD.');
  if (pred.atypicalPresentation >= (pred.thresholds.atypical ?? 0.5))
    recommendations.push('Atypical ischemic presentation pattern detected (fatigue, nausea, jaw/back/epigastric pain): do not rule out ACS on absent chest pain.');
  if ((pred.conditions.PPCM ?? 0) >= (pred.thresholds.PPCM ?? 0.5))
    recommendations.push('PPCM probability above threshold: echocardiogram (LVEF) and NT-proBNP.');
  if (pred.postpartumCvd != null && pred.postpartumCvd >= (pred.thresholds.postpartum ?? 0.5))
    recommendations.push('Elevated 12-month postpartum CVD risk: cardiology follow-up within 6–12 weeks.');
  if (pred.missingModalities.length)
    recommendations.push(`Model ran without: ${pred.missingModalities.join(', ')}. Adding these inputs may change the result.`);
  recommendations.push(...rules.recommendations);

  return {
    riskScore: Math.round(pred.mace12m * 100),
    riskLevel,
    primaryDx: flagged.length ? CONDITION_LABELS[top.c] : rules.primaryDx,
    confidence: Math.round(top.p * 100),
    // Model output is advisory: CRITICAL/HIGH stay human-in-the-loop (TIER_2), never autonomous.
    automationTier: riskLevel === 'CRITICAL' || riskLevel === 'HIGH' ? 'TIER_2' : 'TIER_1',
    differentials: ranked.map((r) => ({ label: CONDITION_LABELS[r.c], probability: +(r.p * 100).toFixed(1) })),
    factors: {
      critical: rules.factors.critical.map((f) => `${f} (rules engine)`),
      moderate: rules.factors.moderate.map((f) => `${f} (rules engine)`),
      protective: rules.factors.protective.map((f) => `${f} (rules engine)`),
    },
    // HCMAN predicts 12-month MACE only; longer horizons keep the rules engine's estimate.
    prognosis: { ...rules.prognosis, oneYear: +(pred.mace12m * 100).toFixed(1) },
    recommendations: [...new Set(recommendations)],
  };
}

export async function callHcman(req: HcmanRequest): Promise<HcmanPrediction> {
  const url = env.HCMAN_INFERENCE_URL;
  if (!url) throw new Error('HCMAN_INFERENCE_URL not configured');
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), env.HCMAN_TIMEOUT_MS);
  try {
    const r = await fetch(`${url.replace(/\/$/, '')}/predict`, {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify(req), signal: ctrl.signal,
    });
    if (!r.ok) throw new Error(`HCMAN service HTTP ${r.status}`);
    return (await r.json()) as HcmanPrediction;
  } finally { clearTimeout(t); }
}

export interface EngineResult { result: RiskResult; engine: 'rules' | 'hcman'; modelVersion: string | null; fallbackReason?: string }

/** Engine selection: ASSESSMENT_ENGINE=hcman uses the model and falls back to rules on error. */
export async function runAssessment(rulesInput: RiskInput, p: PatientForModel, obs: ObservationForModel[]): Promise<EngineResult> {
  if (env.ASSESSMENT_ENGINE !== 'hcman') return { result: assessRisk(rulesInput), engine: 'rules', modelVersion: null };
  try {
    const pred = await callHcman(buildRequest(rulesInput.age, p, obs));
    return { result: toRiskResult(pred, rulesInput), engine: 'hcman', modelVersion: pred.modelVersion };
  } catch (e) {
    return { result: assessRisk(rulesInput), engine: 'rules', modelVersion: null,
             fallbackReason: e instanceof Error ? e.message : String(e) };
  }
}
