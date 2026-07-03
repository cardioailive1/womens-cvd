/**
 * Women's CVD risk engine — TRANSPARENT, RULES-BASED, EXPLAINABLE.
 *
 * IMPORTANT: This is a deterministic decision-support scaffold, NOT a validated
 * diagnostic model and NOT an FDA-cleared medical device. Every point of the score
 * is attributable to a named clinical factor (no black box). Clinicians remain the
 * decision-makers; outputs are advisory (see automationTier).
 *
 * Weightings are illustrative and align with the emphases of ACOG/ESC/AHA guidance
 * on sex-specific cardiovascular risk. Replace with a validated, locally-calibrated
 * model before any real clinical use.
 */

export type RiskLevel = 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';

export interface RiskInput {
  age?: number;
  systolicBp?: number;
  bmi?: number;
  pregnant?: boolean;
  gestationalWeeks?: number | null;
  postpartumMonths?: number | null;
  preeclampsiaHx?: boolean;
  gestationalHtnHx?: boolean;
  gestationalDmHx?: boolean;
  hellpHx?: boolean;
  hypertension?: boolean;
  diabetes?: boolean;
  hyperlipidemia?: boolean;
  smoker?: boolean;
  familyHxEarlyCvd?: boolean;
  pcos?: boolean;
  autoimmune?: boolean;
  migraineWithAura?: boolean;
  fmd?: boolean;
}

interface Weighted { label: string; points: number; severity: 'critical' | 'moderate' | 'protective' }

export interface RiskResult {
  riskScore: number;               // 0..100
  riskLevel: RiskLevel;
  primaryDx: string;
  confidence: number;              // 0..100 (explainability-derived, not probabilistic AI)
  automationTier: 'TIER_1' | 'TIER_2' | 'TIER_3';
  differentials: { label: string; probability: number }[];
  factors: { critical: string[]; moderate: string[]; protective: string[] };
  prognosis: { oneYear: number; fiveYear: number; tenYear: number; lifetime: number };
  recommendations: string[];
}

export function assessRisk(input: RiskInput): RiskResult {
  const w: Weighted[] = [];

  // ---- Critical, women-specific & acute drivers ----
  if (input.pregnant && input.preeclampsiaHx) w.push({ label: 'Current pregnancy with prior preeclampsia', points: 22, severity: 'critical' });
  else if (input.preeclampsiaHx) w.push({ label: 'History of preeclampsia (2–4× lifetime CVD risk)', points: 14, severity: 'critical' });
  if ((input.systolicBp ?? 0) >= 160) w.push({ label: `Severe hypertension (SBP ${input.systolicBp})`, points: 20, severity: 'critical' });
  else if ((input.systolicBp ?? 0) >= 140) w.push({ label: `Stage 2 hypertension (SBP ${input.systolicBp})`, points: 12, severity: 'moderate' });
  else if ((input.systolicBp ?? 0) >= 130) w.push({ label: `Stage 1 hypertension (SBP ${input.systolicBp})`, points: 7, severity: 'moderate' });
  if (input.hellpHx) w.push({ label: 'History of HELLP syndrome', points: 10, severity: 'critical' });

  // ---- Moderate contributors ----
  if (input.hypertension) w.push({ label: 'Chronic hypertension', points: 8, severity: 'moderate' });
  if (input.diabetes) w.push({ label: 'Diabetes mellitus', points: 9, severity: 'moderate' });
  if (input.gestationalDmHx) w.push({ label: 'History of gestational diabetes', points: 6, severity: 'moderate' });
  if (input.gestationalHtnHx) w.push({ label: 'History of gestational hypertension', points: 5, severity: 'moderate' });
  if (input.hyperlipidemia) w.push({ label: 'Hyperlipidemia', points: 7, severity: 'moderate' });
  if (input.smoker) w.push({ label: 'Current smoker', points: 9, severity: 'moderate' });
  if (input.familyHxEarlyCvd) w.push({ label: 'Family history of early CVD', points: 5, severity: 'moderate' });
  if ((input.bmi ?? 0) >= 30) w.push({ label: `Obesity (BMI ${input.bmi})`, points: 6, severity: 'moderate' });
  else if ((input.bmi ?? 0) >= 25) w.push({ label: `Overweight (BMI ${input.bmi})`, points: 3, severity: 'moderate' });
  if (input.pcos) w.push({ label: 'PCOS', points: 4, severity: 'moderate' });
  if (input.autoimmune) w.push({ label: 'Autoimmune disease (SLE/RA)', points: 5, severity: 'moderate' });
  if (input.migraineWithAura) w.push({ label: 'Migraine with aura', points: 3, severity: 'moderate' });
  if (input.fmd) w.push({ label: 'Fibromuscular dysplasia (SCAD association)', points: 6, severity: 'moderate' });
  if ((input.age ?? 0) >= 35 && input.pregnant) w.push({ label: 'Advanced maternal age during pregnancy', points: 4, severity: 'moderate' });

  // ---- Protective ----
  if (!input.smoker) w.push({ label: 'Non-smoker', points: -3, severity: 'protective' });
  if (!input.diabetes) w.push({ label: 'No diabetes mellitus', points: -2, severity: 'protective' });

  const raw = w.reduce((s, x) => s + x.points, 0);
  const riskScore = Math.max(0, Math.min(100, Math.round(raw)));
  const riskLevel: RiskLevel =
    riskScore >= 61 ? 'CRITICAL' : riskScore >= 41 ? 'HIGH' : riskScore >= 21 ? 'MODERATE' : 'LOW';

  // Primary working impression (advisory)
  let primaryDx = 'Elevated cardiovascular risk — general';
  if (input.pregnant && (input.preeclampsiaHx || (input.systolicBp ?? 0) >= 140)) primaryDx = 'Preeclampsia-related cardiovascular disease';
  else if (input.postpartumMonths != null) primaryDx = 'Peripartum cardiovascular monitoring';
  else if (input.fmd) primaryDx = 'Fibromuscular dysplasia / SCAD risk';
  else if (!input.hypertension && !input.diabetes && (input.age ?? 0) >= 45) primaryDx = 'Coronary microvascular disease (consider)';

  // Explainability-based confidence: more corroborating factors → higher confidence.
  const contributing = w.filter((x) => x.severity !== 'protective').length;
  const confidence = Math.min(96, 62 + contributing * 4);

  // Automation tier — how much autonomy the tool is permitted (human-in-the-loop by default).
  const automationTier: RiskResult['automationTier'] =
    riskLevel === 'CRITICAL' || riskLevel === 'HIGH' ? 'TIER_2' : 'TIER_1';

  const differentials =
    input.pregnant
      ? [
          { label: 'Gestational hypertension', probability: 42.1 },
          { label: 'Chronic hypertension', probability: 31.5 },
          { label: 'Coronary microvascular disease', probability: 18.7 },
          { label: 'SCAD (spontaneous coronary artery dissection)', probability: 4.2 },
        ]
      : [
          { label: 'Coronary microvascular disease', probability: 34.0 },
          { label: 'Obstructive coronary artery disease', probability: 28.0 },
          { label: 'Takotsubo (stress) cardiomyopathy', probability: 12.0 },
        ];

  // Simple monotonic prognosis derived from the score (illustrative).
  const s = riskScore / 100;
  const prognosis = {
    oneYear: +(s * 12).toFixed(1),
    fiveYear: +(s * 23).toFixed(1),
    tenYear: +(s * 36).toFixed(1),
    lifetime: +(s * 57).toFixed(1),
  };

  const recommendations: string[] = [];
  if (input.pregnant && (input.systolicBp ?? 0) >= 160)
    recommendations.push('Urgent Maternal-Fetal Medicine consultation; assess for HELLP; consider MgSO4 seizure prophylaxis.');
  if ((input.systolicBp ?? 0) >= 140)
    recommendations.push('Initiate/adjust antihypertensive therapy (labetalol or nifedipine if pregnant; avoid ACE-I/ARB in pregnancy).');
  if (input.preeclampsiaHx)
    recommendations.push('Arrange cardiology follow-up within 6–12 weeks postpartum; annual CVD risk assessment.');
  if (input.fmd) recommendations.push('Head-to-pelvis imaging to screen for systemic FMD; antiplatelet therapy.');
  if (!recommendations.length) recommendations.push('Lifestyle optimization and guideline-directed risk-factor management; routine follow-up.');

  return {
    riskScore, riskLevel, primaryDx, confidence, automationTier,
    differentials,
    factors: {
      critical: w.filter((x) => x.severity === 'critical').map((x) => x.label),
      moderate: w.filter((x) => x.severity === 'moderate').map((x) => x.label),
      protective: w.filter((x) => x.severity === 'protective').map((x) => x.label),
    },
    prognosis,
    recommendations,
  };
}
