/**
 * FHIR R4 resource builders.
 * Maps internal records to HL7 FHIR R4 (http://hl7.org/fhir/R4/) JSON resources.
 * Codes use LOINC (observations) and SNOMED CT where applicable.
 */
import { decryptPHI } from '../lib/security.js';

export interface FhirPatient {
  resourceType: 'Patient';
  id: string;
  identifier: { system: string; value: string }[];
  active: boolean;
  name: { use: string; family: string; given: string[] }[];
  gender: string;
  birthDate?: string;
  telecom?: { system: string; value: string }[];
}

export interface FhirObservation {
  resourceType: 'Observation';
  id: string;
  status: 'final';
  category: { coding: { system: string; code: string; display: string }[] }[];
  code: { coding: { system: string; code: string; display: string }[] };
  subject: { reference: string };
  effectiveDateTime: string;
  valueQuantity: { value: number; unit: string; system: string; code: string };
}

export interface FhirRiskAssessment {
  resourceType: 'RiskAssessment';
  id: string;
  status: 'final';
  subject: { reference: string };
  occurrenceDateTime: string;
  prediction: {
    outcome: { text: string };
    probabilityDecimal?: number;
    whenRange?: { high: { value: number; unit: string } };
  }[];
  note?: { text: string }[];
}

export interface FhirBundle {
  resourceType: 'Bundle';
  type: 'searchset' | 'collection';
  total?: number;
  entry: { fullUrl?: string; resource: unknown }[];
}

const LOINC = 'http://loinc.org';
const UCUM = 'http://unitsofmeasure.org';

export function toFhirGender(sex: string): string {
  switch (sex) {
    case 'FEMALE': return 'female';
    case 'MALE': return 'male';
    case 'OTHER': return 'other';
    default: return 'unknown';
  }
}

export function buildFhirPatient(p: {
  id: string; mrn: string; sex: string; active: boolean;
  firstNameEnc: string; lastNameEnc: string; birthDateEnc: string;
  phoneEnc?: string | null; emailEnc?: string | null;
}): FhirPatient {
  const telecom: { system: string; value: string }[] = [];
  const phone = decryptPHI(p.phoneEnc);
  const email = decryptPHI(p.emailEnc);
  if (phone) telecom.push({ system: 'phone', value: phone });
  if (email) telecom.push({ system: 'email', value: email });

  return {
    resourceType: 'Patient',
    id: p.id,
    identifier: [{ system: 'urn:cardioai:mrn', value: p.mrn }],
    active: p.active,
    name: [{ use: 'official', family: decryptPHI(p.lastNameEnc), given: [decryptPHI(p.firstNameEnc)] }],
    gender: toFhirGender(p.sex),
    birthDate: decryptPHI(p.birthDateEnc) || undefined,
    telecom: telecom.length ? telecom : undefined,
  };
}

export function buildFhirObservation(o: {
  id: string; patientId: string; loincCode: string; display: string;
  value: number; unit: string; effectiveAt: Date;
}): FhirObservation {
  return {
    resourceType: 'Observation',
    id: o.id,
    status: 'final',
    category: [
      { coding: [{ system: 'http://terminology.hl7.org/CodeSystem/observation-category', code: 'vital-signs', display: 'Vital Signs' }] },
    ],
    code: { coding: [{ system: LOINC, code: o.loincCode, display: o.display }] },
    subject: { reference: `Patient/${o.patientId}` },
    effectiveDateTime: o.effectiveAt.toISOString(),
    valueQuantity: { value: o.value, unit: o.unit, system: UCUM, code: o.unit },
  };
}

export function buildFhirRiskAssessment(a: {
  id: string; patientId: string; createdAt: Date; primaryDx: string; confidence: number;
  prognosis: { oneYear: number; fiveYear: number; tenYear: number; lifetime: number };
  differentials: { label: string; probability: number }[];
  physicianNote?: string | null;
}): FhirRiskAssessment {
  const prediction: FhirRiskAssessment['prediction'] = [
    { outcome: { text: `${a.primaryDx} (1-year MACE)` }, probabilityDecimal: a.prognosis.oneYear / 100, whenRange: { high: { value: 1, unit: 'a' } } },
    { outcome: { text: `${a.primaryDx} (10-year MACE)` }, probabilityDecimal: a.prognosis.tenYear / 100, whenRange: { high: { value: 10, unit: 'a' } } },
    ...a.differentials.map((d) => ({ outcome: { text: d.label }, probabilityDecimal: d.probability / 100 })),
  ];
  return {
    resourceType: 'RiskAssessment',
    id: a.id,
    status: 'final',
    subject: { reference: `Patient/${a.patientId}` },
    occurrenceDateTime: a.createdAt.toISOString(),
    prediction,
    note: a.physicianNote ? [{ text: a.physicianNote }] : undefined,
  };
}

export function bundle(resources: unknown[], type: FhirBundle['type'] = 'collection'): FhirBundle {
  return {
    resourceType: 'Bundle',
    type,
    total: resources.length,
    entry: resources.map((r) => ({ resource: r })),
  };
}
