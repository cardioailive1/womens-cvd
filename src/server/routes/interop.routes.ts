import { Router } from 'express';
import { prisma } from '../lib/prisma.js';
import { encryptPHI } from '../lib/security.js';
import { requireAuth, requireRole, type AuthedRequest } from '../middleware/auth.js';
import { audit } from '../middleware/audit.js';
import { buildFhirPatient, buildFhirObservation, buildFhirRiskAssessment, bundle } from '../fhir/index.js';
import { parseHl7, buildAck } from '../hl7/parser.js';

export const interopRouter = Router();

/* ------------------------- FHIR R4 ------------------------- */
const fhir = Router();
fhir.use(requireAuth);

// GET /api/fhir/Patient/:id  → FHIR R4 Patient
fhir.get('/Patient/:id', async (req: AuthedRequest, res, next) => {
  try {
    const p = await prisma.patient.findUnique({ where: { id: req.params.id } });
    if (!p) return res.status(404).json({ resourceType: 'OperationOutcome', issue: [{ severity: 'error', code: 'not-found' }] });
    await audit(req, 'FHIR_PATIENT_READ', `Patient/${p.id}`, 'SUCCESS');
    res.type('application/fhir+json').json(buildFhirPatient(p));
  } catch (err) { next(err); }
});

// GET /api/fhir/Patient/:id/$everything → Bundle of Patient + Observations + RiskAssessments
fhir.get('/Patient/:id/$everything', async (req: AuthedRequest, res, next) => {
  try {
    const p = await prisma.patient.findUnique({
      where: { id: req.params.id },
      include: { observations: true, assessments: true },
    });
    if (!p) return res.status(404).json({ resourceType: 'OperationOutcome', issue: [{ severity: 'error', code: 'not-found' }] });
    const resources: unknown[] = [buildFhirPatient(p)];
    for (const o of p.observations) resources.push(buildFhirObservation(o));
    for (const a of p.assessments) {
      resources.push(buildFhirRiskAssessment({
        id: a.id, patientId: p.id, createdAt: a.createdAt, primaryDx: a.primaryDx, confidence: a.confidence,
        prognosis: a.prognosis as any, differentials: a.differentials as any, physicianNote: a.physicianNote,
      }));
    }
    await audit(req, 'FHIR_EVERYTHING', `Patient/${p.id}`, 'SUCCESS');
    res.type('application/fhir+json').json(bundle(resources, 'searchset'));
  } catch (err) { next(err); }
});

// GET /api/fhir/Observation?patient=:id
fhir.get('/Observation', async (req: AuthedRequest, res, next) => {
  try {
    const patientId = String(req.query.patient ?? '');
    const obs = await prisma.observation.findMany({ where: { patientId }, orderBy: { effectiveAt: 'desc' } });
    await audit(req, 'FHIR_OBSERVATION_SEARCH', `Patient/${patientId}`, 'SUCCESS');
    res.type('application/fhir+json').json(bundle(obs.map(buildFhirObservation), 'searchset'));
  } catch (err) { next(err); }
});

/* ------------------------- HL7 v2 ------------------------- */
const hl7 = Router();
hl7.use(requireAuth);

// POST /api/hl7/ingest  (Content-Type: text/plain)  → parses ADT/ORU, upserts patient + observations, returns ACK
hl7.post('/ingest', requireRole('ADMIN', 'PHYSICIAN', 'NURSE'), async (req: AuthedRequest, res, next) => {
  try {
    const raw = typeof req.body === 'string' ? req.body : (req.body?.message ?? '');
    const parsed = parseHl7(raw);

    let patientId: string | undefined;
    if (parsed.patient?.mrn) {
      const existing = await prisma.patient.findUnique({ where: { mrn: parsed.patient.mrn } });
      const data = {
        firstNameEnc: encryptPHI(parsed.patient.firstName ?? ''),
        lastNameEnc: encryptPHI(parsed.patient.lastName ?? ''),
        birthDateEnc: encryptPHI(parsed.patient.birthDate ?? ''),
        sex: (parsed.patient.sex as any) ?? 'UNKNOWN',
      };
      const patient = existing
        ? await prisma.patient.update({ where: { id: existing.id }, data })
        : await prisma.patient.create({ data: { mrn: parsed.patient.mrn, ...data } });
      patientId = patient.id;

      // Map OBX results → Observations (LOINC where provided).
      for (const obx of parsed.observations) {
        const value = Number(obx.value);
        if (!obx.code || Number.isNaN(value)) continue;
        await prisma.observation.create({
          data: { patientId: patient.id, loincCode: obx.code, display: obx.display ?? obx.code, value, unit: obx.unit ?? '' },
        });
      }
    }

    await audit(req, 'HL7_INGEST', patientId ? `Patient/${patientId}` : 'HL7', 'SUCCESS', {
      messageType: parsed.messageType, observations: parsed.observations.length,
    });

    res.type('text/plain').send(buildAck(parsed.controlId, 'AA', 'Message processed'));
  } catch (err) {
    // Return an HL7 application-error ACK where possible.
    const controlId = (err as any)?.controlId ?? '';
    res.status((err as any)?.status ?? 400).type('text/plain').send(buildAck(controlId, 'AE', (err as Error).message));
  }
});

// POST /api/hl7/parse → returns the structured parse (no persistence) for testing/preview
hl7.post('/parse', async (req: AuthedRequest, res, next) => {
  try {
    const raw = typeof req.body === 'string' ? req.body : (req.body?.message ?? '');
    res.json(parseHl7(raw));
  } catch (err) { next(err); }
});

interopRouter.use('/fhir', fhir);
interopRouter.use('/hl7', hl7);
