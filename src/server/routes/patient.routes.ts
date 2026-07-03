import { Router } from 'express';
import { z } from 'zod';
import { prisma } from '../lib/prisma.js';
import { encryptPHI, decryptPHI } from '../lib/security.js';
import { requireAuth, requireRole, type AuthedRequest } from '../middleware/auth.js';
import { audit } from '../middleware/audit.js';

export const patientRouter = Router();
patientRouter.use(requireAuth);

// Decrypt a patient record into a safe DTO for the UI.
function toDto(p: any) {
  return {
    id: p.id, mrn: p.mrn, sex: p.sex, riskLevel: p.riskLevel, active: p.active,
    firstName: decryptPHI(p.firstNameEnc), lastName: decryptPHI(p.lastNameEnc),
    birthDate: decryptPHI(p.birthDateEnc), phone: decryptPHI(p.phoneEnc), email: decryptPHI(p.emailEnc),
    pregnant: p.pregnant, gestationalWeeks: p.gestationalWeeks, postpartumMonths: p.postpartumMonths,
    preeclampsiaHx: p.preeclampsiaHx, gestationalHtnHx: p.gestationalHtnHx, gestationalDmHx: p.gestationalDmHx,
    hellpHx: p.hellpHx, hypertension: p.hypertension, diabetes: p.diabetes, hyperlipidemia: p.hyperlipidemia,
    smoker: p.smoker, familyHxEarlyCvd: p.familyHxEarlyCvd, pcos: p.pcos, autoimmune: p.autoimmune,
    migraineWithAura: p.migraineWithAura, fmd: p.fmd,
    createdAt: p.createdAt,
  };
}

patientRouter.get('/', async (req: AuthedRequest, res, next) => {
  try {
    const patients = await prisma.patient.findMany({ orderBy: { createdAt: 'desc' } });
    await audit(req, 'PATIENT_LIST', 'Patient', 'SUCCESS', { count: patients.length });
    res.json({ patients: patients.map(toDto) });
  } catch (err) { next(err); }
});

patientRouter.get('/:id', async (req: AuthedRequest, res, next) => {
  try {
    const p = await prisma.patient.findUnique({
      where: { id: req.params.id },
      include: { observations: { orderBy: { effectiveAt: 'desc' }, take: 20 }, assessments: { orderBy: { createdAt: 'desc' }, take: 10 } },
    });
    if (!p) return res.status(404).json({ error: 'Patient not found' });
    await audit(req, 'PATIENT_READ', `Patient/${p.id}`, 'SUCCESS');
    res.json({ patient: { ...toDto(p), observations: p.observations, assessments: p.assessments } });
  } catch (err) { next(err); }
});

const createSchema = z.object({
  mrn: z.string().min(1),
  firstName: z.string().min(1),
  lastName: z.string().min(1),
  birthDate: z.string().optional(),
  phone: z.string().optional(),
  email: z.string().email().optional(),
  sex: z.enum(['FEMALE', 'MALE', 'OTHER', 'UNKNOWN']).default('FEMALE'),
  pregnant: z.boolean().optional(),
  gestationalWeeks: z.number().int().optional(),
  preeclampsiaHx: z.boolean().optional(),
  hypertension: z.boolean().optional(),
  diabetes: z.boolean().optional(),
  pcos: z.boolean().optional(),
});

patientRouter.post('/', requireRole('ADMIN', 'PHYSICIAN', 'NURSE'), async (req: AuthedRequest, res, next) => {
  try {
    const d = createSchema.parse(req.body);
    const patient = await prisma.patient.create({
      data: {
        mrn: d.mrn, sex: d.sex,
        firstNameEnc: encryptPHI(d.firstName), lastNameEnc: encryptPHI(d.lastName),
        birthDateEnc: encryptPHI(d.birthDate ?? ''), phoneEnc: encryptPHI(d.phone ?? ''), emailEnc: encryptPHI(d.email ?? ''),
        pregnant: d.pregnant ?? false, gestationalWeeks: d.gestationalWeeks,
        preeclampsiaHx: d.preeclampsiaHx ?? false, hypertension: d.hypertension ?? false,
        diabetes: d.diabetes ?? false, pcos: d.pcos ?? false,
      },
    });
    await audit(req, 'PATIENT_CREATE', `Patient/${patient.id}`, 'SUCCESS');
    res.status(201).json({ patient: toDto(patient) });
  } catch (err) { next(err); }
});
