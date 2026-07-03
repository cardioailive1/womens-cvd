import { Router } from 'express';
import { z } from 'zod';
import { prisma } from '../lib/prisma.js';
import { requireAuth, requireRole, type AuthedRequest } from '../middleware/auth.js';
import { audit } from '../middleware/audit.js';
import { assessRisk } from '../services/riskEngine.js';

export const assessmentRouter = Router();
assessmentRouter.use(requireAuth);

const runSchema = z.object({
  patientId: z.string(),
  age: z.number().optional(),
  systolicBp: z.number().optional(),
  bmi: z.number().optional(),
});

// Run the (transparent, rules-based) risk assessment for a patient.
assessmentRouter.post('/run', requireRole('ADMIN', 'PHYSICIAN', 'NURSE'), async (req: AuthedRequest, res, next) => {
  try {
    const body = runSchema.parse(req.body);
    const p = await prisma.patient.findUnique({ where: { id: body.patientId } });
    if (!p) return res.status(404).json({ error: 'Patient not found' });

    const result = assessRisk({
      age: body.age, systolicBp: body.systolicBp, bmi: body.bmi,
      pregnant: p.pregnant, gestationalWeeks: p.gestationalWeeks, postpartumMonths: p.postpartumMonths,
      preeclampsiaHx: p.preeclampsiaHx, gestationalHtnHx: p.gestationalHtnHx, gestationalDmHx: p.gestationalDmHx,
      hellpHx: p.hellpHx, hypertension: p.hypertension, diabetes: p.diabetes, hyperlipidemia: p.hyperlipidemia,
      smoker: p.smoker, familyHxEarlyCvd: p.familyHxEarlyCvd, pcos: p.pcos, autoimmune: p.autoimmune,
      migraineWithAura: p.migraineWithAura, fmd: p.fmd,
    });

    const assessment = await prisma.assessment.create({
      data: {
        patientId: p.id, clinicianId: req.user!.id,
        primaryDx: result.primaryDx, confidence: result.confidence,
        riskScore: result.riskScore, riskLevel: result.riskLevel, automationTier: result.automationTier,
        differentials: result.differentials as any, factors: result.factors as any,
        prognosis: result.prognosis as any, recommendations: result.recommendations as any,
      },
    });
    await prisma.patient.update({ where: { id: p.id }, data: { riskLevel: result.riskLevel } });
    await audit(req, 'ASSESSMENT_CREATE', `Assessment/${assessment.id}`, 'SUCCESS', { riskLevel: result.riskLevel });

    res.status(201).json({ assessment: { id: assessment.id, ...result, createdAt: assessment.createdAt } });
  } catch (err) { next(err); }
});

const noteSchema = z.object({ note: z.string().min(1), confirm: z.boolean().optional() });

// Physician review / confirmation (human-in-the-loop).
assessmentRouter.post('/:id/review', requireRole('ADMIN', 'PHYSICIAN'), async (req: AuthedRequest, res, next) => {
  try {
    const { note, confirm } = noteSchema.parse(req.body);
    const updated = await prisma.assessment.update({
      where: { id: req.params.id },
      data: { physicianNote: note, status: confirm ? 'CONFIRMED' : 'DRAFT' },
    });
    await audit(req, 'ASSESSMENT_REVIEW', `Assessment/${updated.id}`, 'SUCCESS', { status: updated.status });
    res.json({ assessment: updated });
  } catch (err) { next(err); }
});
