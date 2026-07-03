import { Router } from 'express';
import { prisma } from '../lib/prisma.js';
import { decryptPHI } from '../lib/security.js';
import { relativeTime } from '../lib/time.js';
import { requireAuth, type AuthedRequest } from '../middleware/auth.js';
import { audit } from '../middleware/audit.js';

export const alertsRouter = Router();
alertsRouter.use(requireAuth);

type Severity = 'CRITICAL' | 'HIGH' | 'INFO';
interface Alert {
  id: string; severity: Severity; patient: string; mrn: string;
  message: string; when: string; action: 'Acknowledge' | 'Review' | 'Schedule' | 'View';
}

// GET /api/alerts — derives clinical alerts from current patient risk levels,
// women-specific flags, postpartum status, and assessment recency.
alertsRouter.get('/', async (req: AuthedRequest, res, next) => {
  try {
    const patients = await prisma.patient.findMany({
      where: { active: true },
      include: { assessments: { orderBy: { createdAt: 'desc' }, take: 1 } },
    });

    const alerts: Alert[] = [];
    for (const p of patients) {
      const name = `${decryptPHI(p.firstNameEnc)} ${decryptPHI(p.lastNameEnc)}`.trim() || p.mrn;
      const latest = p.assessments[0];
      const when = relativeTime(latest?.createdAt ?? p.updatedAt);

      if (p.riskLevel === 'CRITICAL') {
        const msg = p.pregnant && p.preeclampsiaHx
          ? 'Preeclampsia risk with severe features in current pregnancy — urgent MFM review.'
          : p.fmd
            ? 'FMD flagged — screen for SCAD; antiplatelet therapy per protocol.'
            : latest
              ? `Critical cardiovascular risk on latest assessment (score ${latest.riskScore}/100).`
              : 'Critical cardiovascular risk on record.';
        alerts.push({ id: `crit-${p.id}`, severity: 'CRITICAL', patient: name, mrn: p.mrn, message: msg, when, action: 'Acknowledge' });
      } else if (p.riskLevel === 'HIGH') {
        const msg = p.hypertension
          ? 'Blood pressure risk elevated — consider medication review.'
          : p.fmd
            ? 'Fibromuscular dysplasia follow-up imaging recommended.'
            : 'Elevated cardiovascular risk — clinician review recommended.';
        alerts.push({ id: `high-${p.id}`, severity: 'HIGH', patient: name, mrn: p.mrn, message: msg, when, action: 'Review' });
      }

      // Informational: postpartum follow-up, or no assessment on file.
      if (p.postpartumMonths != null) {
        alerts.push({ id: `info-pp-${p.id}`, severity: 'INFO', patient: name, mrn: p.mrn, message: `Postpartum follow-up (${p.postpartumMonths} mo) — schedule cardiovascular review.`, when, action: 'Schedule' });
      } else if (!latest) {
        alerts.push({ id: `info-noassess-${p.id}`, severity: 'INFO', patient: name, mrn: p.mrn, message: 'No risk assessment on file — consider running one.', when, action: 'View' });
      }
    }

    const order: Record<Severity, number> = { CRITICAL: 0, HIGH: 1, INFO: 2 };
    alerts.sort((a, b) => order[a.severity] - order[b.severity]);

    await audit(req, 'ALERTS_LIST', 'Alert', 'SUCCESS', { count: alerts.length });
    res.json({
      alerts,
      counts: {
        CRITICAL: alerts.filter((a) => a.severity === 'CRITICAL').length,
        HIGH: alerts.filter((a) => a.severity === 'HIGH').length,
        INFO: alerts.filter((a) => a.severity === 'INFO').length,
      },
    });
  } catch (err) { next(err); }
});
