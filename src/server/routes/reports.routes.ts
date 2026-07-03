import { Router } from 'express';
import { prisma } from '../lib/prisma.js';
import { requireAuth, type AuthedRequest } from '../middleware/auth.js';
import { audit } from '../middleware/audit.js';

export const reportsRouter = Router();
reportsRouter.use(requireAuth);

// Group a primary diagnosis string into a chart bucket.
function bucket(dx: string): string {
  const d = dx.toLowerCase();
  if (d.includes('preeclampsia')) return 'Preeclampsia';
  if (d.includes('peripartum') || d.includes('ppcm')) return 'Peripartum';
  if (d.includes('fibromuscular') || d.includes('scad')) return 'FMD / SCAD';
  if (d.includes('microvascular')) return 'Microvascular';
  return 'General CVD';
}

// GET /api/reports/summary?days=30 — real stats computed from assessment records.
reportsRouter.get('/summary', async (req: AuthedRequest, res, next) => {
  try {
    const days = Math.min(Math.max(Number(req.query.days ?? 30), 1), 3650);
    const since = new Date(Date.now() - days * 24 * 60 * 60 * 1000);

    const assessments = await prisma.assessment.findMany({
      where: { createdAt: { gte: since } },
      select: { riskLevel: true, primaryDx: true, status: true, createdAt: true },
    });

    const riskLevelBreakdown = { LOW: 0, MODERATE: 0, HIGH: 0, CRITICAL: 0 } as Record<string, number>;
    const distMap = new Map<string, number>();
    let confirmed = 0;
    for (const a of assessments) {
      riskLevelBreakdown[a.riskLevel] = (riskLevelBreakdown[a.riskLevel] ?? 0) + 1;
      const b = bucket(a.primaryDx);
      distMap.set(b, (distMap.get(b) ?? 0) + 1);
      if (a.status === 'CONFIRMED') confirmed++;
    }

    const totalPatients = await prisma.patient.count({ where: { active: true } });

    await audit(req, 'REPORT_SUMMARY', 'Report', 'SUCCESS', { days, assessments: assessments.length });
    res.json({
      period: { days, since: since.toISOString() },
      stats: {
        totalAssessments: assessments.length,
        highRiskIdentified: (riskLevelBreakdown.HIGH ?? 0) + (riskLevelBreakdown.CRITICAL ?? 0),
        criticalCount: riskLevelBreakdown.CRITICAL ?? 0,
        confirmedCount: confirmed,
        totalPatients,
      },
      riskLevelBreakdown,
      distribution: [...distMap.entries()].map(([label, count]) => ({ label, count })).sort((a, b) => b.count - a.count),
    });
  } catch (err) { next(err); }
});
