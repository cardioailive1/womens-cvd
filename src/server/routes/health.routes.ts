import { Router } from 'express';
import { prisma } from '../lib/prisma.js';
import { requireAuth, requireRole, type AuthedRequest } from '../middleware/auth.js';

export const healthRouter = Router();

// Liveness/readiness probe for Render (checks DB connectivity).
healthRouter.get('/', async (_req, res) => {
  try {
    await prisma.$queryRaw`SELECT 1`;
    res.json({ status: 'ok', db: 'up', time: new Date().toISOString(), version: '1.0.0' });
  } catch {
    res.status(503).json({ status: 'degraded', db: 'down', time: new Date().toISOString() });
  }
});

// Audit trail viewer — admin only (SOC 2 CC7 / HIPAA audit controls).
export const auditRouter = Router();
auditRouter.use(requireAuth, requireRole('ADMIN'));
auditRouter.get('/', async (req: AuthedRequest, res, next) => {
  try {
    const take = Math.min(Number(req.query.limit ?? 100), 500);
    const logs = await prisma.auditLog.findMany({ orderBy: { createdAt: 'desc' }, take });
    res.json({ logs });
  } catch (err) { next(err); }
});
