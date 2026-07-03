import type { Response, NextFunction } from 'express';
import type { AuthedRequest } from './auth.js';
import { prisma } from '../lib/prisma.js';
import { logger } from '../lib/logger.js';

type Outcome = 'SUCCESS' | 'DENIED' | 'ERROR';

// Write an append-only audit record. Every PHI access / mutation should call this.
export async function audit(
  req: AuthedRequest,
  action: string,
  resource: string,
  outcome: Outcome = 'SUCCESS',
  meta?: Record<string, unknown>,
) {
  try {
    await prisma.auditLog.create({
      data: {
        actorId: req.user?.id ?? null,
        actorEmail: req.user?.email ?? null,
        action,
        resource,
        outcome,
        ip: (req.headers['x-forwarded-for'] as string)?.split(',')[0]?.trim() || req.ip || null,
        userAgent: req.headers['user-agent'] || null,
        meta: meta ?? undefined,
      },
    });
  } catch (err) {
    // Auditing must never crash the request path, but failures are noteworthy.
    logger.error({ err }, 'audit write failed');
  }
}

// Central error handler — never leak stack traces or PHI to clients.
export function errorHandler(err: unknown, req: AuthedRequest, res: Response, _next: NextFunction) {
  logger.error({ err, path: req.path }, 'unhandled error');
  const status = (err as { status?: number })?.status ?? 500;
  res.status(status).json({ error: status === 500 ? 'Internal server error' : (err as Error).message });
}
