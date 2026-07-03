import type { Request, Response, NextFunction } from 'express';
import { verifyToken } from '../lib/security.js';

export interface AuthedRequest extends Request {
  user?: { id: string; email: string; role: string };
}

export function requireAuth(req: AuthedRequest, res: Response, next: NextFunction) {
  const header = req.headers.authorization;
  if (!header?.startsWith('Bearer ')) {
    return res.status(401).json({ error: 'Missing or malformed Authorization header' });
  }
  try {
    const payload = verifyToken(header.slice(7));
    req.user = { id: payload.sub, email: payload.email, role: payload.role };
    return next();
  } catch {
    return res.status(401).json({ error: 'Invalid or expired token' });
  }
}

// Role-based access control (SOC 2 CC6 / HIPAA §164.312(a)(1) access control).
export function requireRole(...roles: string[]) {
  return (req: AuthedRequest, res: Response, next: NextFunction) => {
    if (!req.user) return res.status(401).json({ error: 'Unauthenticated' });
    if (!roles.includes(req.user.role)) {
      return res.status(403).json({ error: 'Insufficient role', required: roles });
    }
    return next();
  };
}
