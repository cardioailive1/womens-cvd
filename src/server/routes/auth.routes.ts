import { Router } from 'express';
import { z } from 'zod';
import rateLimit from 'express-rate-limit';
import { env } from '../config/env.js';
import { prisma } from '../lib/prisma.js';
import { verifyPassword, hashPassword, signToken } from '../lib/security.js';
import { audit } from '../middleware/audit.js';
import { requireAuth, type AuthedRequest } from '../middleware/auth.js';

export const authRouter = Router();

// Throttle credential-stuffing / brute force (SOC 2 CC6.1).
const loginLimiter = rateLimit({ windowMs: 15 * 60 * 1000, limit: 10, standardHeaders: true, legacyHeaders: false });
const signupLimiter = rateLimit({ windowMs: 15 * 60 * 1000, limit: 20, standardHeaders: true, legacyHeaders: false });

// Helper: shape a user for the client + mint a session token (auto-login).
function sessionResponse(user: { id: string; email: string; fullName: string; role: string; specialty: string | null; mustChangePassword: boolean }) {
  const token = signToken({ sub: user.id, email: user.email, role: user.role });
  return {
    token,
    user: { id: user.id, email: user.email, fullName: user.fullName, role: user.role, specialty: user.specialty, mustChangePassword: user.mustChangePassword },
  };
}

const loginSchema = z.object({ email: z.string().email(), password: z.string().min(1) });

authRouter.post('/login', loginLimiter, async (req: AuthedRequest, res, next) => {
  try {
    const { email, password } = loginSchema.parse(req.body);
    const user = await prisma.user.findUnique({ where: { email } });
    if (!user || !user.active || !(await verifyPassword(password, user.passwordHash))) {
      await audit(req, 'LOGIN', `User/${email}`, 'DENIED');
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    await prisma.user.update({ where: { id: user.id }, data: { lastLoginAt: new Date() } });
    const token = signToken({ sub: user.id, email: user.email, role: user.role });
    req.user = { id: user.id, email: user.email, role: user.role };
    await audit(req, 'LOGIN', `User/${user.id}`, 'SUCCESS');
    return res.json({
      token,
      user: { id: user.id, email: user.email, fullName: user.fullName, role: user.role, specialty: user.specialty, mustChangePassword: user.mustChangePassword },
    });
  } catch (err) {
    return next(err);
  }
});

// Public config the login screen uses to decide what to show.
authRouter.get('/config', async (_req, res, next) => {
  try {
    const userCount = await prisma.user.count();
    res.json({ needsBootstrap: userCount === 0, openSignup: env.ALLOW_OPEN_SIGNUP });
  } catch (err) {
    next(err);
  }
});

const signupSchema = z.object({
  email: z.string().email(),
  fullName: z.string().min(1),
  password: z.string().min(10, 'Password must be at least 10 characters'),
});

// First-run bootstrap: create the very first ADMIN. Works ONLY while zero users
// exist, so a fresh deployment is never locked out. No auth required by design;
// it self-disables the moment any account exists.
authRouter.post('/bootstrap', signupLimiter, async (req: AuthedRequest, res, next) => {
  try {
    const d = signupSchema.parse(req.body);
    if ((await prisma.user.count()) > 0) {
      await audit(req, 'BOOTSTRAP', 'User', 'DENIED');
      return res.status(409).json({ error: 'Setup already completed. Ask an administrator to provision your account.' });
    }
    const user = await prisma.user.create({
      data: { email: d.email, fullName: d.fullName, role: 'ADMIN', passwordHash: await hashPassword(d.password) },
    });
    req.user = { id: user.id, email: user.email, role: user.role };
    await audit(req, 'BOOTSTRAP', `User/${user.id}`, 'SUCCESS');
    return res.status(201).json(sessionResponse(user));
  } catch (err) {
    return next(err);
  }
});

// Open self-registration (only when ALLOW_OPEN_SIGNUP=true). New users get the
// least-privilege READONLY role; an admin elevates them later from the Users page.
authRouter.post('/register', signupLimiter, async (req: AuthedRequest, res, next) => {
  try {
    if (!env.ALLOW_OPEN_SIGNUP) {
      return res.status(403).json({ error: 'Self-registration is disabled. Contact an administrator.' });
    }
    const d = signupSchema.parse(req.body);
    if (await prisma.user.findUnique({ where: { email: d.email } })) {
      return res.status(409).json({ error: 'An account with that email already exists.' });
    }
    const user = await prisma.user.create({
      data: { email: d.email, fullName: d.fullName, role: 'READONLY', passwordHash: await hashPassword(d.password) },
    });
    req.user = { id: user.id, email: user.email, role: user.role };
    await audit(req, 'REGISTER', `User/${user.id}`, 'SUCCESS');
    return res.status(201).json(sessionResponse(user));
  } catch (err) {
    return next(err);
  }
});

authRouter.get('/me', requireAuth, async (req: AuthedRequest, res, next) => {
  try {
    const user = await prisma.user.findUnique({
      where: { id: req.user!.id },
      select: { id: true, email: true, fullName: true, role: true, specialty: true, mustChangePassword: true },
    });
    return res.json({ user });
  } catch (err) {
    return next(err);
  }
});

const changePwSchema = z.object({
  currentPassword: z.string().min(1),
  newPassword: z.string().min(10, 'New password must be at least 10 characters'),
});

// Self-service password change — used to rotate an admin-issued temp password.
authRouter.post('/change-password', requireAuth, async (req: AuthedRequest, res, next) => {
  try {
    const { currentPassword, newPassword } = changePwSchema.parse(req.body);
    const user = await prisma.user.findUnique({ where: { id: req.user!.id } });
    if (!user || !(await verifyPassword(currentPassword, user.passwordHash))) {
      await audit(req, 'PASSWORD_CHANGE', `User/${req.user!.id}`, 'DENIED');
      return res.status(401).json({ error: 'Current password is incorrect' });
    }
    await prisma.user.update({
      where: { id: user.id },
      data: { passwordHash: await hashPassword(newPassword), mustChangePassword: false },
    });
    await audit(req, 'PASSWORD_CHANGE', `User/${user.id}`, 'SUCCESS');
    return res.json({ ok: true });
  } catch (err) {
    return next(err);
  }
});
