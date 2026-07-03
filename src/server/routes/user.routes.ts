import { Router } from 'express';
import { z } from 'zod';
import { prisma } from '../lib/prisma.js';
import { hashPassword } from '../lib/security.js';
import { requireAuth, requireRole, type AuthedRequest } from '../middleware/auth.js';
import { audit } from '../middleware/audit.js';

// All routes here are ADMIN-only. Clinical systems provision accounts via an
// administrator (or an upstream IdP), not open self-registration.
export const userRouter = Router();
userRouter.use(requireAuth, requireRole('ADMIN'));

const SAFE_SELECT = {
  id: true, email: true, fullName: true, role: true, specialty: true,
  active: true, mustChangePassword: true, lastLoginAt: true, createdAt: true,
} as const;

const ROLES = ['ADMIN', 'PHYSICIAN', 'NURSE', 'READONLY'] as const;

// GET /api/users — list all users (never returns password hashes)
userRouter.get('/', async (req: AuthedRequest, res, next) => {
  try {
    const users = await prisma.user.findMany({ select: SAFE_SELECT, orderBy: { createdAt: 'desc' } });
    await audit(req, 'USER_LIST', 'User', 'SUCCESS', { count: users.length });
    res.json({ users });
  } catch (err) { next(err); }
});

const createSchema = z.object({
  email: z.string().email(),
  fullName: z.string().min(1),
  role: z.enum(ROLES),
  specialty: z.string().optional(),
  // Temporary password set by the admin; user is forced to change it on first login.
  tempPassword: z.string().min(10, 'Temp password must be at least 10 characters'),
});

// POST /api/users — provision a new account
userRouter.post('/', async (req: AuthedRequest, res, next) => {
  try {
    const d = createSchema.parse(req.body);
    const exists = await prisma.user.findUnique({ where: { email: d.email } });
    if (exists) return res.status(409).json({ error: 'A user with that email already exists' });

    const user = await prisma.user.create({
      data: {
        email: d.email, fullName: d.fullName, role: d.role, specialty: d.specialty,
        passwordHash: await hashPassword(d.tempPassword),
        mustChangePassword: true,
      },
      select: SAFE_SELECT,
    });
    await audit(req, 'USER_CREATE', `User/${user.id}`, 'SUCCESS', { role: user.role });
    res.status(201).json({ user });
  } catch (err) { next(err); }
});

const updateSchema = z.object({
  fullName: z.string().min(1).optional(),
  role: z.enum(ROLES).optional(),
  specialty: z.string().optional(),
  active: z.boolean().optional(),
});

// PATCH /api/users/:id — update role / status / profile (deactivate by setting active:false)
userRouter.patch('/:id', async (req: AuthedRequest, res, next) => {
  try {
    const d = updateSchema.parse(req.body);
    // Guard: an admin cannot lock themselves out by self-demoting/deactivating the last admin.
    if (req.params.id === req.user!.id && (d.role && d.role !== 'ADMIN' || d.active === false)) {
      const admins = await prisma.user.count({ where: { role: 'ADMIN', active: true } });
      if (admins <= 1) return res.status(400).json({ error: 'Cannot demote or deactivate the last active admin' });
    }
    const user = await prisma.user.update({ where: { id: req.params.id }, data: d, select: SAFE_SELECT });
    await audit(req, 'USER_UPDATE', `User/${user.id}`, 'SUCCESS', d);
    res.json({ user });
  } catch (err) { next(err); }
});

const resetSchema = z.object({ tempPassword: z.string().min(10) });

// POST /api/users/:id/reset-password — set a new temp password; forces change on next login
userRouter.post('/:id/reset-password', async (req: AuthedRequest, res, next) => {
  try {
    const { tempPassword } = resetSchema.parse(req.body);
    const user = await prisma.user.update({
      where: { id: req.params.id },
      data: { passwordHash: await hashPassword(tempPassword), mustChangePassword: true },
      select: SAFE_SELECT,
    });
    await audit(req, 'USER_RESET_PASSWORD', `User/${user.id}`, 'SUCCESS');
    res.json({ user });
  } catch (err) { next(err); }
});
