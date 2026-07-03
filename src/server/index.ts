import express from 'express';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import helmet from 'helmet';
import cors from 'cors';
import rateLimit from 'express-rate-limit';
import pinoHttp from 'pino-http';
import { env } from './config/env.js';
import { logger } from './lib/logger.js';
import { errorHandler } from './middleware/audit.js';
import { authRouter } from './routes/auth.routes.js';
import { userRouter } from './routes/user.routes.js';
import { patientRouter } from './routes/patient.routes.js';
import { assessmentRouter } from './routes/assessment.routes.js';
import { interopRouter } from './routes/interop.routes.js';
import { healthRouter, auditRouter } from './routes/health.routes.js';
import { runSeed } from './seed.js';

const app = express();

// --- Security hardening ---
app.set('trust proxy', 1); // Render sits behind a proxy; needed for correct client IP + rate limiting.
app.use(
  helmet({
    contentSecurityPolicy: { directives: { defaultSrc: ["'self'"], objectSrc: ["'none'"], frameAncestors: ["'none'"] } },
    hsts: { maxAge: 63072000, includeSubDomains: true, preload: true },
  }),
);
app.use(
  cors({
    origin: env.CORS_ORIGIN.startsWith('http') ? env.CORS_ORIGIN : `https://${env.CORS_ORIGIN}`,
    credentials: true,
  }),
);
app.use(pinoHttp({ logger }));

// HL7 endpoints accept raw text; everything else JSON.
app.use('/api/hl7', express.text({ type: ['text/plain', 'application/hl7-v2', 'text/*'], limit: '256kb' }));
app.use(express.json({ limit: '512kb' }));

// Global rate limit (per-route limits layered on top).
app.use('/api', rateLimit({ windowMs: 60 * 1000, limit: 300, standardHeaders: true, legacyHeaders: false }));

// --- Routes ---
app.use('/api/health', healthRouter);
app.use('/api/auth', authRouter);
app.use('/api/users', userRouter);
app.use('/api/patients', patientRouter);
app.use('/api/assessments', assessmentRouter);
app.use('/api', interopRouter);          // exposes /api/fhir/* and /api/hl7/*
app.use('/api/audit', auditRouter);

// --- Serve the built React client (single-service deploy) ---
// In production the compiled server lives at dist/server/, and Vite emits the
// client to dist/public/. Serve those static assets and fall back to index.html
// for client-side routes, while leaving /api/* to return JSON 404s.
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const clientDir = path.resolve(__dirname, '../public');
if (env.NODE_ENV === 'production') {
  app.use(express.static(clientDir));
  app.get('*', (req, res, next) => {
    if (req.path.startsWith('/api')) return next();
    res.sendFile(path.join(clientDir, 'index.html'));
  });
}

app.use((_req, res) => res.status(404).json({ error: 'Not found' }));
app.use(errorHandler);

async function start() {
  if (env.SEED_ON_BOOT) {
    try { await runSeed(); } catch (err) { logger.error({ err }, 'seed failed (continuing)'); }
  }
  app.listen(env.PORT, () => logger.info(`CardioAI listening on :${env.PORT} (${env.NODE_ENV})${env.NODE_ENV === 'production' ? ' — serving API + client' : ' — API only (Vite serves client in dev)'}`));
}
start();
