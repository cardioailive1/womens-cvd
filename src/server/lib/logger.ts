import pino from 'pino';
import { env } from '../config/env.js';

// Redact common PHI/secret paths so they never reach logs (HIPAA minimum-necessary).
export const logger = pino({
  level: env.NODE_ENV === 'production' ? 'info' : 'debug',
  redact: {
    paths: [
      'req.headers.authorization',
      'req.headers.cookie',
      '*.password',
      '*.passwordHash',
      '*.firstName',
      '*.lastName',
      '*.birthDate',
      '*.phone',
      '*.email',
      '*.ssn',
    ],
    censor: '[REDACTED]',
  },
  transport:
    env.NODE_ENV === 'development'
      ? { target: 'pino-pretty', options: { colorize: true, translateTime: 'SYS:HH:MM:ss' } }
      : undefined,
});
