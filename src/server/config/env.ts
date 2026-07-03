import 'dotenv/config';
import { z } from 'zod';

const schema = z.object({
  NODE_ENV: z.enum(['development', 'production', 'test']).default('development'),
  PORT: z.coerce.number().default(4000),
  DATABASE_URL: z.string().url(),
  JWT_SECRET: z.string().min(24, 'JWT_SECRET must be at least 24 chars'),
  JWT_EXPIRES_IN: z.string().default('30m'),
  PHI_ENCRYPTION_KEY: z
    .string()
    .refine((v) => Buffer.from(v, 'base64').length === 32, 'PHI_ENCRYPTION_KEY must decode to 32 bytes (base64)'),
  CORS_ORIGIN: z.string().default('http://localhost:5173'),
  SEED_ON_BOOT: z
    .string()
    .optional()
    .transform((v) => v === 'true'),
  // When true, anyone may self-register (as READONLY). First-run admin bootstrap
  // is always available while zero users exist, regardless of this flag.
  ALLOW_OPEN_SIGNUP: z
    .string()
    .optional()
    .transform((v) => v === 'true'),
});

const parsed = schema.safeParse(process.env);
if (!parsed.success) {
  // Fail fast — never boot a PHI service with a bad config.
  console.error('❌ Invalid environment configuration:');
  console.error(parsed.error.flatten().fieldErrors);
  process.exit(1);
}

export const env = parsed.data;
