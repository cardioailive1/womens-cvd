import crypto from 'node:crypto';
import bcrypt from 'bcryptjs';
import jwt from 'jsonwebtoken';
import { env } from '../config/env.js';

/* ------------------------------------------------------------------ *
 * Field-level PHI encryption — AES-256-GCM (authenticated encryption)
 * Format stored in DB:  iv:authTag:ciphertext  (all base64)
 * Supports HIPAA §164.312(a)(2)(iv) encryption of ePHI at rest.
 * ------------------------------------------------------------------ */
const KEY = Buffer.from(env.PHI_ENCRYPTION_KEY, 'base64'); // 32 bytes
const ALGO = 'aes-256-gcm';

export function encryptPHI(plaintext: string | null | undefined): string {
  if (plaintext == null) return '';
  const iv = crypto.randomBytes(12);
  const cipher = crypto.createCipheriv(ALGO, KEY, iv);
  const enc = Buffer.concat([cipher.update(String(plaintext), 'utf8'), cipher.final()]);
  const tag = cipher.getAuthTag();
  return [iv.toString('base64'), tag.toString('base64'), enc.toString('base64')].join(':');
}

export function decryptPHI(payload: string | null | undefined): string {
  if (!payload) return '';
  const [ivB64, tagB64, dataB64] = payload.split(':');
  if (!ivB64 || !tagB64 || !dataB64) return '';
  try {
    const decipher = crypto.createDecipheriv(ALGO, KEY, Buffer.from(ivB64, 'base64'));
    decipher.setAuthTag(Buffer.from(tagB64, 'base64'));
    const dec = Buffer.concat([decipher.update(Buffer.from(dataB64, 'base64')), decipher.final()]);
    return dec.toString('utf8');
  } catch {
    return '';
  }
}

/* ------------------------------------------------------------------ *
 * Password hashing
 * ------------------------------------------------------------------ */
export async function hashPassword(pw: string): Promise<string> {
  return bcrypt.hash(pw, 12);
}
export async function verifyPassword(pw: string, hash: string): Promise<boolean> {
  return bcrypt.compare(pw, hash);
}

/* ------------------------------------------------------------------ *
 * JWT
 * ------------------------------------------------------------------ */
export interface TokenPayload {
  sub: string;
  email: string;
  role: string;
}
export function signToken(payload: TokenPayload): string {
  return jwt.sign(payload, env.JWT_SECRET, { expiresIn: env.JWT_EXPIRES_IN as jwt.SignOptions['expiresIn'] });
}
export function verifyToken(token: string): TokenPayload {
  return jwt.verify(token, env.JWT_SECRET) as TokenPayload;
}
