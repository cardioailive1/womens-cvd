/**
 * Seeds SYNTHETIC (non-real) demo data. Safe to run repeatedly (idempotent by email/MRN).
 * Default admin: admin@cardioai.demo / ChangeMe!2026  — CHANGE IMMEDIATELY in production.
 */
import { prisma } from './lib/prisma.js';
import { hashPassword, encryptPHI } from './lib/security.js';
import { logger } from './lib/logger.js';

export async function runSeed() {
  const existing = await prisma.user.findUnique({ where: { email: 'admin@cardioai.demo' } });
  if (existing) {
    logger.info('seed: admin already present, skipping');
    return;
  }

  await prisma.user.create({
    data: {
      email: 'admin@cardioai.demo',
      passwordHash: await hashPassword('ChangeMe!2026'),
      fullName: 'Dr. Demo Admin',
      role: 'ADMIN',
      specialty: 'Cardiology',
    },
  });
  await prisma.user.create({
    data: {
      email: 'physician@cardioai.demo',
      passwordHash: await hashPassword('ChangeMe!2026'),
      fullName: 'Dr. Demo Physician',
      role: 'PHYSICIAN',
      specialty: 'Maternal-Fetal Medicine',
    },
  });

  const patients = [
    { mrn: 'WC001', first: 'Sarah', last: 'Johnson', sex: 'FEMALE', pregnant: true, gestationalWeeks: 28, preeclampsiaHx: true, hypertension: true, riskLevel: 'CRITICAL' as const },
    { mrn: 'WC002', first: 'Maria', last: 'Garcia', sex: 'FEMALE', hypertension: true, diabetes: false, riskLevel: 'HIGH' as const },
    { mrn: 'WC003', first: 'Jennifer', last: 'Lee', sex: 'FEMALE', postpartumMonths: 3, riskLevel: 'MODERATE' as const },
    { mrn: 'WC004', first: 'Emily', last: 'Chen', sex: 'FEMALE', fmd: true, hypertension: true, riskLevel: 'HIGH' as const },
    { mrn: 'WC005', first: 'Jessica', last: 'Brown', sex: 'FEMALE', pcos: true, riskLevel: 'LOW' as const },
  ];

  for (const p of patients) {
    const patient = await prisma.patient.create({
      data: {
        mrn: p.mrn, sex: p.sex as any, riskLevel: p.riskLevel,
        firstNameEnc: encryptPHI(p.first), lastNameEnc: encryptPHI(p.last),
        birthDateEnc: encryptPHI('1988-05-15'),
        pregnant: p.pregnant ?? false, gestationalWeeks: p.gestationalWeeks,
        postpartumMonths: (p as any).postpartumMonths,
        preeclampsiaHx: p.preeclampsiaHx ?? false, hypertension: p.hypertension ?? false,
        diabetes: (p as any).diabetes ?? false, pcos: p.pcos ?? false, fmd: p.fmd ?? false,
      },
    });
    await prisma.observation.createMany({
      data: [
        { patientId: patient.id, loincCode: '8480-6', display: 'Systolic blood pressure', value: p.riskLevel === 'CRITICAL' ? 165 : 132, unit: 'mm[Hg]' },
        { patientId: patient.id, loincCode: '8462-4', display: 'Diastolic blood pressure', value: p.riskLevel === 'CRITICAL' ? 105 : 84, unit: 'mm[Hg]' },
        { patientId: patient.id, loincCode: '39156-5', display: 'Body mass index', value: 29.5, unit: 'kg/m2' },
      ],
    });
  }

  logger.info('seed: created demo users and synthetic patients');
}

// Allow running directly: `npm run seed`
if (import.meta.url === `file://${process.argv[1]}`) {
  runSeed().then(() => process.exit(0)).catch((e) => { console.error(e); process.exit(1); });
}
