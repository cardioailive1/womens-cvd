# Compliance & Regulatory Notes

This document is deliberately honest about a distinction that marketing language often blurs:

> **Code can implement _technical safeguards_. It cannot, by itself, make an organization "HIPAA compliant,"
> "SOC 2 certified," or a product "FDA approved / cleared."** Those are organizational, legal, and process
> outcomes achieved by people and audits — not by shipping software.

Below is what this codebase actually implements, and what still sits with your organization.

---

## 1. HIPAA Security Rule (45 CFR §164.302–318)

HIPAA compliance is achieved by a **covered entity / business associate** through administrative, physical,
and technical safeguards **plus** signed Business Associate Agreements (BAAs). This repo addresses part of
the *technical* safeguards only.

| HIPAA technical safeguard | Implemented here | Still your responsibility |
|---|---|---|
| Access control §164.312(a)(1) | JWT auth + role-based access control (`middleware/auth.ts`) | Unique-user provisioning, periodic access reviews, emergency-access procedure |
| Encryption at rest §164.312(a)(2)(iv) | AES-256-GCM field-level PHI encryption (`lib/security.ts`) | Disk/DB-level encryption, key management/rotation (KMS/HSM) |
| Encryption in transit §164.312(e) | HSTS + TLS assumed at the platform edge (Render) | Enforce TLS everywhere; certificate management |
| Audit controls §164.312(b) | Append-only `AuditLog` for auth + PHI access/mutations | Log retention (6 yrs), review cadence, tamper-evident storage/SIEM |
| Integrity §164.312(c) | GCM auth tags detect ciphertext tampering | Broader integrity monitoring, backups |
| Person/entity authentication §164.312(d) | Password hashing (bcrypt), short-lived JWTs | MFA, SSO/IdP integration, session policies |
| Automatic logoff §164.312(a)(2)(iii) | Short JWT expiry (`JWT_EXPIRES_IN`) | UI idle-timeout policy |
| **Organizational** | — | **Signed BAA with Render and any subprocessors**, risk analysis, workforce training, breach-notification process, sanction policy |

Minimum-necessary logging is supported by PHI redaction in the logger (`lib/logger.ts`).

---

## 2. SOC 2 (AICPA Trust Services Criteria)

SOC 2 is an **attestation by an independent CPA firm** over a period of time. There is no "SOC 2 code."
This repo provides evidence-generating controls that map to several criteria:

| TSC | Supporting control in code |
|---|---|
| CC6.1 Logical access | RBAC, JWT, bcrypt, login rate limiting |
| CC6.6 Boundary protection | Helmet security headers, CORS allow-list, CSP |
| CC6.7 Data in transit/at rest | TLS/HSTS + AES-256-GCM field encryption |
| CC7.2 Monitoring | Structured logging (pino) + append-only audit trail |
| CC7.3/CC7.4 Incident handling | Audit records with actor/IP/outcome for investigation |
| CC8.1 Change management | IaC (`render.yaml`), typed migrations (Prisma) |

**Your responsibility:** define the system boundary, write policies, run the controls for the audit window
(typically 3–12 months), engage an auditor, and remediate findings.

---

## 3. FDA — Software as a Medical Device (SaMD)

Whether this software is a regulated **medical device** depends on its intended use. A tool that drives
diagnosis or treatment of a specific patient generally is; purely educational reference material generally
isn't. **This platform is shipped as decision-support/education and is _not_ FDA-cleared.**

Design choices that anticipate a SaMD pathway:
- **Transparency / explainability.** The risk engine is rules-based and returns every contributing factor —
  no black box. This supports IEC 62304 traceability and FDA's transparency expectations.
- **Human-in-the-loop.** Assessments default to advisory; physicians confirm (`/assessments/:id/review`),
  and an `automationTier` is attached to each result.
- **Traceability.** Immutable audit logging of who ran/confirmed what, and when.
- **Clear labeling.** In-app disclaimers state the tool is non-diagnostic and uses synthetic data.

**Before clinical use you must:** define intended use & risk category (IMDRF/FDA), replace the illustrative
engine with a **clinically validated, locally calibrated** model, follow IEC 62304 (lifecycle) and ISO 14971
(risk management), complete clinical validation, and obtain the appropriate FDA pathway (e.g. 510(k)/De Novo)
or a documented enforcement-discretion determination.

---

## 4. International quality standards

| Standard | Relevance | Status here |
|---|---|---|
| ISO/IEC 27001 | Information security management system | Technical controls present; ISMS/certification is organizational |
| ISO 13485 | Medical-device quality management system | Not implemented (process/QMS, not code) |
| ISO 14971 | Medical-device risk management | Anticipated by design; formal risk file is yours to produce |
| IEC 62304 | Medical-device software lifecycle | Structure/traceability supports it; full lifecycle docs are yours |
| ISO 27799 | Health-informatics security | Supported by encryption/audit/access controls |
| GDPR (if EU data) | Lawful basis, DPIA, data-subject rights | Encryption/audit help; DPO, DPIA, and processes are organizational |

---

## 5. Honest summary

What you can truthfully say after deploying this repo:

- ✅ "PHI is encrypted at rest with AES-256-GCM and access is role-restricted and fully audited."
- ✅ "The system implements technical safeguards that map to the HIPAA Security Rule and SOC 2 criteria."
- ✅ "The clinical logic is transparent and explainable, with human-in-the-loop confirmation."

What you may **not** truthfully say without the organizational work:

- ❌ "This app is HIPAA compliant." (Requires BAAs, policies, training, risk analysis, ongoing operations.)
- ❌ "This app is SOC 2 certified." (Requires an independent auditor's report over an observation period.)
- ❌ "This app is FDA approved/cleared." (Requires validation and an FDA regulatory pathway.)

Treat this repository as a strong, honest *starting point* for the technical portion of those journeys.
