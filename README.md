# CardioAI — Women's Cardiovascular Disease Platform

A modern, deployable **single-service** full-stack clinical **decision-support** app focused on
women-specific cardiovascular disease (preeclampsia/PPCM, SCAD, Takotsubo, coronary microvascular disease,
FMD). One Express server exposes a **FHIR R4** + **HL7 v2** API *and* serves the compiled **React** client,
with the technical controls that support HIPAA / SOC 2 / FDA-SaMD / ISO programs.

> ⚠️ **Read this first.** This is a demonstration/reference platform. The risk engine is a *transparent,
> rules-based* scaffold — **not** a validated diagnostic model and **not** an FDA-cleared device. All demo
> data is synthetic. Software alone is never "HIPAA compliant," "SOC 2 certified," or "FDA approved" — those
> are organizational programs. See [`COMPLIANCE.md`](./COMPLIANCE.md) for what's implemented in code versus
> what remains your organization's responsibility.

---

## Stack

| Layer | Technology |
|-------|-----------|
| Client | React 18 + TypeScript + Vite + React Router (built to `dist/public`) |
| Server | Node 20 + TypeScript (ESM) + Express (serves API **and** the built client) |
| Database | PostgreSQL + Prisma ORM |
| Auth | JWT (short-lived) + bcrypt + role-based access control; admin-provisioned accounts |
| Security | AES-256-GCM PHI field encryption, Helmet, CORS, rate limiting, append-only audit log |
| Interop | FHIR R4 (Patient / Observation / RiskAssessment / Bundle), HL7 v2 (ADT/ORU parser + ACK) |
| Deploy | Render.com Blueprint (`render.yaml`) — **one** web service + database |

---

## Unified project layout

Backend and frontend now live under a single `src/` and ship as one package.

```
cardioai-cvd/
├── render.yaml            # One-click Render blueprint (single web service + db)
├── package.json           # Unified deps + scripts for both halves
├── index.html             # Vite entry (loads src/client/main.tsx)
├── vite.config.ts         # Client build → dist/public; dev proxy /api → :4000
├── tsconfig.json          # Client TS project (DOM/JSX)
├── tsconfig.server.json   # Server TS project (Node/ESM) → dist/server
├── tsconfig.node.json     # Tooling (vite.config.ts)
├── .env.example
├── prisma/schema.prisma
├── README.md · COMPLIANCE.md
└── src/
    ├── server/            # Express API
    │   ├── index.ts       # bootstrap + security + serves dist/public in prod
    │   ├── auth/sso.ts     # documented SSO/IdP seam (deferred)
    │   ├── config/env.ts
    │   ├── lib/           # prisma, security (crypto/JWT/hash), logger
    │   ├── middleware/    # auth (RBAC), audit, error handler
    │   ├── fhir/ · hl7/   # FHIR R4 builders, HL7 v2 parser
    │   ├── services/riskEngine.ts
    │   ├── routes/        # auth, users, patients, assessments, interop, health
    │   └── seed.ts
    └── client/            # React SPA
        ├── main.tsx · App.tsx · api.ts · auth.tsx · styles.css
        └── pages/         # Login, Dashboard, Patients, Assessment, Diagnostics,
                          #   Alerts, Reports, Interop, Users, Account
```

**How the two halves connect:** in production `npm run build` compiles the server to `dist/server/` and the
client to `dist/public/`; the Express server serves those static assets and falls back to `index.html` for
client-side routes, while `/api/*` stays JSON. In development, Vite serves the client on `:5173` and proxies
`/api` to the Express server on `:4000`, so you still get instant HMR.

---

## Local development

### Prerequisites
- Node.js 20+
- PostgreSQL 16 (or Docker: `docker run -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:16`)

### Setup
```bash
cp .env.example .env          # then edit values

# generate the two required secrets and paste into .env:
node -e "console.log('JWT_SECRET=' + require('crypto').randomBytes(48).toString('base64'))"
node -e "console.log('PHI_ENCRYPTION_KEY=' + require('crypto').randomBytes(32).toString('base64'))"

npm install
npx prisma db push                   # creates tables from the schema
npm run seed                          # optional: synthetic demo data
npm run dev                           # server :4000 + client :5173 together
```

Open http://localhost:5173 and sign in with a demo account below.

### Run the production build locally
```bash
npm run build      # → dist/server + dist/public
npm start          # single server on :4000 serving API + client
```

### Demo accounts (synthetic)
| Email | Role | Password |
|-------|------|----------|
| `admin@cardioai.demo` | ADMIN | `ChangeMe!2026` |
| `physician@cardioai.demo` | PHYSICIAN | `ChangeMe!2026` |

**Change these immediately for any non-throwaway environment.**

---

## Scripts

| Command | Does |
|---------|------|
| `npm run dev` | Runs server (tsx watch) + client (Vite) concurrently |
| `npm run build` | Builds client → `dist/public` and server → `dist/server` |
| `npm start` | Runs the compiled single service (`node dist/server/index.js`) |
| `npm run typecheck` | Type-checks both TS projects |
| `npm run seed` | Loads synthetic demo data |
| `npm run migrate:dev` / `migrate:deploy` | Prisma migrations |

---

## Deploy to Render.com

1. Push this repository to GitHub.
2. Render → **New + → Blueprint** → select the repo. `render.yaml` provisions:
   - `cardioai-db` — managed PostgreSQL 16
   - `cardioai` — one Node web service that builds the client, compiles the server, runs
     `prisma db push`, and starts `node dist/server/index.js`. `JWT_SECRET` and
     `PHI_ENCRYPTION_KEY` are auto-generated and stored encrypted.
3. First boot seeds synthetic data (`SEED_ON_BOOT=true`). **Set it to `false`** afterward and rotate the
   demo passwords.

> For real PHI workloads you must also sign a **BAA with Render**, move the DB to a paid tier with
> backups/PITR, restrict `ipAllowList`, and complete the organizational steps in `COMPLIANCE.md`.

---

## Accounts, roles & auth

JWT-based (bcrypt hashing, short-lived tokens); access is role-gated by `requireRole(...)` middleware.

**Account creation — three paths:**

1. **First-run bootstrap.** On a brand-new deployment with zero accounts, the login screen shows
   "Create the first administrator." `POST /api/auth/bootstrap` works *only* while no users exist, so a
   fresh deploy is never locked out; it self-disables the moment any account is created.
2. **Open self-registration** (when `ALLOW_OPEN_SIGNUP=true`, the default in `render.yaml`). Anyone can
   create an account from the login screen. New users get the least-privilege **READONLY** role; an admin
   elevates them from the Users page. Set `ALLOW_OPEN_SIGNUP=false` to require admin provisioning only.
3. **Admin provisioning.** An `ADMIN` creates accounts from the in-app **Users** page (or `POST /api/users`),
   assigns a role, and issues a temporary password the user must change on first login.

| Role | Can do |
|------|--------|
| `ADMIN` | Everything, incl. user management + audit log |
| `PHYSICIAN` | Patients, run + confirm assessments, interop |
| `NURSE` | Patients, run assessments, HL7 ingest |
| `READONLY` | View only |

**SSO / IdP (deferred).** Federated login is intentionally not wired up yet. The integration seam is
documented at `src/server/auth/sso.ts`: any OIDC/SAML callback only has to resolve an app user and mint the
same JWT, so enabling it later is additive.

> **Schema / database sync.** The build uses `prisma db push` to make the database match
> `schema.prisma` (no migration files are shipped, so this "just works" on a fresh Render database). It
> fails safe: a future schema change that would drop data stops the deploy rather than destroying it. For a
> real production system, adopt versioned migrations instead — run `npx prisma migrate dev --name init` to
> start a migration history, commit `prisma/migrations/`, and switch the Render build to `prisma migrate
> deploy`.

> **Dev dependencies at build.** The Render build runs `npm ci --include=dev` because the build step needs
> `typescript`, `vite`, and the `@types/*` packages. Without `--include=dev`, `NODE_ENV=production` prunes
> them and the client type-check fails with missing-JSX-types errors.

---

## Key API endpoints

| Method | Path | Notes |
|--------|------|-------|
| `GET` | `/api/auth/config` | Public — tells the UI if bootstrap/open-signup are available |
| `POST` | `/api/auth/bootstrap` | Creates first ADMIN; works only while 0 users exist |
| `POST` | `/api/auth/register` | Open self-signup (READONLY); when `ALLOW_OPEN_SIGNUP=true` |
| `POST` | `/api/auth/login` | Returns JWT; rate-limited; audited |
| `POST` | `/api/auth/change-password` | Self-service; rotates admin-issued temp passwords |
| `GET` | `/api/users` | **ADMIN** — list accounts (no hashes) |
| `POST` | `/api/users` | **ADMIN** — provision account (temp password, forced change) |
| `PATCH` | `/api/users/:id` | **ADMIN** — change role / activate / deactivate |
| `POST` | `/api/users/:id/reset-password` | **ADMIN** — issue new temp password |
| `GET` | `/api/patients` | RBAC; PHI decrypted server-side to DTO |
| `POST` | `/api/patients` | PHI encrypted at rest (AES-256-GCM) |
| `POST` | `/api/assessments/run` | Runs the transparent risk engine |
| `POST` | `/api/assessments/:id/review` | Physician confirmation (human-in-the-loop) |
| `GET` | `/api/fhir/Patient/:id` | FHIR R4 Patient resource |
| `GET` | `/api/fhir/Patient/:id/$everything` | FHIR Bundle (Patient + Observations + RiskAssessments) |
| `GET` | `/api/fhir/Observation?patient=:id` | FHIR Observation search set |
| `POST` | `/api/hl7/ingest` | Parses ADT/ORU (text/plain), upserts, returns HL7 ACK |
| `POST` | `/api/hl7/parse` | Structured parse preview (no persistence) |
| `GET` | `/api/audit` | Append-only audit trail (ADMIN only) |
| `GET` | `/api/alerts` | Live clinical alerts derived from patient risk levels |
| `GET` | `/api/reports/summary?days=30` | Stats + diagnosis distribution from assessment records |
| `GET` | `/api/health` | Liveness/readiness (DB probe) |

---

## The risk engine (please read)

`src/server/services/riskEngine.ts` is **deterministic and fully explainable** — every point in a score maps
to a named clinical factor, returned so a clinician can see exactly why a score was produced. Weightings are
*illustrative* and reflect the emphases of ACOG/ESC/AHA guidance on sex-specific CVD risk. **Before any
clinical use, replace it with a clinically validated, locally calibrated model** and route it through your
regulatory/quality process. Outputs are advisory and default to human-in-the-loop (`automationTier`).

## License / use
Provided as an engineering reference. You are responsible for clinical validation, regulatory clearance,
and compliance certification before any use involving real patients or PHI.
