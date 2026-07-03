/**
 * SSO / IdP integration seam — INTENTIONALLY NOT WIRED UP YET.
 *
 * Decision (per project scope): accounts are provisioned by an ADMIN today
 * (see routes/user.routes.ts). Federated login via an external Identity Provider
 * is deferred to "later". This file marks the single place that work plugs into,
 * so the rest of the codebase already speaks in terms an IdP can satisfy.
 *
 * ── How the seam works ───────────────────────────────────────────────────────
 * The app's session currency is the JWT minted in lib/security.ts::signToken,
 * carrying { sub, email, role }. ANY authentication method only has to end by
 * resolving an application User and minting that same token. So adding SSO later
 * means implementing ONE function shape and adding two routes — nothing else in
 * the app changes.
 *
 * ── To add OIDC (e.g. Okta, Entra ID, Auth0, Google Workspace) later ─────────
 *   1. npm i openid-client
 *   2. Env: OIDC_ISSUER_URL, OIDC_CLIENT_ID, OIDC_CLIENT_SECRET, OIDC_REDIRECT_URI
 *   3. Add routes:
 *        GET  /api/auth/sso/login     -> redirect to IdP authorization endpoint
 *        GET  /api/auth/sso/callback  -> exchange code, verify id_token, then
 *                                        call resolveFederatedUser() below and
 *                                        signToken() exactly like password login.
 *   4. Decide provisioning policy via FEDERATION_MODE:
 *        - 'strict'  : only pre-provisioned emails may log in (recommended for PHI)
 *        - 'jit'     : just-in-time create a READONLY user on first SSO login,
 *                      then an admin elevates the role.
 *   5. Map IdP group/role claims -> app Role in mapIdpRoleClaim().
 *
 * ── For SAML instead of OIDC ─────────────────────────────────────────────────
 *   Use @node-saml/node-saml; the callback still ends at resolveFederatedUser().
 *
 * Until then, these throw so nothing silently half-works.
 */

// Mirrors the Role enum in prisma/schema.prisma (kept local so this stub has no
// dependency on the generated client).
export type Role = 'ADMIN' | 'PHYSICIAN' | 'NURSE' | 'READONLY';

export type FederationMode = 'strict' | 'jit';

export interface FederatedIdentity {
  email: string;
  fullName?: string;
  idpRoleClaim?: string;
}

/** Map an IdP group/role claim onto the application's Role enum. Adjust per your IdP. */
export function mapIdpRoleClaim(_claim?: string): Role {
  // e.g. switch on Okta groups: 'cardioai-admins' -> 'ADMIN', etc.
  return 'READONLY';
}

/**
 * Resolve (and, in 'jit' mode, provision) an application User from a verified
 * federated identity. Returns the user id to embed in the JWT.
 * Implement when SSO is turned on.
 */
export async function resolveFederatedUser(
  _identity: FederatedIdentity,
  _mode: FederationMode = 'strict',
): Promise<{ id: string; email: string; role: Role }> {
  throw new Error('SSO/IdP not enabled. Provision accounts via /api/users (admin) for now.');
}
