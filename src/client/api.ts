// Thin typed API client. Prepends VITE_API_BASE_URL (Render) or uses the Vite proxy in dev.
const RAW = import.meta.env.VITE_API_BASE_URL as string | undefined;
const BASE = RAW ? (RAW.startsWith('http') ? RAW : `https://${RAW}`) : '';

export interface User { id: string; email: string; fullName: string; role: string; specialty?: string; mustChangePassword?: boolean }
export interface ManagedUser {
  id: string; email: string; fullName: string; role: string; specialty?: string;
  active: boolean; mustChangePassword: boolean; lastLoginAt?: string | null; createdAt: string;
}
export interface Patient {
  id: string; mrn: string; firstName: string; lastName: string; birthDate: string; sex: string;
  riskLevel: 'LOW' | 'MODERATE' | 'HIGH' | 'CRITICAL';
  pregnant: boolean; gestationalWeeks?: number | null; preeclampsiaHx: boolean; hypertension: boolean;
  diabetes: boolean; fmd: boolean; pcos: boolean;
  observations?: { id: string; display: string; value: number; unit: string; effectiveAt: string }[];
  assessments?: { id: string; primaryDx: string; riskLevel: string; riskScore: number; createdAt: string }[];
}
export interface AssessmentResult {
  id: string; primaryDx: string; confidence: number; riskScore: number; riskLevel: string;
  automationTier: string;
  differentials: { label: string; probability: number }[];
  factors: { critical: string[]; moderate: string[]; protective: string[] };
  prognosis: { oneYear: number; fiveYear: number; tenYear: number; lifetime: number };
  recommendations: string[]; createdAt: string;
}

let token: string | null = sessionStorage.getItem('cardioai_token');
export function setToken(t: string | null) {
  token = t;
  if (t) sessionStorage.setItem('cardioai_token', t);
  else sessionStorage.removeItem('cardioai_token');
}
export function getToken() { return token; }

async function req<T>(path: string, opts: RequestInit = {}, raw = false): Promise<T> {
  const headers: Record<string, string> = { ...(opts.headers as Record<string, string>) };
  if (!raw) headers['Content-Type'] = 'application/json';
  if (token) headers['Authorization'] = `Bearer ${token}`;
  const res = await fetch(`${BASE}${path}`, { ...opts, headers });
  if (res.status === 401) { setToken(null); }
  const ct = res.headers.get('content-type') ?? '';
  const payload = ct.includes('json') ? await res.json() : await res.text();
  if (!res.ok) throw new Error((payload as any)?.error ?? `Request failed (${res.status})`);
  return payload as T;
}

export const api = {
  login: (email: string, password: string) =>
    req<{ token: string; user: User }>('/api/auth/login', { method: 'POST', body: JSON.stringify({ email, password }) }),
  me: () => req<{ user: User }>('/api/auth/me'),
  patients: () => req<{ patients: Patient[] }>('/api/patients'),
  patient: (id: string) => req<{ patient: Patient }>(`/api/patients/${id}`),
  createPatient: (data: Record<string, unknown>) =>
    req<{ patient: Patient }>('/api/patients', { method: 'POST', body: JSON.stringify(data) }),
  runAssessment: (patientId: string, extra: Record<string, unknown> = {}) =>
    req<{ assessment: AssessmentResult }>('/api/assessments/run', { method: 'POST', body: JSON.stringify({ patientId, ...extra }) }),
  reviewAssessment: (id: string, note: string, confirm: boolean) =>
    req(`/api/assessments/${id}/review`, { method: 'POST', body: JSON.stringify({ note, confirm }) }),
  fhirEverything: (id: string) => req<unknown>(`/api/fhir/Patient/${id}/$everything`),
  hl7Parse: (message: string) => req<unknown>('/api/hl7/parse', { method: 'POST', body: message, headers: { 'Content-Type': 'text/plain' } }, true),
  auditLogs: () => req<{ logs: any[] }>('/api/audit'),
  // Self-service
  changePassword: (currentPassword: string, newPassword: string) =>
    req<{ ok: true }>('/api/auth/change-password', { method: 'POST', body: JSON.stringify({ currentPassword, newPassword }) }),
  // Admin user management
  users: () => req<{ users: ManagedUser[] }>('/api/users'),
  createUser: (data: { email: string; fullName: string; role: string; specialty?: string; tempPassword: string }) =>
    req<{ user: ManagedUser }>('/api/users', { method: 'POST', body: JSON.stringify(data) }),
  updateUser: (id: string, data: Partial<{ fullName: string; role: string; specialty: string; active: boolean }>) =>
    req<{ user: ManagedUser }>(`/api/users/${id}`, { method: 'PATCH', body: JSON.stringify(data) }),
  resetUserPassword: (id: string, tempPassword: string) =>
    req<{ user: ManagedUser }>(`/api/users/${id}/reset-password`, { method: 'POST', body: JSON.stringify({ tempPassword }) }),
};
