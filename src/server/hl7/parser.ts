/**
 * Minimal, dependency-free HL7 v2.x parser.
 * Handles the segment/field/component hierarchy and extracts the fields
 * CardioAI cares about from ADT (admit/register) and ORU (observation result) messages.
 *
 * HL7 delimiters (from MSH-1/MSH-2): | ^ ~ \ &
 */

export interface Hl7Segment {
  name: string;
  fields: string[][]; // fields → components
}

export interface ParsedHl7 {
  messageType: string;      // e.g. "ADT^A01" or "ORU^R01"
  controlId: string;
  segments: Hl7Segment[];
  patient?: {
    mrn?: string;
    firstName?: string;
    lastName?: string;
    birthDate?: string;     // YYYYMMDD → YYYY-MM-DD
    sex?: string;
  };
  observations: { code?: string; display?: string; value?: string; unit?: string }[];
}

function splitComponents(field: string): string[] {
  return field.split('^');
}

export function parseHl7(raw: string): ParsedHl7 {
  const clean = raw.replace(/\r\n/g, '\r').replace(/\n/g, '\r').trim();
  const lines = clean.split('\r').filter(Boolean);
  if (!lines.length || !lines[0].startsWith('MSH')) {
    throw Object.assign(new Error('Not a valid HL7 message (missing MSH segment)'), { status: 400 });
  }

  const segments: Hl7Segment[] = lines.map((line) => {
    const parts = line.split('|');
    const name = parts[0];
    // MSH is special: field 1 is the encoding chars ("^~\&"), so keep raw fields aligned.
    const fields = parts.slice(1).map(splitComponents);
    return { name, fields };
  });

  const msh = segments[0];
  // MSH-9 message type is index 7 after the encoding-chars shift.
  const msgTypeComp = msh.fields[7] ?? [];
  const messageType = msgTypeComp.slice(0, 2).filter(Boolean).join('^') || 'UNKNOWN';
  const controlId = (msh.fields[8]?.[0]) ?? '';

  const result: ParsedHl7 = { messageType, controlId, segments, observations: [] };

  const pid = segments.find((s) => s.name === 'PID');
  if (pid) {
    const mrn = pid.fields[2]?.[0] || pid.fields[3]?.[0];
    const nameField = pid.fields[4] ?? [];
    const birth = pid.fields[6]?.[0];
    const sexRaw = pid.fields[7]?.[0];
    result.patient = {
      mrn,
      lastName: nameField[0],
      firstName: nameField[1],
      birthDate: birth && birth.length >= 8 ? `${birth.slice(0, 4)}-${birth.slice(4, 6)}-${birth.slice(6, 8)}` : undefined,
      sex: sexRaw === 'F' ? 'FEMALE' : sexRaw === 'M' ? 'MALE' : sexRaw ? 'OTHER' : 'UNKNOWN',
    };
  }

  for (const seg of segments.filter((s) => s.name === 'OBX')) {
    const codeField = seg.fields[2] ?? []; // OBX-3 observation identifier
    const value = seg.fields[4]?.[0];      // OBX-5 value
    const unit = seg.fields[5]?.[0];       // OBX-6 units
    result.observations.push({ code: codeField[0], display: codeField[1], value, unit });
  }

  return result;
}

// Build a minimal HL7 ACK message (application accept).
export function buildAck(controlId: string, code: 'AA' | 'AE' = 'AA', text = ''): string {
  const ts = new Date().toISOString().replace(/[-:TZ.]/g, '').slice(0, 14);
  return [
    `MSH|^~\\&|CARDIOAI|CARDIOAI|SENDER|SENDER|${ts}||ACK|${controlId}|P|2.5`,
    `MSA|${code}|${controlId}|${text}`,
  ].join('\r');
}
