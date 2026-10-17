// Pure helpers for the Logs page - no DOM, no imports, unit-tested in Node.
//
// The WebSocket pushes `{type: "log", data: {...}}`; app.js re-dispatches the
// record as `new CustomEvent("studio:log", {detail: record})`, so listeners
// must unwrap `.detail`. Anything that still slips through (a raw record, a
// half-built object) is normalised here rather than rendered as
// "Invalid Date" with undefined fields.

/**
 * Unwrap the payload from a CustomEvent, tolerating a bare record.
 * Returns `null` when there is nothing usable to show.
 */
export function unwrapLogPayload(ev) {
  if (ev == null) return null;
  if (typeof ev === "object" && "detail" in ev) return ev.detail ?? null;
  return ev;
}

/**
 * Coerce one log record into a renderable shape.
 *
 * - non-objects are dropped (return null),
 * - a missing/non-numeric `ts` falls back to `nowSec` (the honest moment the
 *   frontend received it, never a fabricated timestamp),
 * - a missing `level`/`component` become "unknown",
 * - a non-string `message` is JSON-stringified so nothing is silently hidden.
 */
export function normalizeLogRecord(raw, nowSec) {
  if (raw == null || typeof raw !== "object") return null;

  const ts = Number(raw.ts);
  const level = typeof raw.level === "string" && raw.level
    ? raw.level.toLowerCase()
    : "unknown";
  const component = typeof raw.component === "string" && raw.component
    ? raw.component
    : "unknown";
  const message = typeof raw.message === "string"
    ? raw.message
    : safeStringify(raw.message ?? raw);
  const instanceId = typeof raw.instance_id === "string" ? raw.instance_id : "";

  return {
    ts: Number.isFinite(ts) && ts > 0 ? ts : Number(nowSec) || 0,
    level,
    component,
    message,
    instance_id: instanceId,
  };
}

function safeStringify(value) {
  try {
    return String(JSON.stringify(value) ?? String(value));
  } catch {
    return String(value);
  }
}
