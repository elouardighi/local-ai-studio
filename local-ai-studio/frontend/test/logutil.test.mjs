// Unit tests for the Logs page normalizer (pure logic, no DOM).
// Run directly:  node --test frontend/test/

import test from "node:test";
import assert from "node:assert/strict";

import { normalizeLogRecord, unwrapLogPayload } from "../src/logutil.js";

const NOW = 1700000000;

test("unwrapLogPayload reads CustomEvent.detail (the Invalid Date bug)", () => {
  const record = { ts: 1, level: "info", component: "c", message: "m" };
  const ev = { detail: record }; // what app.js dispatches
  assert.equal(unwrapLogPayload(ev), record);
  assert.equal(unwrapLogPayload({ detail: null }), null);
  assert.equal(unwrapLogPayload(null), null);
  // tolerates a bare record too, for robustness
  assert.deepEqual(unwrapLogPayload(record), record);
});

test("a well-formed record passes through unchanged", () => {
  const rec = { ts: 1700000123.5, level: "error", component: "gateway",
    message: "boom", instance_id: "abc" };
  assert.deepEqual(normalizeLogRecord(rec, NOW), rec);
});

test("missing ts falls back to honest arrival time, never Invalid Date", () => {
  const out = normalizeLogRecord({ level: "info", component: "c",
    message: "m" }, NOW);
  assert.equal(out.ts, NOW);            // finite -> renders a real time
  assert.ok(Number.isFinite(out.ts));
  // non-numeric junk is treated the same as missing
  assert.equal(normalizeLogRecord({ ts: "not-a-date" }, NOW).ts, NOW);
  assert.equal(normalizeLogRecord({ ts: NaN }, NOW).ts, NOW);
  assert.equal(normalizeLogRecord({ ts: -5 }, NOW).ts, NOW);
});

test("missing level/component degrade to 'unknown', level lowercased", () => {
  const out = normalizeLogRecord({ ts: 1, message: "m" }, NOW);
  assert.equal(out.level, "unknown");
  assert.equal(out.component, "unknown");
  assert.equal(normalizeLogRecord({ ts: 1, level: "WARNING" }, NOW).level,
    "warning");
});

test("non-string message is stringified, never silently dropped", () => {
  assert.equal(normalizeLogRecord({ ts: 1, message: { code: 7 } }, NOW)
    .message, '{"code":7}');
  assert.equal(normalizeLogRecord({ ts: 1, message: 42 }, NOW).message, "42");
});

test("garbage input is dropped instead of rendered", () => {
  assert.equal(normalizeLogRecord(null, NOW), null);
  assert.equal(normalizeLogRecord(undefined, NOW), null);
  assert.equal(normalizeLogRecord("a string", NOW), null);
  assert.equal(normalizeLogRecord(42, NOW), null);
  assert.equal(unwrapLogPayload(undefined), null);
});

test("records shaped like the real WS payload normalize cleanly", () => {
  // Real payload seen on the wire: engine stderr lines from the log store.
  const wire = { ts: 1760000000.123, level: "info", component: "engine[1]",
    message: "llama_model_loader: loaded meta data with 29 key-value pairs",
    instance_id: "22096ab12cd3" };
  const out = normalizeLogRecord(wire, NOW);
  assert.equal(out.ts, wire.ts);
  assert.equal(out.level, "info");
  assert.ok(out.message.includes("llama_model_loader"));
});
