// Logs & events page: live tail, filtering, copy/export.

import { el } from "./api.js";
import { normalizeLogRecord, unwrapLogPayload } from "./logutil.js";

const $ = (sel) => document.querySelector(sel);
const MAX_RENDERED = 400;
let lines = [];
let live = true;

export function initLogs() {
  $("#log-search").addEventListener("input", debounce(render, 250));
  $("#log-level").addEventListener("change", render);
  $("#log-live").addEventListener("change", (ev) => { live = ev.target.checked; });
  $("#log-copy").addEventListener("click", copyAll);
  $("#log-export").addEventListener("click", exportAll);
  window.addEventListener("studio:log", onLog);
  load();
}

async function load() {
  try {
    const data = await fetch("/api/logs?limit=400").then((r) => r.json());
    lines = sanitizeAll(data.lines || []);
    render();
  } catch (err) {
    console.warn("log load failed", err);
  }
}

function onLog(ev) {
  // app.js dispatches `new CustomEvent("studio:log", {detail: record})`, so
  // the record lives on `.detail` - reading the event itself was the bug that
  // rendered every live row as "Invalid Date" with undefined fields.
  const record = normalizeLogRecord(unwrapLogPayload(ev), Date.now() / 1000);
  if (!record) return; // never render a half-formed row
  lines.push(record);
  if (lines.length > 2000) lines = lines.slice(-2000);
  if (live) render();
}

function sanitizeAll(rawLines) {
  return rawLines
    .map((l) => normalizeLogRecord(l, Date.now() / 1000))
    .filter(Boolean);
}

function render() {
  const host = $("#log-list");
  const query = $("#log-search").value.toLowerCase();
  const level = $("#log-level").value;

  const filtered = lines.filter((l) => {
    // Records are normalized on arrival; this guard means a stray malformed
    // record can never surface as "Invalid Date" again.
    if (!l || !Number.isFinite(l.ts)) return false;
    if (level && l.level !== level) return false;
    if (query && !`${l.component} ${l.message}`.toLowerCase().includes(query)) return false;
    return true;
  }).slice(-MAX_RENDERED);

  host.replaceChildren();
  if (!filtered.length) {
    host.append(el("p", { class: "hint", text: "No log lines match." }));
    return;
  }
  for (const line of filtered) {
    host.append(el("div", { class: "log-line" }, [
      el("span", { class: "log-time", text: new Date(line.ts * 1000).toLocaleTimeString() }),
      el("span", { class: `log-level ${line.level}`, text: line.level }),
      el("span", { class: "log-comp", text: line.component }),
      el("span", { class: "log-msg", text: line.instance_id ? `${line.message}` : line.message,
        title: line.instance_id || "" }),
    ]));
  }
  host.scrollTop = host.scrollHeight;
}

function asText() {
  return lines.map((l) =>
    `${new Date(l.ts * 1000).toISOString()} [${l.level}] [${l.component}]${l.instance_id ? ` [${l.instance_id}]` : ""} ${l.message}`
  ).join("\n");
}

async function copyAll() {
  try {
    await navigator.clipboard.writeText(asText());
    console.info("copied");
  } catch {
    console.warn("clipboard unavailable");
  }
}

function exportAll() {
  const blob = new Blob([asText()], { type: "text/plain" });
  const a = el("a", { href: URL.createObjectURL(blob), download: `local-ai-studio-logs-${Date.now()}.txt` });
  a.click();
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}
