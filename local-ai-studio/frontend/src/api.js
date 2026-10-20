// Shared API client: one place for fetch, errors, SSE and the WebSocket.

const BASE = "";  // same origin

export class ApiError extends Error {
  constructor(message, status, body) {
    super(message);
    this.status = status;
    this.body = body;
  }
}

async function request(path, options = {}) {
  const res = await fetch(BASE + path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  const text = await res.text();
  let data = null;
  try { data = text ? JSON.parse(text) : null; } catch { data = { raw: text }; }
  if (!res.ok) {
    const msg = data?.error?.message || data?.detail || `${res.status} ${res.statusText}`;
    throw new ApiError(msg, res.status, data);
  }
  return data;
}

export const api = {
  get: (path) => request(path),
  post: (path, body) => request(path, { method: "POST", body: body ? JSON.stringify(body) : undefined }),
  put: (path, body) => request(path, { method: "PUT", body: JSON.stringify(body) }),
  del: (path, body) => request(path, { method: "DELETE", body: body ? JSON.stringify(body) : undefined }),
  // Some HTTP stacks drop the body of a DELETE, so the parameters travel
  // in the query string as well.
  delWithQuery: (path, params) => request(`${path}${query(params)}`, { method: "DELETE" }),
};

/** Build a query string, skipping empty values. */
export function query(params) {
  const parts = [];
  for (const [key, value] of Object.entries(params || {})) {
    if (value === null || value === undefined || value === "" ||
        value === "all" || value === false) continue;
    parts.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
  }
  return parts.length ? `?${parts.join("&")}` : "";
}

// ---- OpenAI-compatible gateway (used by chat + playground) -----------------

export async function chatCompletion(payload, { onDelta, signal } = {}) {
  const streaming = Boolean(payload.stream);
  const res = await fetch("/v1/chat/completions", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal,
  });

  if (!res.ok) {
    let detail = `${res.status}`;
    try {
      const body = await res.json();
      detail = body?.error?.message || detail;
    } catch { /* keep status */ }
    throw new ApiError(detail, res.status);
  }

  if (!streaming) return res.json();

  // Parse SSE manually so we control cancellation and partial rendering.
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let full = { content: "", reasoning: "" };
  let usage = null;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();
    for (const line of lines) {
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (payload === "[DONE]") continue;
      let chunk;
      try { chunk = JSON.parse(payload); } catch { continue; }
      if (chunk.usage) usage = chunk.usage;
      const choice = chunk.choices?.[0]?.delta || {};
      if (choice.content) full.content += choice.content;
      if (choice.reasoning_content) full.reasoning += choice.reasoning_content;
      if (onDelta) onDelta(full, chunk);
    }
  }
  return { content: full.content, reasoning: full.reasoning, usage };
}

// ---- live event stream ------------------------------------------------------

export class EventStream {
  constructor({ onEvent, onStatus }) {
    this.onEvent = onEvent;
    this.onStatus = onStatus;
    this.ws = null;
    this.closed = false;
    this.retry = 0;
    this._connect();
  }

  _connect() {
    if (this.closed) return;
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    const ws = new WebSocket(`${proto}//${location.host}/api/events`);
    this.ws = ws;

    ws.onopen = () => {
      this.retry = 0;
      if (this.onStatus) this.onStatus(true);
    };
    ws.onmessage = (ev) => {
      let data;
      try { data = JSON.parse(ev.data); } catch { return; }
      if (this.onEvent) this.onEvent(data);
    };
    ws.onclose = () => {
      if (this.onStatus) this.onStatus(false);
      if (this.closed) return;
      // Exponential backoff, capped at 10 s.
      this.retry += 1;
      const delay = Math.min(1000 * 2 ** (this.retry - 1), 10000);
      setTimeout(() => this._connect(), delay);
    };
    ws.onerror = () => ws.close();
  }

  close() {
    this.closed = true;
    if (this.ws) this.ws.close();
  }
}

// ---- small DOM helpers ------------------------------------------------------

export const $ = (sel) => document.querySelector(sel);
export const $$ = (sel) => Array.from(document.querySelectorAll(sel));

export function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "html") node.innerHTML = v;
    else if (k.startsWith("on") && typeof v === "function") {
      node.addEventListener(k.slice(2).toLowerCase(), v);
    } else if (v !== null && v !== undefined) node.setAttribute(k, v);
  }
  for (const child of [].concat(children)) {
    // Booleans come from `cond && el(...)` expressions; appending false
    // would render the literal text "false" into the page.
    if (child === null || child === undefined || typeof child === "boolean") {
      continue;
    }
    node.append(child.nodeType ? child : document.createTextNode(child));
  }
  return node;
}

export function toast(message, kind = "info", ms = 3800) {
  const host = document.getElementById("toast-host");
  const node = el("div", { class: `toast ${kind}`, text: message });
  host.append(node);
  setTimeout(() => node.remove(), ms);
}

export function fmtBytes(n) {
  if (n === null || n === undefined) return "Unavailable";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let v = Number(n);
  let i = 0;
  while (v >= 1024 && i < units.length - 1) { v /= 1024; i += 1; }
  return `${v.toFixed(v >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}

export function fmtTime(ts) {
  if (!ts) return "—";
  return new Date(ts * 1000).toLocaleTimeString();
}

export function statePill(state) {
  const map = {
    ready: "ok", busy: "info", loading_model: "warn", starting_engine: "warn",
    queued: "warn", validating: "warn", estimating_memory: "warn",
    crashed: "err", engine_failed: "err", out_of_memory: "err",
    validation_failed: "err", unloading: "warn", unloaded: "",
  };
  return el("span", { class: `pill small ${map[state] || ""}`, text: state });
}
