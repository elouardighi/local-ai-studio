// Application shell: routing, live events, dashboard rendering.

import { api, EventStream, el, $, $$, fmtBytes, statePill, toast } from "./api.js";
import { initModels } from "./models.js";
import { initInstances } from "./instances.js";
import { initChat } from "./chat.js";
import { initPlayground } from "./playground.js";
import { initMonitor } from "./monitor.js";
import { initLogs } from "./logs.js";
import { initSettings } from "./settings.js";

const store = {
  instances: [],
  models: [],
  resources: null,
  system: null,
  listeners: [],
};

// Exposed for support/debugging from the browser console.
window.__studio = store;

export function onStoreChange(fn) { store.listeners.push(fn); }
export function getStore() { return store; }

// ---------------------------------------------------------------- routing ---

function route(page) {
  $$(".nav-item").forEach((b) => b.classList.toggle("active", b.dataset.page === page));
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${page}`));
  location.hash = page;
  window.dispatchEvent(new CustomEvent("page:change", { detail: page }));
}

$("#nav").addEventListener("click", (ev) => {
  const btn = ev.target.closest(".nav-item");
  if (btn) route(btn.dataset.page);
});

// -------------------------------------------------------------- dashboard ---

function setMeter(fillId, textId, value, total, text) {
  const fill = document.getElementById(fillId);
  const label = document.getElementById(textId);
  if (!fill || !label) return;
  if (value === null || value === undefined || total === null || total === undefined) {
    fill.style.width = "0%";
    label.textContent = "Unavailable";
    return;
  }
  const pct = Math.max(0, Math.min(100, (value / total) * 100));
  fill.style.width = `${pct.toFixed(1)}%`;
  fill.classList.toggle("danger", pct > 92);
  fill.classList.toggle("warn", pct > 80 && pct <= 92);
  label.textContent = text;
}

function renderDashboard() {
  const s = store.system;
  const r = store.resources;
  if (!s) return;

  $("#stat-api").textContent = s.api_base || "—";
  $("#stat-instances").textContent = String(s.instances_loaded ?? store.instances.filter((i) => ["ready", "busy"].includes(i.state)).length);
  $("#stat-requests").textContent = String(s.active_requests ?? 0);
  $("#side-api").textContent = s.api_base ? new URL(s.api_base).port : "—";

  const status = $("#dash-status");
  status.textContent = s.status || "unknown";
  status.className = `pill ${s.status === "ok" ? "ok" : s.status === "degraded" ? "err" : ""}`;

  // Instance list on the dashboard
  const host = $("#dash-instances");
  host.replaceChildren();
  const loaded = store.instances.filter((i) => ["ready", "busy", "loading_model", "starting_engine"].includes(i.state));
  if (!loaded.length) {
    host.append(el("p", { class: "hint", text: "Nothing loaded yet. Load a model from the Models page." }));
  } else {
    for (const inst of loaded) {
      host.append(el("div", { class: "row-card" }, [
        el("div", { class: "row-main" }, [
          el("div", { class: "row-title" }, [inst.model_name, statePill(inst.state)]),
          el("div", { class: "row-sub", text: `port ${inst.port ?? "—"} · pid ${inst.pid ?? "—"} · ${inst.gpu_layers}/${inst.total_layers ?? "?"} layers on GPU` }),
        ]),
        el("div", { class: "row-meta" }, [
          el("span", { class: "pill small", text: `ctx ${inst.context}` }),
          el("span", { class: "pill small", text: `${inst.active_requests ?? 0} req` }),
        ]),
      ]));
    }
  }

  // GPU / memory meters
  const gpu = r?.gpu || s.gpu || {};
  $("#gpu-name").textContent = gpu.name || (gpu.available === false ? "Unavailable" : "detecting…");
  const vramText = gpu.total_mb
    ? `${fmtBytes((gpu.dedicated_used_mb || 0) * 1024 * 1024)} / ${fmtBytes(gpu.total_mb * 1024 * 1024)}`
    : "Unavailable";
  setMeter("vram-fill", "vram-text", gpu.dedicated_used_mb, gpu.total_mb, vramText);

  const util = averageUtil(gpu);
  setMeter("gpu-fill", "gpu-text", util, 100, util === null ? "Unavailable" : `${util.toFixed(0)} %`);
  $("#gpu-hint").textContent = gpu.available === false
    ? "GPU counters unavailable on this system; values are not invented."
    : "";

  setMeter("ram-fill", "ram-text", r ? (r.ram_total_bytes - r.ram_available_bytes) : null,
    r?.ram_total_bytes,
    r ? `${fmtBytes(r.ram_total_bytes - r.ram_available_bytes)} / ${fmtBytes(r.ram_total_bytes)}` : "—");
  setMeter("commit-fill", "commit-text", r?.commit_used_bytes, r?.commit_limit_bytes,
    r?.commit_used_percent !== undefined && r?.commit_used_percent !== null
      ? `${fmtBytes(r.commit_used_bytes)} (${r.commit_used_percent}%)` : "Unavailable");
  setMeter("page-fill", "page-text", r?.pagefile_used_bytes, r?.pagefile_total_bytes,
    r ? `${fmtBytes(r.pagefile_used_bytes)} / ${fmtBytes(r.pagefile_total_bytes)}` : "—");

  $("#side-gpu").textContent = gpu.total_mb ? `${Math.round((gpu.dedicated_used_mb || 0) / 1024)}G` : "—";
  $("#side-ram").textContent = r ? `${Math.round((r.ram_total_bytes - r.ram_available_bytes) / 1073741824)}G` : "—";
}

function averageUtil(gpu) {
  const byLuid = gpu?.utilization_by_luid;
  if (!byLuid || !Object.keys(byLuid).length) return null;
  const vals = Object.values(byLuid).map(Number).filter((v) => !Number.isNaN(v));
  if (!vals.length) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}

function renderEvents(events) {
  const host = $("#dash-events");
  host.replaceChildren();
  const recent = (events || []).slice(-14).reverse();
  if (!recent.length) {
    host.append(el("p", { class: "hint", text: "No events yet." }));
    return;
  }
  for (const ev of recent) {
    const msg = typeof ev.data === "string" ? ev.data
      : ev.data?.message || ev.data?.model_name || JSON.stringify(ev.data).slice(0, 120);
    host.append(el("div", { class: "log-line" }, [
      el("span", { class: "log-time", text: new Date((ev.ts || Date.now() / 1000) * 1000).toLocaleTimeString() }),
      el("span", { class: `log-level ${ev.type === "log" ? (ev.data?.level || "info") : "info"}`, text: ev.type }),
      el("span", { class: "log-msg", text: String(msg).slice(0, 200) }),
    ]));
  }
}

// ------------------------------------------------------------------ boot ---

async function refreshSystem() {
  try {
    store.system = await api.get("/api/system");
    renderDashboard();
  } catch (err) {
    toast(`System query failed: ${err.message}`, "error");
  }
}

async function refreshInstances() {
  try {
    const data = await api.get("/api/instances");
    store.instances = data.instances || [];
    $("#nav-instances").textContent = String(
      store.instances.filter((i) => ["ready", "busy"].includes(i.state)).length);
    store.listeners.forEach((fn) => fn(store));
    renderDashboard();
  } catch (err) {
    console.warn("instances refresh failed", err);
  }
}

async function main() {
  // Page modules register their own event handlers.
  initModels(store);
  initInstances(store);
  initChat(store);
  initPlayground(store);
  initMonitor(store);
  initLogs(store);
  initSettings(store);

  $("#dash-refresh").addEventListener("click", () => {
    refreshSystem();
    refreshInstances();
    toast("Refreshed", "success", 1400);
  });

  window.addEventListener("hashchange", () => {
    const page = location.hash.replace("#", "");
    if (page) route(page);
  });

  await Promise.all([refreshSystem(), refreshInstances()]);
  const initial = location.hash.replace("#", "") || "dashboard";
  route(initial);

  // Live updates over WebSocket with automatic reconnect.
  const stream = new EventStream({
    onStatus: (ok) => {
      const node = $("#conn-status");
      node.textContent = ok ? "live" : "reconnecting…";
      node.className = `brand-sub ${ok ? "ok" : "err"}`;
    },
    onEvent: (event) => handleEvent(event),
  });
  window.addEventListener("beforeunload", () => stream.close());
}

function handleEvent(event) {
  switch (event.type) {
    case "snapshot":
      if (event.data.instances) {
        store.instances = event.data.instances;
        $("#nav-instances").textContent = String(
          store.instances.filter((i) => ["ready", "busy"].includes(i.state)).length);
        store.listeners.forEach((fn) => fn(store));
      }
      if (event.data.resources) store.resources = event.data.resources;
      renderEvents(event.data.recent || []);
      renderDashboard();
      window.dispatchEvent(new CustomEvent("studio:resources", { detail: event.data.resources }));
      window.dispatchEvent(new CustomEvent("studio:instances"));
      break;
    case "resources":
      store.resources = event.data;
      renderDashboard();
      window.dispatchEvent(new CustomEvent("studio:resources", { detail: event.data }));
      break;
    case "instance_state":
      if (event.data) {
        const idx = store.instances.findIndex((i) => i.instance_id === event.data.instance_id);
        if (idx >= 0) store.instances[idx] = event.data;
        else store.instances.push(event.data);
        store.listeners.forEach((fn) => fn(store));
        $("#nav-instances").textContent = String(
          store.instances.filter((i) => ["ready", "busy"].includes(i.state)).length);
        renderDashboard();
        window.dispatchEvent(new CustomEvent("studio:instances", { detail: event.data }));
      }
      break;
    case "log":
      window.dispatchEvent(new CustomEvent("studio:log", { detail: event.data }));
      if (["error", "warning"].includes(event.data?.level)) {
        // Surface failures immediately; info stays in the Logs page.
        if (event.data.level === "error") toast(event.data.message, "error", 6000);
      }
      renderEventsAppend(event);
      break;
    default:
      window.dispatchEvent(new CustomEvent("studio:event", { detail: event }));
  }
}

const eventBuffer = [];
function renderEventsAppend(event) {
  eventBuffer.push(event);
  if (eventBuffer.length > 40) eventBuffer.shift();
}

// Refresh system numbers periodically (monitor page handles chart history).
setInterval(() => { if (document.visibilityState === "visible") refreshSystem(); }, 5000);
setInterval(() => { if (document.visibilityState === "visible") refreshInstances(); }, 3000);

main().catch((err) => {
  console.error(err);
  toast(`Startup failed: ${err.message}`, "error", 10000);
});
