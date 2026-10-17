// Loaded instances page: state, live measurements and lifecycle actions.

import { api, el, fmtBytes, toast, statePill } from "./api.js";

let store;
const $ = (sel) => document.querySelector(sel);

export function initInstances(s) {
  store = s;
  $("#instances-refresh")?.addEventListener("click", refresh);
  window.addEventListener("studio:instances", render);
  refresh();
}

async function refresh() {
  try {
    const data = await api.get("/api/instances");
    store.instances = data.instances || [];
    render();
  } catch (err) {
    toast(`Instances query failed: ${err.message}`, "error");
  }
}

function render() {
  const host = $("#instances-list");
  if (!host) return;
  host.replaceChildren();

  const instances = store.instances || [];
  const active = instances.filter((i) => i.state !== "unloaded");
  if (!active.length) {
    host.append(el("p", { class: "hint",
      text: "No instances loaded. Models on disk are available in the Models page; loading one creates an instance here." }));
    // Show history below, clearly separated.
    const past = instances.filter((i) => i.state === "unloaded");
    if (past.length) {
      host.append(el("h2", { text: "Session history", style: "margin-top:18px" }));
      for (const inst of past.slice(-6)) host.append(historyCard(inst));
    }
    return;
  }

  for (const inst of active) host.append(instanceCard(inst));
  const past = instances.filter((i) => i.state === "unloaded");
  if (past.length) {
    host.append(el("h2", { text: "Session history", style: "margin-top:18px" }));
    for (const inst of past.slice(-6)) host.append(historyCard(inst));
  }
}

function historyCard(inst) {
  return el("div", { class: "row-card", style: "opacity:0.65" }, [
    el("div", { class: "row-main" }, [
      el("div", { class: "row-title" }, [inst.model_name, statePill(inst.state)]),
      el("div", { class: "row-sub", text: `unloaded ${inst.unloaded_at ? new Date(inst.unloaded_at * 1000).toLocaleString() : "—"}` }),
    ]),
  ]);
}

function instanceCard(inst) {
  const m = inst.measurements || {};
  const est = inst.estimate || {};
  const loading = ["validating", "estimating_memory", "queued", "starting_engine", "loading_model"].includes(inst.state);

  const kv = el("div", { class: "kv" }, [
    field("Instance ID", inst.instance_id),
    field("State", inst.state),
    field("Engine", `${inst.engine} · ${inst.backend}`),
    field("Device", inst.device_name || (loading ? "detecting…" : "Unavailable")),
    field("PID", inst.pid ?? "—"),
    field("Internal port", inst.port ?? "—"),
    field("GPU layers", `${inst.gpu_layers ?? "—"} / ${inst.total_layers ?? "?"}`),
    field("Context", inst.context?.toLocaleString() ?? "—"),
    field("VRAM (estimated)", est.estimated_gpu_mb ? `${Math.round(est.estimated_gpu_mb)} MB` : "—"),
    field("RAM (estimated)", est.estimated_ram_mb ? `${Math.round(est.estimated_ram_mb)} MB` : "—"),
    field("Engine RSS (measured)", m.rss_mb ? `${m.rss_mb} MB` : "Unavailable"),
    field("GPU util (measured)", m.gpu_util_percent !== null && m.gpu_util_percent !== undefined ? `${m.gpu_util_percent} %` : "Unavailable"),
    field("Speed", inst.last_tps ? `${inst.last_tps} tok/s` : "no request yet"),
    field("Active requests", String(inst.active_requests ?? 0)),
    field("Tokens generated", (inst.tokens_generated ?? 0).toLocaleString()),
    field("Load time", inst.load_duration_s ? `${inst.load_duration_s.toFixed(1)} s` : "—"),
    field("Started", inst.started_at ? new Date(inst.started_at * 1000).toLocaleTimeString() : "—"),
  ]);

  const actions = [];
  if (loading) {
    actions.push(el("button", { class: "btn small danger", text: "Cancel load",
      onclick: () => act(`/api/instances/${inst.instance_id}/cancel`, "Cancelling…") }));
  } else {
    actions.push(el("button", { class: "btn small danger", text: "Unload",
      onclick: () => act(`/api/instances/${inst.instance_id}/unload`, "Unloading…") }));
    actions.push(el("button", { class: "btn small", text: "Restart",
      onclick: () => act(`/api/instances/${inst.instance_id}/restart`, "Restarting…") }));
  }
  actions.push(el("button", { class: "btn small", text: "View logs",
    onclick: () => viewLogs(inst) }));

  return el("div", { class: "row-card", style: "flex-direction:column;align-items:stretch" }, [
    el("div", { style: "display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:center" }, [
      el("div", { class: "row-main" }, [
        el("div", { class: "row-title" }, [inst.model_name, statePill(inst.state)]),
        el("div", { class: "row-sub", text: inst.model_path }),
      ]),
      el("div", { class: "row-actions" }, actions),
    ]),
    kv,
    inst.last_error
      ? el("div", { class: "hint", style: "color:var(--red)", text: `Last error: ${inst.last_error}` })
      : null,
  ]);
}

function field(label, value) {
  return el("div", {}, [el("span", { text: label }), el("strong", { text: String(value) })]);
}

async function act(path, pendingText) {
  toast(pendingText, "info", 2500);
  try {
    await api.post(path);
    toast("Done", "success", 2000);
    await refresh();
    window.dispatchEvent(new CustomEvent("studio:instances"));
  } catch (err) {
    toast(err.message, "error", 8000);
    await refresh();
  }
}

async function viewLogs(inst) {
  try {
    const data = await api.get(`/api/instances/${inst.instance_id}/logs?limit=60`);
    const engine = (data.engine_lines || []).map((l) => l.line).join("\n");
    const backend = (data.backend_logs || []).map((l) => `[${l.level}] ${l.message}`).join("\n");
    const text = `=== engine output (pid ${inst.pid ?? "—"}) ===\n${engine || "(no engine output captured)"}\n\n=== backend log ===\n${backend || "(none)"}`;
    const blob = new Blob([text], { type: "text/plain" });
    const a = el("a", { href: URL.createObjectURL(blob), download: `instance-${inst.instance_id}-logs.txt` });
    a.click();
    toast("Logs downloaded", "success");
  } catch (err) {
    toast(`Could not fetch logs: ${err.message}`, "error");
  }
}
