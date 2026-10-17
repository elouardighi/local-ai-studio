// Settings page: schema-driven form with validation and restart warnings.

import { api, el, toast } from "./api.js";

const $ = (sel) => document.querySelector(sel);

// Field metadata drives the form so new settings need no new markup.
const SCHEMA = [
  { section: "Model discovery", fields: [
    { key: "model_dirs", label: "Model directories (one per line)", type: "list" },
    { key: "converted_dir", label: "Converted models directory", type: "text" },
  ]},
  { section: "API gateway", restart: true, fields: [
    { key: "api_host", label: "API host", type: "text" },
    { key: "api_port", label: "API port", type: "number" },
    { key: "api_key", label: "API key (leave empty to disable auth)", type: "password" },
    { key: "allow_remote", label: "Allow binding to non-loopback interfaces", type: "checkbox",
      warn: "Only enable this on a network you trust. The API has no rate limiting by default." },
    { key: "cors_origins", label: "CORS origins (one per line)", type: "list" },
  ]},
  { section: "Engine", restart: true, fields: [
    { key: "engine_path", label: "llama-server executable", type: "text" },
    { key: "engine_backend", label: "Backend", type: "select", options: ["vulkan", "cpu"] },
    { key: "engine_port_start", label: "Engine port range start", type: "number" },
    { key: "engine_port_end", label: "Engine port range end", type: "number" },
  ]},
  { section: "Load defaults", fields: [
    { key: "default_ngl", label: "Default GPU layers", type: "number" },
    { key: "default_ctx", label: "Default context length", type: "number" },
    { key: "default_threads", label: "CPU threads (0 = auto)", type: "number" },
    { key: "default_batch", label: "Batch size", type: "number" },
    { key: "default_ubatch", label: "Micro-batch size", type: "number" },
    { key: "flash_attention", label: "Flash attention", type: "select", options: ["auto", "on", "off"] },
    { key: "kv_on_gpu", label: "KV cache on GPU", type: "checkbox" },
    { key: "max_instances", label: "Max concurrent instances", type: "number" },
    { key: "auto_load_on_request", label: "Auto-load a model when a request names it", type: "checkbox" },
    { key: "idle_unload_minutes", label: "Unload after idle minutes (0 = never)", type: "number" },
    { key: "advanced_args", label: "Advanced engine arguments (one per line, allowlisted)", type: "list" },
  ]},
  { section: "Memory guardrails", fields: [
    { key: "auto_adjust_offload", label: "Automatically reduce offload/context when memory is short", type: "checkbox" },
    { key: "reserve_ram_mb", label: "Reserve this much RAM (MB)", type: "number" },
    { key: "warn_commit_ratio", label: "Warn above commit usage ratio", type: "number" },
  ]},
  { section: "Monitoring & logs", fields: [
    { key: "sample_interval_seconds", label: "Sampling interval (seconds)", type: "number" },
    { key: "history_max_samples", label: "Max metric samples kept in memory", type: "number" },
    { key: "log_max_lines", label: "Max log lines kept", type: "number" },
  ]},
];

let settings = {};

export function initSettings() {
  $("#settings-save").addEventListener("click", save);
  $("#settings-reset").addEventListener("click", reset);
  load();
}

async function load() {
  try {
    const data = await api.get("/api/settings");
    settings = data.settings;
    renderForm(data.requires_restart || []);
  } catch (err) {
    toast(`Could not load settings: ${err.message}`, "error");
  }
}

function renderForm(restartFields) {
  const host = $("#settings-form");
  host.replaceChildren();

  for (const group of SCHEMA) {
    const grid = el("div", { class: "settings-grid" });
    for (const field of group.fields) {
      grid.append(fieldNode(field, restartFields.includes(field.key)));
    }
    host.append(el("div", { class: "card settings-section" }, [
      el("h2", {}, [
        group.section,
        group.restart ? el("span", { class: "pill small warn", text: "restart required" }) : null,
      ]),
      grid,
    ]));
  }
}

function fieldNode(field, needsRestart) {
  const id = `set-${field.key}`;
  const value = settings[field.key];
  let control;

  if (field.type === "checkbox") {
    return el("label", { class: "checkbox" }, [
      el("input", { type: "checkbox", id, checked: value ? "" : null,
        "data-key": field.key, "data-type": "checkbox" }),
      ` ${field.label}`,
      needsRestart ? el("span", { class: "pill small warn", text: "restart" }) : null,
      field.warn ? el("span", { class: "hint", text: field.warn }) : null,
    ]);
  }

  if (field.type === "list") {
    control = el("textarea", { id, rows: "3", "data-key": field.key, "data-type": "list" });
    control.value = (value || []).join("\n");
  } else if (field.type === "select") {
    control = el("select", { id, "data-key": field.key, "data-type": "select" });
    for (const opt of field.options) {
      control.append(el("option", { value: opt, text: opt, selected: opt === value ? "" : null }));
    }
  } else if (field.type === "password") {
    control = el("input", { type: "password", id, placeholder: value ? "•••••• (set)" : "not set",
      "data-key": field.key, "data-type": "password" });
  } else {
    control = el("input", { type: field.type === "number" ? "number" : "text", id,
      "data-key": field.key, "data-type": field.type });
    control.value = value ?? "";
  }

  return el("label", { class: field.type === "list" ? "full" : "" }, [
    field.label,
    control,
    needsRestart ? el("span", { class: "pill small warn", text: "restart required" }) : null,
    field.warn ? el("span", { class: "hint", text: field.warn }) : null,
  ]);
}

async function save() {
  const payload = {};
  document.querySelectorAll("[data-key]").forEach((node) => {
    const key = node.dataset.key;
    const type = node.dataset.type;
    if (type === "checkbox") payload[key] = node.checked;
    else if (type === "list") payload[key] = node.value.split("\n").map((s) => s.trim()).filter(Boolean);
    else if (type === "number") payload[key] = Number(node.value);
    else if (type === "password") { if (node.value) payload[key] = node.value; }
    else payload[key] = node.value;
  });

  try {
    const data = await api.put("/api/settings", payload);
    settings = data.settings;
    toast("Settings saved", "success");
    // Reloading the form re-reads masked/normalized values.
    load();
  } catch (err) {
    toast(`Settings rejected: ${err.message}`, "error", 9000);
  }
}

async function reset() {
  if (!confirm("Restore every setting to its default? Your models and history are not touched.")) return;
  try {
    const data = await api.post("/api/settings/reset");
    settings = data.settings;
    toast("Defaults restored", "success");
    load();
  } catch (err) {
    toast(`Reset failed: ${err.message}`, "error");
  }
}
