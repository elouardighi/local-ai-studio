// Models page: discovery list, filters, details and the load dialog.

import { api, el, fmtBytes, toast } from "./api.js";

let store;

export function initModels(s) {
  store = s;
  $("#models-rescan")?.addEventListener("click", rescan);
  $("#model-search")?.addEventListener("input", render);
  $("#model-filter")?.addEventListener("change", render);
  // Instance states change without a page reload (load/unload elsewhere);
  // re-join them into the model list so badges never go stale.
  window.addEventListener("studio:instances", () => {
    syncLoaded();
    updateSummary();
    render();
  });
  refresh();
}

const $ = (sel) => document.querySelector(sel);

function syncLoaded() {
  const instances = store.instances || [];
  for (const m of store.models || []) {
    // A model can have old unloaded records plus one live instance; prefer
    // the live one so the badge reflects reality.
    const inst =
      instances.find((i) => i.model_id === m.id &&
        ["ready", "busy", "loading"].includes(i.state)) ||
      instances.find((i) => i.model_id === m.id);
    m.loaded = !!inst && ["ready", "busy"].includes(inst.state);
    m.instance_state = inst?.state ?? null;
    m.instance_id = inst?.instance_id ?? null;
  }
}

function updateSummary() {
  const host = $("#models-summary");
  if (!host || !store.models) return;
  const loaded = store.models.filter((m) => m.loaded).length;
  host.textContent =
    `${store.models.length} models discovered · ${loaded} loaded` +
    (store.lastScan ? ` · scanned ${new Date(store.lastScan * 1000).toLocaleString()}` : "") +
    ((store.scanErrors || []).length ? ` · ${store.scanErrors.length} scan warnings` : "");
}

async function refresh() {
  try {
    const data = await api.get("/api/models");
    store.models = data.models || [];
    store.lastScan = data.last_scan;
    store.scanErrors = data.scan_errors || [];
    syncLoaded();
    updateSummary();
    render();
  } catch (err) {
    toast(`Model scan failed: ${err.message}`, "error");
  }
}

async function rescan() {
  const btn = $("#models-rescan");
  btn.disabled = true;
  btn.textContent = "Scanning…";
  try {
    const data = await api.post("/api/models/scan");
    toast(`Found ${data.count} models`, "success");
    await refresh();
  } catch (err) {
    toast(`Rescan failed: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Rescan";
  }
}

function render() {
  const host = $("#models-list");
  if (!host) return;
  host.replaceChildren();

  const query = ($("#model-search")?.value || "").toLowerCase();
  const filter = $("#model-filter")?.value || "all";

  const items = store.models.filter((m) => {
    if (query && !`${m.name} ${m.architecture} ${m.path}`.toLowerCase().includes(query)) return false;
    if (filter === "loaded") return m.loaded;
    if (filter === "supported") return m.compatibility?.status === "supported";
    if (filter === "unverified") return m.compatibility?.status === "unverified";
    if (filter === "convertible") return m.compatibility?.status === "convertible";
    return true;
  });

  if (!items.length) {
    host.append(el("p", { class: "hint", text: "No models match. Check the model directories in Settings, then rescan." }));
    return;
  }

  for (const model of items) host.append(modelCard(model));
}

function compatPill(status) {
  const map = { supported: "ok", unverified: "warn", convertible: "warn", error: "err" };
  return el("span", { class: `pill small ${map[status] || ""}`, text: status });
}

function modelCard(model) {
  const caps = model.capabilities || {};  const capTags = [
    caps.vision && "vision",
    caps.moe && `MoE${model.expert_count ? ` ${model.expert_used_count}/${model.expert_count}` : ""}`,
    caps.embedding && "embedding",
    caps.chat_template && "chat template",
  ].filter(Boolean);

  const meta = [
    el("span", { class: "pill small", text: model.format }),
    el("span", { class: "pill small", text: model.quantization || "quant unknown" }),
    el("span", { class: "pill small", text: model.architecture }),
    el("span", { class: "pill small", text: model.size_human }),
    compatPill(model.compatibility?.status),
    ...capTags.map((t) => el("span", { class: "pill small", text: t })),
    model.loaded && el("span", { class: "pill small ok", text: `loaded (${model.instance_state})` }),
  ];

  const actions = [];
  if (model.compatibility?.status === "convertible") {
    actions.push(el("button", {
      class: "btn small", text: "Conversion guide",
      onclick: () => toast("Safetensors models need a GGUF conversion with llama.cpp's convert_hf_to_gguf.py. Converted files go to your converted-models directory; originals are never touched.", "info", 9000),
    }));
  } else if (model.compatibility?.status !== "error") {
    actions.push(el("button", {
      class: "btn small primary", text: model.loaded ? "Load another" : "Load",
      onclick: () => openLoadDialog(model),
    }));
  }
  actions.push(el("button", {
    class: "btn small", text: "Details",
    onclick: () => showDetails(model),
  }));

  const card = el("div", { class: "row-card" }, [
    el("div", { class: "row-main" }, [
      el("div", { class: "row-title" }, [model.name]),
      el("div", { class: "row-sub", text: model.path }),
      el("div", { class: "row-meta" }, meta),
      model.compatibility?.reason
        ? el("div", { class: "hint", text: model.compatibility.reason })
        : null,
    ]),
    el("div", { class: "row-actions" }, actions),
  ]);
  // Details toggles an inline block on this card.
  const detailsBtn = actions.find((a) => a.textContent === "Details");
  if (detailsBtn) detailsBtn.onclick = () => showDetails(model, card);
  return card;
}

function showDetails(model, card) {
  // Toggle an inline detail block on the card - no shared-dialog gymnastics.
  const existing = card.querySelector(".detail-block");
  if (existing) { existing.remove(); return; }
  const est = model.memory_estimate_mb;
  const lines = [
    `name:            ${model.name}`,
    `path:            ${model.path}`,
    `format:          ${model.format}${model.gguf_version ? ` v${model.gguf_version}` : ""}`,
    `architecture:    ${model.architecture}`,
    `quantization:    ${model.quantization || "unknown"}`,
    `size:            ${model.size_human}`,
    `tensors:         ${model.tensor_count ?? "unknown"}`,
    `layers:          ${model.block_count ?? "unknown"}`,
    `context (max):   ${model.context_length ? model.context_length.toLocaleString() : "unknown"}`,
    `experts:         ${model.expert_count ? `${model.expert_used_count} used of ${model.expert_count}` : "not MoE / unknown"}`,
    `tokenizer:       ${model.tokenizer || "unknown"}`,
    `license:         ${model.license || "unknown"}`,
    `vision:          ${model.capabilities?.vision ? "yes (mmproj present)" : "no"}`,
    `chat template:   ${model.capabilities?.chat_template ? "yes" : "unknown"}`,
    `memory estimate: ${est ? `${est} MB` : "run a load estimate"}`,
    `compatibility:   ${model.compatibility?.status} — ${model.compatibility?.reason || ""}`,
    "",
    "files:",
    ...(model.files || []).map((f) => `  ${fmtBytes(f.size_bytes).padStart(9)}  ${f.path}`),
  ].join("\n");
  card.querySelector(".row-main").append(
    el("pre", { class: "code-block detail-block", text: lines, style: "margin-top:10px" }));
}

// ----------------------------------------------------------- load dialog ---

let currentModel = null;

async function openLoadDialog(model) {
  currentModel = model;
  const dlg = document.querySelector("#load-dialog");
  const body = dlg.querySelector(".dialog-body");

  // Rebuild dialog contents (it is shared with the details view).
  body.dataset.mode = "load";
  body.innerHTML = "";
  body.append(
    el("h2", { id: "load-title", text: `Load ${model.name}` }),
    el("div", { id: "load-estimate", class: "estimate", text: "Computing memory plan…" }),
    gpuModeRow(model),
    el("label", { class: "checkbox" }, [
      el("input", { type: "checkbox", id: "load-mmproj", checked: model.mmproj_path ? "" : null,
        disabled: model.mmproj_path ? null : "" }),
      ` Load vision projector (${model.mmproj_path ? "present" : "none found"})`,
    ]),
    el("label", { class: "checkbox" }, [
      el("input", { type: "checkbox", id: "load-force" }),
      " Force load even if the memory plan looks risky",
    ]),
    el("div", { class: "hint", id: "load-progress" }),
    el("div", { class: "dialog-actions" }, [
      el("button", { class: "btn", text: "Cancel", onclick: () => dlg.close() }),
      el("button", { class: "btn primary", id: "load-start", text: "Load", onclick: startLoad }),
    ]),
  );

  // Live memory estimate whenever settings change.
  const recompute = () => updateEstimate(model);
  body.querySelector("#load-gpu-mode").addEventListener("change", recompute);
  body.querySelector("#load-ctx").addEventListener("input", debounce(recompute, 350));
  body.querySelector("#load-ngl").addEventListener("input", debounce(recompute, 350));
  recompute();

  dlg.showModal();
}

function gpuModeRow(model) {
  const ctx = model.context_length ? Math.min(8192, model.context_length) : 8192;
  return el("div", { class: "row" }, [
    el("label", {}, [
      "GPU mode",
      el("select", { id: "load-gpu-mode" }, [
        el("option", { value: "auto", text: "Automatic (fit to free VRAM)" }),
        el("option", { value: "max", text: "Maximum GPU offload" }),
        el("option", { value: "manual", text: "Manual layers" }),
        el("option", { value: "cpu", text: "CPU only" }),
      ]),
    ]),
    el("label", {}, ["GPU layers", el("input", { type: "number", id: "load-ngl", value: "99", min: "0", max: "999" })]),
    el("label", {}, ["Context length", el("input", { type: "number", id: "load-ctx", value: String(ctx), min: "512", step: "512" })]),
  ]);
}

async function updateEstimate(model) {
  const body = document.querySelector("#load-dialog .dialog-body");
  const out = body.querySelector("#load-estimate");
  if (!out) return;
  const payload = {
    model_id: model.id,
    gpu_mode: body.querySelector("#load-gpu-mode").value,
    gpu_layers: Number(body.querySelector("#load-ngl").value),
    context: Number(body.querySelector("#load-ctx").value),
  };
  try {
    const data = await api.post("/api/models/estimate", payload);
    const p = data.plan;
    const lines = [
      `GPU layers:     ${data.gpu_layers} / ${p.total_layers ?? "?"}`,
      `weights:        ${(p.weights_bytes / 1073741824).toFixed(2)} GB ` +
        `(${(p.gpu_weight_bytes / 1073741824).toFixed(2)} GB on GPU, ` +
        `${(p.cpu_weight_bytes / 1073741824).toFixed(2)} GB in RAM)`,
      `KV cache:       ~${(p.kv_cache_bytes / 1048576).toFixed(0)} MB (${p.kv_cache_confidence} confidence)`,
      `compute buffers:~${p.compute_buffer_mb} MB`,
      `estimated GPU:  ${Math.round(p.estimated_gpu_mb)} MB`,
      `estimated RAM:  ${Math.round(p.estimated_ram_mb / 1024 * 100) / 100} GB`,
      `verdict:        ${p.verdict}`,
      ...p.warnings.map((w) => `⚠ ${w}`),
      "estimate only — measured values appear after loading",
    ];
    out.textContent = lines.join("\n");
    out.style.borderColor = p.verdict === "risk_of_paging" ? "var(--red)" : "";
  } catch (err) {
    out.textContent = `estimate failed: ${err.message}`;
  }
}

async function startLoad() {
  const body = document.querySelector("#load-dialog .dialog-body");
  const progress = body.querySelector("#load-progress");
  const btn = body.querySelector("#load-start");
  const payload = {
    model_id: currentModel.id,
    gpu_mode: body.querySelector("#load-gpu-mode").value,
    gpu_layers: Number(body.querySelector("#load-ngl").value),
    context: Number(body.querySelector("#load-ctx").value),
    use_mmproj: Boolean(body.querySelector("#load-mmproj")?.checked),
    force: Boolean(body.querySelector("#load-force")?.checked),
  };
  btn.disabled = true;
  progress.textContent = "Starting engine… (this can take a minute for a large model)";
  try {
    // Poll instance state so the dialog shows real progress.
    const loadPromise = api.post("/api/models/load", payload);
    const poll = setInterval(async () => {
      try {
        const data = await api.get("/api/instances");
        const inst = (data.instances || []).find((i) => i.model_id === currentModel.id
          && ["validating", "estimating_memory", "queued", "starting_engine", "loading_model"].includes(i.state));
        if (inst) progress.textContent = `State: ${inst.state}…`;
      } catch { /* ignore transient poll errors */ }
    }, 1500);

    const result = await loadPromise;
    clearInterval(poll);
    const inst = result.instance;
    progress.textContent = `Ready in ${inst.load_duration_s?.toFixed(1) ?? "?"} s (pid ${inst.pid}, port ${inst.port})`;
    toast(`${currentModel.name} loaded`, "success");
    document.querySelector("#load-dialog").close();
  } catch (err) {
    progress.textContent = `Failed: ${err.message}`;
    btn.disabled = false;
    toast(`Load failed: ${err.message}`, "error", 8000);
  }
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}
