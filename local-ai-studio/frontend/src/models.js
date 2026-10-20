// Model library: search, import folders, check compatibility, download.
//
// The page is a thin view over /api/library. Searching happens on the
// server so that a large library stays responsive, and every filter is
// combined there. Nothing on this page starts a download or a model load
// without the user asking for it in as many words.

import { api, el, fmtBytes, query, toast } from "./api.js";

let store;
const $ = (sel) => document.querySelector(sel);

const filters = {
  q: "",
  architecture: "all",
  quantization: "all",
  format: "all",
  min_size: "",
  max_size: "",
  integrity: "all",
  compatibility: "all",
  duplicates_only: false,
  sort: "size",
  order: "desc",
};

let facets = { architectures: [], quantizations: [], formats: [], integrity_statuses: [] };
let currentModel = null;
let currentPlan = null;

export function initModels(s) {
  store = s;
  $("#models-rescan")?.addEventListener("click", rescan);
  $("#model-search")?.addEventListener("input", debounce(() => {
    readFilters();
    search();
  }, 250));
  $("#library-filters")?.addEventListener("change", (ev) => {
    if (ev.target.id === "model-search") return;
    readFilters();
    search();
  });
  $("#library-import-folder")?.addEventListener("click", openFolderDialog);
  $("#library-download")?.addEventListener("click", openDownloadDialog);
  $("#model-sort")?.addEventListener("change", () => { readFilters(); search(); });
  $("#model-order")?.addEventListener("click", () => {
    filters.order = filters.order === "desc" ? "asc" : "desc";
    $("#model-order").textContent = filters.order === "desc" ? "↓" : "↑";
    search();
  });
  window.addEventListener("studio:instances", () => { syncLoaded(); render(); });
  $("#opencode-sync")?.addEventListener("click", syncOpenCode);
  refresh();
  refreshOpenCode();
}

// ---------------------------------------------------------------------------
// OpenCode
// ---------------------------------------------------------------------------

async function refreshOpenCode() {
  const note = $("#opencode-note");
  if (!note) return;
  try {
    const status = await api.get("/api/library/opencode");
    if (!status.enabled) {
      note.textContent = status.reason || "not connected";
      note.className = "hint";
      return;
    }
    if (status.error) {
      note.textContent = status.error;
      note.className = "hint err-text";
      return;
    }
    const missing = status.missing || [];
    const extra = status.extra || [];
    const inLibrary = (status.library_models || []).length;
    if (missing.length) {
      note.textContent = `${inLibrary} model(s) in the library, ${missing.length} not in ${status.path} yet`;
      note.className = "hint warn-text";
    } else if (extra.length) {
      note.textContent = `in step - ${status.count} listed, ${inLibrary} in the library. Also listed: ${extra.join(", ")} (not touched)`;
      note.className = "hint";
    } else {
      note.textContent = `in step with the library (${status.count} models) - ${status.path}`;
      note.className = "hint";
    }
  } catch (err) {
    note.textContent = `Could not check: ${err.message}`;
    note.className = "hint err-text";
  }
}

async function syncOpenCode() {
  const btn = $("#opencode-sync");
  const note = $("#opencode-note");
  if (btn) { btn.disabled = true; btn.textContent = "Syncing…"; }
  try {
    const result = await api.post("/api/library/opencode/sync");
    if (result.synced) {
      const added = result.added || [];
      toast(added.length
        ? `Added ${added.length} model(s) to OpenCode: ${added.join(", ")}`
        : "OpenCode is already in step", "success", 5000);
    } else {
      toast(result.reason || result.error || "Nothing to sync", "info");
    }
  } catch (err) {
    toast(`Sync failed: ${err.message}`, "error", 7000);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = "Sync now"; }
    await refreshOpenCode();
  }
}

function readFilters() {
  filters.q = $("#model-search")?.value || "";
  filters.architecture = $("#filter-architecture")?.value || "all";
  filters.quantization = $("#filter-quantization")?.value || "all";
  filters.format = $("#filter-format")?.value || "all";
  filters.min_size = $("#filter-min-size")?.value.trim() || "";
  filters.max_size = $("#filter-max-size")?.value.trim() || "";
  filters.integrity = $("#filter-integrity")?.value || "all";
  filters.compatibility = $("#filter-compatibility")?.value || "all";
  filters.duplicates_only = Boolean($("#filter-duplicates")?.checked);
  filters.sort = $("#model-sort")?.value || "size";
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async function refresh() {
  await search();
}

async function search() {
  const host = $("#models-list");
  if (host) host.replaceChildren(el("p", { class: "hint", text: "Searching…" }));
  try {
    const data = await api.get(`/api/library/models${query(filters)}`);
    store.models = data.models || [];
    store.lastScan = data.last_scan;
    store.scanErrors = data.scan_errors || [];
    facets = data.facets || facets;
    syncLoaded();
    renderFacets();
    updateSummary(data);
    render();
  } catch (err) {
    if (host) {
      host.replaceChildren(el("p", { class: "hint", text: `Search failed: ${err.message}` }));
    }
    toast(`Library query failed: ${err.message}`, "error");
  }
}

function syncLoaded() {
  const instances = store.instances || [];
  for (const m of store.models || []) {
    const inst =
      instances.find((i) => i.model_id === m.id &&
        ["ready", "busy", "loading_model", "starting_engine"].includes(i.state)) ||
      instances.find((i) => i.model_id === m.id);
    m.loaded = !!inst && ["ready", "busy"].includes(inst.state);
    m.instance_state = inst?.state ?? null;
    m.instance_id = inst?.instance_id ?? null;
  }
}

function updateSummary(data) {
  const host = $("#models-summary");
  if (!host) return;
  const total = data?.total ?? (store.models || []).length;
  const shown = (store.models || []).length;
  const loaded = (store.models || []).filter((m) => m.loaded).length;
  const problems = (store.models || []).filter((m) =>
    ["incomplete", "corrupt"].includes(m.integrity?.status)).length;
  const dupes = (store.models || []).filter((m) => m.duplicate_count > 0).length;
  host.textContent =
    (shown === total ? `${total} models` : `${shown} of ${total} models`) +
    ` · ${loaded} loaded` +
    (problems ? ` · ${problems} with problems` : "") +
    (dupes ? ` · ${dupes} duplicated` : "") +
    (store.lastScan ? ` · scanned ${new Date(store.lastScan * 1000).toLocaleString()}` : "") +
    ((store.scanErrors || []).length ? ` · ${store.scanErrors.length} scan warnings` : "");
}

function renderFacets() {
  fillSelect("#filter-architecture", facets.architectures, "All architectures");
  fillSelect("#filter-quantization", facets.quantizations, "All quantizations");
  fillSelect("#filter-format", facets.formats, "All formats");
  fillSelect("#filter-integrity", facets.integrity_statuses, "Any file state");
}

function fillSelect(selector, values, allLabel) {
  const sel = $(selector);
  if (!sel) return;
  const current = sel.value;
  sel.replaceChildren(el("option", { value: "all", text: allLabel }));
  for (const v of values || []) {
    sel.append(el("option", { value: v, text: v }));
  }
  sel.value = values?.includes(current) || current === "all" ? current : "all";
}

function render() {
  const host = $("#models-list");
  if (!host) return;
  host.replaceChildren();

  const items = store.models || [];
  if (!items.length) {
    host.append(el("p", { class: "hint", text: "No models match these filters. Clear the search box, or import another folder." }));
    return;
  }
  for (const model of items) host.append(modelCard(model));
}

// ---------------------------------------------------------------------------
// Cards
// ---------------------------------------------------------------------------

const INTEGRITY_STYLE = {
  ok: "ok",
  unverified: "warn",
  incomplete: "err",
  corrupt: "err",
};

function integrityPill(model) {
  const report = model.integrity || {};
  const status = report.status || "unknown";
  if (status === "ok") return null;      // the common case needs no badge
  return el("span", {
    class: `pill small ${INTEGRITY_STYLE[status] || "warn"}`,
    text: status,
    title: report.reason || "",
  });
}

function modelCard(model) {
  const caps = model.capabilities || {};
  const capTags = [
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
    integrityPill(model),
    ...capTags.map((t) => el("span", { class: "pill small", text: t })),
    model.loaded && el("span", { class: "pill small ok", text: `loaded (${model.instance_state})` }),
  ];

  const recommended = model.recommended || {};
  if (recommended.context) {
    meta.push(el("span", {
      class: `pill small ${recommended.context < 8192 ? "warn" : ""}`,
      text: `ctx ${recommended.context.toLocaleString()}`,
      title: recommended.reason || "",
    }));
  }

  const warnings = [];
  if (model.duplicate_count) {
    warnings.push(el("div", {
      class: "hint warn-text",
      text: `The same file exists in ${model.duplicate_count} other place${model.duplicate_count > 1 ? "s" : ""} - only one copy is needed.`,
    }));
  }
  if (model.variant_count) {
    warnings.push(el("div", {
      class: "hint",
      text: `${model.variant_count} other quantization${model.variant_count > 1 ? "s" : ""} of this model ${model.variant_count > 1 ? "are" : "is"} in the library.`,
    }));
  }
  const integrity = model.integrity || {};
  if (["incomplete", "corrupt"].includes(integrity.status)) {
    warnings.push(el("div", { class: "hint err-text", text: integrity.reason || "This file looks damaged." }));
  }
  if (recommended.warning) {
    warnings.push(el("div", { class: "hint warn-text", text: recommended.warning }));
  }

  const actions = el("div", { class: "row-actions" });

  const card = el("div", { class: "row-card" }, [
    el("div", { class: "row-main" }, [
      el("div", { class: "row-title" }, [model.name]),
      el("div", { class: "row-sub", text: model.path }),
      el("div", { class: "row-meta" }, meta),
      model.compatibility?.reason
        ? el("div", { class: "hint", text: model.compatibility.reason })
        : null,
      ...warnings,
    ]),
    actions,
  ]);

  if (model.compatibility?.status === "convertible") {
    actions.append(el("button", {
      class: "btn small", text: "Conversion guide",
      onclick: () => toast("Safetensors models need a GGUF conversion with llama.cpp's convert_hf_to_gguf.py. Converted files go to your converted-models directory; originals are never touched.", "info", 9000),
    }));
  } else if (model.compatibility?.status !== "error") {
    actions.append(el("button", {
      class: "btn small primary", text: model.loaded ? "Load another" : "Load",
      onclick: () => openLoadDialog(model),
    }));
  }

  // The details and check panels open inside their own card, so the card
  // is passed along rather than guessed at.
  actions.append(el("button", {
    class: "btn small", text: "Check", title: "Run the compatibility checks",
    onclick: () => showPreflight(model, card),
  }));
  actions.append(el("button", {
    class: "btn small", text: "Details",
    onclick: () => showDetails(model, card),
  }));
  actions.append(el("button", {
    class: "btn small danger", text: "Delete",
    title: "Delete this model from disk",
    onclick: () => openDeleteDialog(model),
  }));
  return card;
}

// ---------------------------------------------------------------------------
// Deleting a model
// ---------------------------------------------------------------------------

async function openDeleteDialog(model) {
  const dlg = $("#delete-dialog");
  if (!dlg) return;
  const body = dlg.querySelector(".dialog-body");
  body.innerHTML = "";
  body.append(
    el("h2", { text: `Delete ${model.name}?` }),
    el("div", { id: "delete-plan", class: "estimate", text: "Working out what would be removed…" }),
    el("div", { class: "dialog-actions" }, [
      el("button", { class: "btn", type: "button", text: "Cancel", onclick: () => dlg.close() }),
      el("button", { class: "btn danger", type: "button", id: "delete-start", text: "Delete", disabled: true, onclick: () => startDelete(model) }),
    ]),
  );
  dlg.showModal();

  try {
    const plan = await api.get(`/api/library/models/${encodeURIComponent(model.id)}/delete-plan`);
    const lines = [
      "These files would be removed:",
      ...plan.files.map(f => `  ${fmtBytes(f.size_bytes).padStart(9)}  ${f.path}`),
      "",
      `frees about ${fmtBytes(plan.total_bytes)}`,
      plan.projector
        ? `The vision projector (${fmtBytes(plan.projector.size_bytes)}) is kept unless you ask for it below.`
        : "This model has no vision projector.",
      plan.loaded
        ? "\nThis model is LOADED. Unload it first."
        : "",
      "",
      plan.recoverable
        ? "Deleted files go to the Windows Recycle Bin, so you can put them back."
        : `This one cannot be put back: ${plan.recoverable_reason}.`,
    ];
    body.querySelector("#delete-plan").textContent =
      lines.filter((l) => l !== "").join("\n");
    const start = body.querySelector("#delete-start");
    if (start) start.disabled = !!plan.loaded;
    if (plan.projector) {
      body.querySelector("#delete-plan").after(
        el("label", { class: "checkbox" }, [
          el("input", { type: "checkbox", id: "delete-mmproj" }),
          ` Also delete the vision projector (${fmtBytes(plan.projector.size_bytes)})`,
        ]));
    }
  } catch (err) {
    body.querySelector("#delete-plan").textContent = `Cannot plan the delete: ${err.message}`;
  }
}

async function startDelete(model) {
  const body = document.querySelector("#delete-dialog .dialog-body");
  const progress = body.querySelector("#delete-plan");
  const btn = body.querySelector("#delete-start");
  btn.disabled = true;
  progress.textContent = "Deleting…";
  try {
    const data = await api.delWithQuery(
      `/api/library/models/${encodeURIComponent(model.id)}`,
      { confirm: true, delete_mmproj: Boolean(body.querySelector("#delete-mmproj")?.checked) });
    toast(`Deleted ${data.model_name} (${fmtBytes(data.freed_bytes)} freed)`, "success", 5000);
    document.querySelector("#delete-dialog").close();
    await search();
  } catch (err) {
    progress.textContent = `Could not delete: ${err.message}`;
    btn.disabled = false;
    toast(`Delete failed: ${err.message}`, "error", 8000);
  }
}

function compatPill(status) {
  const map = { supported: "ok", unverified: "warn", convertible: "warn", error: "err" };
  return el("span", { class: `pill small ${map[status] || ""}`, text: status });
}

function showDetails(model, card) {
  const existing = card?.querySelector(".detail-block");
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
    `file state:      ${model.integrity?.status || "unknown"}` +
      (model.integrity?.reason ? ` - ${model.integrity.reason}` : ""),
    `context:         ${(model.recommended?.context || 0).toLocaleString()} usable` +
      (model.recommended?.max_context ? ` of ${model.recommended.max_context.toLocaleString()} the model was trained for` : ""),
    `context note:    ${model.recommended?.reason || "unknown"}` +
      (model.recommended?.warning ? `\n                 ${model.recommended.warning}` : ""),
    `fingerprint:     ${model.integrity?.signature ? model.integrity.signature.slice(0, 16) + "…" : "none"}`,
    `duplicates:      ${model.duplicate_count || 0}`,
    `variants:        ${model.variant_count || 0}`,
    `memory estimate: ${est ? `${est} MB` : "run a load estimate"}`,
    `compatibility:   ${model.compatibility?.status} — ${model.compatibility?.reason || ""}`,
    "",
    "files:",
    ...(model.files || []).map((f) => `  ${fmtBytes(f.size_bytes).padStart(9)}  ${f.path}`),
  ].join("\n");
  card?.querySelector(".row-main")?.append(
    el("pre", { class: "code-block detail-block", text: lines, style: "margin-top:10px" }));
}

// ---------------------------------------------------------------------------
// Preflight
// ---------------------------------------------------------------------------

async function showPreflight(model, card) {
  const host = card?.querySelector(".row-main");
  if (!host) return;
  const existing = host.querySelector(".detail-block");
  if (existing) { existing.remove(); return; }
  const block = el("pre", { class: "code-block detail-block", text: "Checking…" });
  host.append(block);
  try {
    const report = await api.post("/api/library/preflight", { model_id: model.id });
    const lines = [`verdict: ${report.verdict.toUpperCase()}`];
    for (const c of report.checks) {
      const mark = c.status === "ok" ? "✓" : c.status === "warn" ? "!" : "✗";
      lines.push(`${mark} ${c.name}: ${c.detail}`);
    }
    block.textContent = lines.join("\n");
  } catch (err) {
    block.textContent = `check failed: ${err.message}`;
  }
}

// ---------------------------------------------------------------------------
// Load dialog
// ---------------------------------------------------------------------------

async function openLoadDialog(model) {
  currentModel = model;
  const dlg = $("#load-dialog");
  const body = dlg.querySelector(".dialog-body");

  body.dataset.mode = "load";
  body.innerHTML = "";
  body.append(
    el("h2", { id: "load-title", text: `Load ${model.name}` }),
    el("div", { id: "load-preflight", class: "estimate", text: "Checking compatibility…" }),
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
      el("button", { class: "btn", type: "button", text: "Cancel", onclick: () => dlg.close() }),
      el("button", { class: "btn primary", type: "button", id: "load-start", text: "Load", onclick: startLoad }),
    ]),
  );

  const recompute = () => { updateEstimate(model); runPreflight(model); };
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

async function runPreflight(model) {
  const body = document.querySelector("#load-dialog .dialog-body");
  const out = body?.querySelector("#load-preflight");
  if (!out) return;
  try {
    const report = await api.post("/api/library/preflight", {
      model_id: model.id,
      context: Number(body.querySelector("#load-ctx").value),
      gpu_layers: Number(body.querySelector("#load-ngl").value),
      gpu_mode: body.querySelector("#load-gpu-mode").value,
    });
    const lines = [`verdict: ${report.verdict.toUpperCase()}`];
    for (const c of report.checks) {
      const mark = c.status === "ok" ? "✓" : c.status === "warn" ? "!" : "✗";
      lines.push(`${mark} ${c.name}: ${c.detail}`);
    }
    out.textContent = lines.join("\n");
    out.classList.toggle("danger", report.verdict === "blocked");
  } catch (err) {
    out.textContent = `check failed: ${err.message}`;
  }
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

// ---------------------------------------------------------------------------
// Folder import
// ---------------------------------------------------------------------------

async function openFolderDialog() {
  const dlg = $("#folder-dialog");
  if (!dlg) return;
  $("#folder-add")?.addEventListener("click", addFolder);
  $("#folder-browse")?.addEventListener("click", () => browseFolders());
  await renderFolders();
  dlg.showModal();
}

async function renderFolders() {
  const host = $("#folder-list");
  if (!host) return;
  host.replaceChildren();
  try {
    const data = await api.get("/api/library/folders");
    for (const f of data.folders || []) {
      host.append(el("div", { class: "row-card small" }, [
        el("div", { class: "row-main" }, [
          el("div", { class: "row-title", text: f.path }),
          el("div", { class: "row-sub", text: f.exists ? (f.imported ? "imported" : "default") : "not accessible" }),
        ]),
        el("div", { class: "row-actions" }, [
          el("button", { class: "btn small danger", type: "button", text: "Remove",
            onclick: () => removeFolder(f.path) }),
        ]),
      ]));
    }
    if (!(data.folders || []).length) {
      host.append(el("p", { class: "hint", text: "No folders yet." }));
    }
  } catch (err) {
    host.replaceChildren(el("p", { class: "hint", text: err.message }));
  }
}

async function browseFolders(path) {
  const host = $("#folder-browser");
  if (!host) return;
  host.replaceChildren(el("p", { class: "hint", text: "Reading…" }));
  try {
    const data = await api.get(`/api/library/browse${query({ path: path || "" })}`);
    host.replaceChildren();
    if (data.parent) {
      host.append(el("button", { class: "btn small", type: "button", text: "↑ up one level",
        onclick: () => browseFolders(data.parent) }));
    }
    for (const entry of data.entries || []) {
      host.append(el("button", { class: "browser-item", type: "button", text: entry.name,
        onclick: () => { $("#folder-path").value = entry.path; browseFolders(entry.path); } }));
    }
    if (!(data.entries || []).length && !data.parent) {
      host.append(el("p", { class: "hint", text: "Nothing to show." }));
    }
    host.append(el("button", { class: "btn small", type: "button", text: "Use this folder",
      onclick: () => { $("#folder-path").value = data.path; } }));
  } catch (err) {
    host.replaceChildren(el("p", { class: "hint", text: err.message }));
  }
}

async function addFolder() {
  const path = $("#folder-path").value.trim();
  if (!path) { toast("Enter or browse for a folder first", "info"); return; }
  try {
    const data = await api.post("/api/library/folders", { path });
    toast(data.already_present ? "That folder is already in the library" : `Imported ${data.path}`, "success");
    await renderFolders();
    await search();
  } catch (err) {
    toast(`Import failed: ${err.message}`, "error", 7000);
  }
}

async function removeFolder(path) {
  if (!confirm(`Stop scanning ${path}? The folder and its files are not touched.`)) return;
  try {
    await api.del("/api/library/folders", { path });
    toast("Folder removed from the library", "success");
    await renderFolders();
    await search();
  } catch (err) {
    toast(`Could not remove the folder: ${err.message}`, "error");
  }
}

// ---------------------------------------------------------------------------
// Hugging Face download
// ---------------------------------------------------------------------------

async function openDownloadDialog() {
  const dlg = $("#download-dialog");
  if (!dlg) return;
  $("#hf-search")?.addEventListener("click", searchHub);
  $("#hf-query")?.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter") { ev.preventDefault(); searchHub(); }
  });
  $("#hf-download")?.addEventListener("click", startDownload);
  $("#hf-enable")?.addEventListener("click", enableDownloads);
  await checkDownloadFeature();
  dlg.showModal();
}

async function checkDownloadFeature() {
  const note = $("#download-note");
  const enable = $("#hf-enable");
  if (!note) return;
  try {
    const data = await api.get("/api/library/downloads");
    if (data.enabled) {
      note.textContent = "Nothing is downloaded until you review a plan and approve it.";
      note.className = "hint";
      if (enable) enable.hidden = true;
    } else {
      note.textContent = "You can look at what is on the Hub. Downloading is turned off - press 'Allow downloads' to switch it on.";
      note.className = "hint";
      if (enable) enable.hidden = false;
    }
  } catch (err) {
    note.textContent = `Could not check the download setting: ${err.message}`;
  }
}

async function enableDownloads() {
  try {
    await api.put("/api/settings", { allow_hf_download: true });
    toast("Downloads are now allowed. Every file still needs your approval.", "success", 4000);
    await checkDownloadFeature();
  } catch (err) {
    toast(`Could not enable downloads: ${err.message}`, "error");
  }
}
async function searchHub() {
  const q = $("#hf-query").value.trim();
  const host = $("#hf-results");
  if (!host) return;
  if (!q) { host.replaceChildren(); return; }
  host.replaceChildren(el("p", { class: "hint", text: "Searching…" }));
  // The Hub's own filter keeps the results to repositories that actually
  // ship GGUF files - the only format that can be loaded here.
  const ggufOnly = $("#hf-gguf-only")?.checked !== false;
  try {
    const data = await api.get(`/api/library/hf/search${query({
      q, limit: 25, filter: ggufOnly ? "gguf" : "",
    })}`);
    host.replaceChildren();
    for (const r of data.results || []) {
      host.append(el("div", { class: "row-card small", onclick: () => listFiles(r.repo_id) }, [
        el("div", { class: "row-main" }, [
          el("div", { class: "row-title", text: r.repo_id }),
          el("div", { class: "row-sub", text: `${r.downloads || 0} downloads · ${r.likes || 0} likes${r.pipeline_tag ? ` · ${r.pipeline_tag}` : ""}` }),
        ]),
      ]));
    }
    if (!(data.results || []).length) {
      host.append(el("p", { class: "hint", text:
        ggufOnly
          ? `No repository with GGUF files matched "${q}". Untick the GGUF box to see the others.`
          : `Nothing found for "${q}".` }));
    }
    if (data.downloads_enabled === false) {
      host.append(el("p", { class: "hint", text: "Browsing is free. Press 'Allow downloads' to fetch one of these." }));
    }
  } catch (err) {
    host.replaceChildren(el("p", { class: "hint", text: err.message }));
  }
}

async function listFiles(repo) {
  const host = $("#hf-files");
  if (!host) return;
  host.replaceChildren(el("p", { class: "hint", text: "Reading the repository…" }));
  try {
    const data = await api.get(`/api/library/hf/files${query({ repo })}`);
    host.replaceChildren(el("h3", { text: repo, style: "margin-top:10px" }));
    for (const f of data.files || []) {
      if (!f.is_model) continue;
      const row = el("div", { class: "row-card small" }, [
        el("div", { class: "row-main" }, [
          el("div", { class: "row-title", text: f.filename }),
          el("div", { class: "row-sub", text: `${f.size_human}${f.hash ? ` · ${f.hash_algorithm}: ${f.hash.slice(0, 16)}…` : " · no hash published"}` }),
        ]),
        el("div", { class: "row-actions" }, [
          el("button", { class: "btn small", type: "button", text: "Prepare",
            onclick: () => prepareDownload(repo, f) }),
        ]),
      ]);
      host.append(row);
    }
    if (!(data.files || []).some((f) => f.is_model)) {
      host.append(el("p", { class: "hint", text:
        "This repository has no .gguf files. It probably stores the weights " +
        "in another format, which has to be converted to GGUF first." }));
    }
  } catch (err) {
    host.replaceChildren(el("p", { class: "hint", text: err.message }));
  }
}

async function prepareDownload(repo, file) {
  const host = $("#download-plan");
  const btn = $("#hf-download");
  currentPlan = null;
  if (btn) btn.disabled = true;
  if (!host) return;
  host.replaceChildren(el("p", { class: "hint", text: "Working out the details…" }));
  try {
    const data = await api.post("/api/library/downloads/plan", {
      repo_id: repo, filename: file.path,
    });
    currentPlan = data.plan;
    const p = data.plan;
    const lines = [
      `file:          ${p.filename}`,
      `size:          ${p.size_human}`,
      `to:            ${p.destination}`,
      `hash:          ${p.hash ? `${p.hash_algorithm}: ${p.hash}` : "none published - size only"}`,
      `free space:    ${fmtBytes(p.free_bytes)} (needs about ${fmtBytes(p.required_bytes)})`,
      ...(p.warnings || []).map((w) => `⚠ ${w}`),
      "",
      data.downloads_enabled
        ? "Nothing has been downloaded yet. Press Download to fetch it."
        : "Nothing has been downloaded yet. Downloading is still switched " +
          "off - press 'Allow downloads' first, then Download.",
    ];
    host.textContent = lines.join("\n");
    if (btn) btn.disabled = false;
    if (!data.downloads_enabled) {
      const enable = $("#hf-enable");
      if (enable) enable.hidden = false;
    }
  } catch (err) {
    host.textContent = `This download cannot be prepared: ${err.message}`;
    host.classList.add("danger");
  }
}

async function startDownload() {
  const host = $("#download-plan");
  const btn = $("#hf-download");
  if (!currentPlan) return;
  btn.disabled = true;

  // The transfer needs the switch on as well as this approval.
  try {
    const state = await api.get("/api/library/downloads");
    if (!state.enabled) {
      host.textContent = "Downloads are switched off. Press 'Allow downloads' " +
        "to turn them on, then press Download again. Nothing has been fetched.";
      const enable = $("#hf-enable");
      if (enable) enable.hidden = false;
      btn.disabled = false;
      return;
    }
  } catch (err) {
    host.textContent = `Could not check the download setting: ${err.message}`;
    btn.disabled = false;
    return;
  }

  host.textContent = "Starting the download…";
  try {
    const data = await api.post("/api/library/downloads", {
      plan_id: currentPlan.plan_id, approve: true,
    });
    const id = data.download.download_id;
    const watch = setInterval(async () => {
      try {
        const list = await api.get("/api/library/downloads");
        const rec = (list.downloads || []).find((d) => d.download_id === id);
        if (!rec) return;
        const pct = rec.percent != null ? ` (${rec.percent}%)` : "";
        host.textContent = `${rec.status}${pct}`;
        if (["completed", "failed", "cancelled"].includes(rec.status)) {
          clearInterval(watch);
          if (rec.status === "completed") {
            host.textContent = `Downloaded to ${rec.destination}. Rescanning the library…`;
            toast("Download finished", "success");
            await search();
          } else {
            host.textContent = `${rec.status}: ${rec.error || "no reason recorded"}`;
            toast(`Download ${rec.status}`, "error", 8000);
          }
        }
      } catch { /* keep watching */ }
    }, 1000);
  } catch (err) {
    host.textContent = `The download was refused: ${err.message}`;
    btn.disabled = false;
  }
}

// ---------------------------------------------------------------------------

async function rescan() {
  const btn = $("#models-rescan");
  btn.disabled = true;
  btn.textContent = "Scanning…";
  try {
    const data = await api.post("/api/library/rescan");
    toast(`Found ${data.count} models`, "success");
    await search();
  } catch (err) {
    toast(`Rescan failed: ${err.message}`, "error");
  } finally {
    btn.disabled = false;
    btn.textContent = "Rescan";
  }
}

function debounce(fn, ms) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}
