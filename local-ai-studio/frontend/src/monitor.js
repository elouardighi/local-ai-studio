// Resource monitor: bounded history rendered to canvas (no external libs).

const $ = (sel) => document.querySelector(sel);
let rangeSeconds = 60;
let timer = null;

export function initMonitor() {
  $("#monitor-range")?.addEventListener("click", (ev) => {
    const btn = ev.target.closest("button");
    if (!btn) return;
    rangeSeconds = Number(btn.dataset.seconds);
    $("#monitor-range").querySelectorAll("button").forEach((b) => b.classList.toggle("active", b === btn));
    draw();
  });
  window.addEventListener("studio:resources", draw);
  timer = setInterval(() => { if (visible()) draw(); }, 2000);
  window.addEventListener("beforeunload", () => clearInterval(timer));
  draw();
}

function visible() {
  return document.visibilityState === "visible"
    && document.getElementById("page-monitor").classList.contains("active");
}

async function draw() {
  let samples = [];
  try {
    const res = await fetch(`/api/system/metrics?seconds=${rangeSeconds}`);
    const data = await res.json();
    samples = data.samples || [];
  } catch { return; }

  if (!samples.length) {
    info("chart-vram-info", "No samples yet — the monitor collects one per second.");
    return;
  }

  // VRAM (per adapter dedicated usage summed)
  const vram = samples.map((s) => s.gpu?.dedicated_used_mb ?? null);
  const total = samples.at(-1)?.gpu?.total_mb ?? null;
  line("chart-vram", [
    { values: vram, color: "#4d9fec", fill: true },
  ], { suffix: " MB", max: total || undefined });
  info("chart-vram-info", total
    ? `peak ${Math.max(...vram.filter((v) => v !== null)).toFixed(0)} MB of ${Math.round(total)} MB`
    : "GPU total unknown; showing used MB only");

  // GPU utilization
  const util = samples.map((s) => {
    const byLuid = s.gpu?.utilization_by_luid;
    if (!byLuid) return null;
    const vals = Object.values(byLuid).map(Number).filter((v) => !Number.isNaN(v));
    return vals.length ? vals.reduce((a, b) => a + b, 0) : null;
  });
  line("chart-gpu", [{ values: util, color: "#a371f7", fill: true }], { suffix: " %", max: 100 });
  const anyUtil = util.some((v) => v !== null);
  info("chart-gpu-info", anyUtil
    ? "sum of GPU engine utilization across adapters"
    : "Unavailable: Windows GPU counters did not report utilization");

  // RAM + commit
  line("chart-mem", [
    { values: samples.map((s) => (s.ram_total_bytes - s.ram_available_bytes) / 1073741824), color: "#3fb950", fill: true },
    { values: samples.map((s) => (s.commit_used_bytes ?? 0) / 1073741824), color: "#d29922" },
  ], { suffix: " GB" });
  const last = samples.at(-1);
  info("chart-mem-info", last
    ? `RAM ${(last.ram_used_percent).toFixed(0)}% · commit ${last.commit_used_percent?.toFixed?.(0) ?? "?"}% of limit`
    : "");

  // CPU
  line("chart-cpu", [{ values: samples.map((s) => s.cpu_percent), color: "#f85149", fill: true }], { suffix: " %", max: 100 });
  info("chart-cpu-info", `${samples.length} samples over ${rangeSeconds}s window`);
}

function info(id, text) {
  const node = document.getElementById(id);
  if (node) node.textContent = text || "";
}

/** Minimal multi-series line chart with an optional filled area. */
function line(canvasId, series, { suffix = "", max } = {}) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || 400;
  const height = Number(canvas.getAttribute("height")) || 140;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.scale(dpr, dpr);
  ctx.clearRect(0, 0, width, height);

  const pad = { l: 44, r: 10, t: 10, b: 18 };
  const plotW = width - pad.l - pad.r;
  const plotH = height - pad.t - pad.b;

  const all = series.flatMap((s) => s.values.filter((v) => v !== null && !Number.isNaN(v)));
  if (!all.length) {
    ctx.fillStyle = "#656d76";
    ctx.font = "12px Segoe UI";
    ctx.fillText("Unavailable", pad.l, height / 2);
    return;
  }
  const dataMax = Math.max(...all);
  const dataMin = Math.min(0, ...all);
  const top = max ?? (dataMax * 1.15 || 1);

  // grid + y labels
  ctx.strokeStyle = "#22272e";
  ctx.fillStyle = "#656d76";
  ctx.font = "10px Consolas";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = pad.t + (plotH / 4) * i;
    ctx.beginPath();
    ctx.moveTo(pad.l, y);
    ctx.lineTo(width - pad.r, y);
    ctx.stroke();
    const value = top - ((top - dataMin) / 4) * i;
    ctx.fillText(`${value.toFixed(0)}${suffix}`, 4, y + 3);
  }

  for (const s of series) {
    const points = s.values.map((v, i) => ({ v, i })).filter((p) => p.v !== null);
    if (!points.length) continue;
    const x = (i) => pad.l + (plotW * i) / Math.max(1, s.values.length - 1);
    const y = (v) => pad.t + plotH - ((v - dataMin) / (top - dataMin || 1)) * plotH;

    ctx.beginPath();
    points.forEach((p, idx) => {
      if (idx === 0) ctx.moveTo(x(p.i), y(p.v));
      else ctx.lineTo(x(p.i), y(p.v));
    });
    if (s.fill) {
      ctx.save();
      ctx.lineTo(x(points.at(-1).i), pad.t + plotH);
      ctx.lineTo(x(points[0].i), pad.t + plotH);
      ctx.closePath();
      ctx.fillStyle = hexA(s.color, 0.15);
      ctx.fill();
      ctx.restore();
    }
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 1.6;
    ctx.stroke();
  }
}

function hexA(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${alpha})`;
}
