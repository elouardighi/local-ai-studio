// API playground: send raw requests to the local gateway and show real output.

import { el, toast } from "./api.js";

const $ = (sel) => document.querySelector(sel);

export function initPlayground() {
  document.querySelectorAll("[data-test]").forEach((btn) => {
    btn.addEventListener("click", () => runTest(btn.dataset.test));
  });
  $("#pg-send").addEventListener("click", () => send());
}

async function runTest(kind) {
  if (kind === "health") return request("/health", {}, false);
  if (kind === "models") return request("/v1/models", {}, false);
  if (kind === "chat" || kind === "stream") {
    // Prefer a genuinely loaded model so the example always works.
    let model = "";
    try {
      const data = await fetch("/api/instances").then((r) => r.json());
      const ready = (data.instances || []).find((i) => ["ready", "busy"].includes(i.state));
      model = ready?.model_name || "";
    } catch { /* fall through with empty model */ }
    $("#pg-endpoint").value = "/v1/chat/completions";
    $("#pg-body").value = JSON.stringify({
      model,
      messages: [{ role: "user", content: "Hello from Local AI Studio" }],
      stream: kind === "stream",
      temperature: 0.7,
    }, null, 2);
    $("#pg-stream").checked = kind === "stream";
    return send();
  }
}

async function send() {
  const endpoint = $("#pg-endpoint").value.trim();
  const streaming = $("#pg-stream").checked;
  const out = $("#pg-response");
  const status = $("#pg-status");
  const latency = $("#pg-latency");

  let body = null;
  if (["POST", "PUT"].includes(method())) {
    try {
      body = JSON.parse($("#pg-body").value);
    } catch (err) {
      toast(`Body is not valid JSON: ${err.message}`, "error");
      return;
    }
    body.stream = streaming;
    $("#pg-body").value = JSON.stringify(body, null, 2);
  }

  renderExample(endpoint, body, streaming);

  status.textContent = "…";
  latency.textContent = "";
  out.textContent = streaming ? "" : "waiting…";
  const started = performance.now();

  try {
    const res = await fetch(endpoint, {
      method: method(),
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    status.textContent = String(res.status);
    status.className = `pill small ${res.ok ? "ok" : "err"}`;

    if (streaming && res.ok) {
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        out.textContent = buffer;
        latency.textContent = `${((performance.now() - started) / 1000).toFixed(2)} s (streaming)`;
      }
    } else {
      const text = await res.text();
      try { out.textContent = JSON.stringify(JSON.parse(text), null, 2); }
      catch { out.textContent = text; }
      latency.textContent = `${((performance.now() - started) / 1000).toFixed(2)} s`;
    }
  } catch (err) {
    status.textContent = "failed";
    status.className = "pill small err";
    out.textContent = `Request failed: ${err.message}`;
  }
}

function method() {
  const ep = $("#pg-endpoint").value.trim().toUpperCase();
  return ep.includes("CHAT") || ep.includes("COMPLETIONS") || ep.includes("EMBEDDINGS")
    || ep.includes("LOAD") || ep.includes("SCAN") ? "POST" : "GET";
}

function renderExample(endpoint, body, streaming) {
  const base = location.origin;
  const json = body ? JSON.stringify(body) : "";
  const curl = [
    `curl -X ${method()} "${base}${endpoint}"`,
    body ? `  -H "Content-Type: application/json" \\` : "",
    body ? `  -d '${json}'` : "",
  ].filter(Boolean).join("\n");

  const python = body
    ? `import requests\n\nresp = requests.${method().toLowerCase()}(\n    "${base}${endpoint}",\n    json=${json.replace(/\btrue\b/g, "True").replace(/\bfalse\b/g, "False").replace(/\bnull\b/g, "None")},\n    timeout=300,\n)\nprint(resp.json())`
    : `import requests\n\nresp = requests.get("${base}${endpoint}", timeout=10)\nprint(resp.json())`;

  const openai = body && endpoint.startsWith("/v1/")
    ? `\n\n# OpenAI Python client:\nfrom openai import OpenAI\n\nclient = OpenAI(base_url="${base}/v1", api_key="local")\nresp = client.chat.completions.create(\n    model=${JSON.stringify(body.model || "your-model")},\n    messages=${JSON.stringify(body.messages || [{"role": "user", "content": "Hello"}])},\n)\nprint(resp.choices[0].message.content)`
    : "";

  $("#pg-example").textContent =
    `# curl\n${curl}\n\n# Python\n${python}${openai}${streaming ? "\n\n# streaming uses text/event-stream; parse `data:` lines" : ""}`;
}
