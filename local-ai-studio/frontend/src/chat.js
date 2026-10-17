// Chat playground: real streaming against the OpenAI-compatible gateway.

import { api, chatCompletion, el, toast } from "./api.js";

let store;
const $ = (sel) => document.querySelector(sel);
const history = [];   // [{role, content}]
let controller = null;
let generating = false;

export function initChat(s) {
  store = s;
  $("#chat-send").addEventListener("click", send);
  $("#chat-stop").addEventListener("click", stop);
  $("#chat-clear").addEventListener("click", clear);
  $("#chat-input").addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); send(); }
  });
  window.addEventListener("studio:instances", refreshModels);
  refreshModels();
}

function refreshModels() {
  const select = $("#chat-model");
  const ready = (store.instances || []).filter((i) => ["ready", "busy"].includes(i.state));
  const previous = select.value;
  select.replaceChildren();
  if (!ready.length) {
    select.append(el("option", { value: "", text: "No model loaded — load one first" }));
    select.disabled = true;
    return;
  }
  select.disabled = false;
  for (const inst of ready) {
    select.append(el("option", { value: inst.model_name, text: `${inst.model_name} (${inst.state})` }));
  }
  if (ready.some((i) => i.model_name === previous)) select.value = previous;
}

function addMessage(role, text = "") {
  $("#chat-empty")?.remove();
  const node = el("div", { class: `msg ${role}` }, [
    el("div", { class: "role", text: role }),
    el("div", { class: "bubble", text }),
    el("div", { class: "foot" }),
  ]);
  $("#chat-messages").append(node);
  $("#chat-messages").scrollTop = $("#chat-messages").scrollHeight;
  return node;
}

async function send() {
  const input = $("#chat-input");
  const text = input.value.trim();
  const model = $("#chat-model").value;
  if (!text || !model || generating) return;

  input.value = "";
  const system = $("#chat-system").value.trim();

  const messages = [];
  if (system) messages.push({ role: "system", content: system });
  messages.push(...history, { role: "user", content: text });
  history.push({ role: "user", content: text });

  const userNode = addMessage("user", text);
  const botNode = addMessage("assistant", "");
  const bubble = botNode.querySelector(".bubble");
  const reasoning = el("div", { class: "reasoning", style: "display:none" });
  botNode.insertBefore(reasoning, bubble);
  const foot = botNode.querySelector(".foot");
  bubble.textContent = "…";

  const payload = {
    model,
    messages,
    stream: true,
    temperature: Number($("#chat-temp").value),
    top_p: Number($("#chat-topp").value),
    max_tokens: Number($("#chat-max").value),
    repeat_penalty: Number($("#chat-repeat").value),
  };

  controller = new AbortController();
  setGenerating(true);
  const started = performance.now();

  try {
    const result = await chatCompletion(payload, {
      signal: controller.signal,
      onDelta: (full) => {
        bubble.textContent = full.content || (full.reasoning ? "thinking…" : "");
        if (full.reasoning && $("#chat-reasoning").checked) {
          reasoning.style.display = "block";
          reasoning.textContent = full.reasoning;
        }
        $("#chat-messages").scrollTop = $("#chat-messages").scrollHeight;
      },
    });

    const seconds = (performance.now() - started) / 1000;
    const finalText = result.content || "";
    bubble.textContent = finalText || "(empty response — the model produced only reasoning text; raise max tokens)";
    if (result.reasoning && $("#chat-reasoning").checked) {
      reasoning.style.display = "block";
      reasoning.textContent = result.reasoning;
    }
    history.push({ role: "assistant", content: finalText });

    const tokens = result.usage?.completion_tokens;
    foot.append(
      el("span", { text: `${seconds.toFixed(2)} s` }),
      tokens ? el("span", { text: `${tokens} tokens` }) : null,
      tokens ? el("span", { text: `${(tokens / seconds).toFixed(1)} tok/s` }) : null,
      el("button", { class: "btn small", text: "Copy", onclick: () => copy(finalText) }),
    );
    $("#chat-meta").textContent =
      `${seconds.toFixed(2)} s · ${tokens ?? "?"} tokens` +
      (tokens ? ` · ${(tokens / seconds).toFixed(1)} tok/s` : "");
  } catch (err) {
    if (err.name === "AbortError") {
      bubble.textContent = bubble.textContent + " ⏹ stopped";
      foot.append(el("span", { text: "stopped by user" }));
    } else {
      bubble.textContent = "";
      foot.append(el("span", { style: "color:var(--red)", text: `error: ${err.message}` }));
      toast(`Chat failed: ${err.message}`, "error", 7000);
    }
  } finally {
    setGenerating(false);
    controller = null;
  }
}

function stop() {
  if (controller) controller.abort();
}

function setGenerating(on) {
  generating = on;
  $("#chat-send").disabled = on;
  $("#chat-stop").disabled = !on;
  $("#chat-input").disabled = on;
}

function clear() {
  history.length = 0;
  $("#chat-messages").replaceChildren(
    el("div", { class: "chat-empty", id: "chat-empty", text: "Conversation cleared." }));
  $("#chat-meta").textContent = "";
}

async function copy(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied", "success", 1500);
  } catch {
    toast("Clipboard unavailable in this browser context", "error");
  }
}
