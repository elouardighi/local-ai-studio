# Known Limitations

Read this before relying on Local AI Studio for anything important. Everything
here was observed on the machine this was built on (RX 7800 XT, 16 GB VRAM,
Windows 11, Python 3.12).

---

## Context length is bounded by VRAM, not by the model

This is the single most important limitation, and the easiest to get wrong.

A model's GGUF metadata advertises a context length — 32k, 128k, 256k, sometimes
"150k". **That number is a statement about the architecture, not about your
GPU.** The KV cache grows linearly with context, and it has to live somewhere.

What actually fits on a 16 GB card, measured with this project's own memory
planner at full GPU offload:

| Model | Weights | Practical context |
|---|---|---|
| Qwen3.6-35B-A3B (IQ2_M) | 10.9 GB | ~8k |
| Qwen3.8-27B (IQ2_M) | 9.6 GB | ~8k |
| Abl 27b (IQ2_M) | 9.9 GB | ~8k |
| Bonsai-27B | 3.5 GB | ~8–16k |
| Gemma-4-E2B | 4.7 GB | ~16k+ |

The rule of thumb: **a large model on a 16 GB card gets roughly 8k tokens of
real context, not 150k.**

Declaring a larger `limit.context` in a client config (OpenCode, etc.) does not
create the memory. The client will happily build a longer prompt, and then one
of two honest things happens:

- the engine rejects it with a context-length error, or
- the memory planner refuses to load the model at all (`model_load_failed`,
  with the reason in the message).

**Do this instead:** set the client's context limit to what actually fits, and
raise `context` in the load request only for smaller models where you can
afford it. `POST /api/models/estimate` will tell you the plan before you
commit.

To get genuinely long context you need either a much larger GPU, a smaller
model, or CPU-side KV cache (`kv_on_gpu: false`) — the last of which is
functional but slow enough that it isn't pleasant.

---

## One model at a time

`max_instances` defaults to `1`. A single large model already fills most of a
16 GB card, so a second one has nowhere to live.

You can raise it in Settings, and auto-load will evict an *idle* model to make
room (`unload_idle`), but it will never evict one that's mid-generation. If you
ask for a second model while the first is busy, you get a `409 model_busy` —
not a silent stall.

Switching models costs the full load time again: ~5 s for a small model, ~30 s
for a 35B.

---

## The memory planner is a heuristic

It estimates weights, KV cache and compute buffers from GGUF metadata and
current free VRAM/RAM, then refuses plans it expects to page. That's
conservative on purpose — a refused load with a reason beats a machine that
stops responding for two minutes.

But it is an **estimate**, not a guarantee:

- GGUF metadata can be wrong or missing (unusual quants, hand-converted files).
- Compute-buffer sizing varies between llama.cpp versions.
- Another process allocating VRAM *after* the plan is made can still push you
  into paging. The planner only sees the moment it looks.
- The `risk_of_paging` verdict is based on currently available RAM. If your
  machine is unusually busy, a model that normally loads cleanly may be
  refused.

If a load is refused and you know better, pass `"force": true`. If it loads and
then the machine swaps, lower `context` or `gpu_layers`.

---

## Auto-load waits, and the wait is real

With auto-load on, the first request to an unloaded model blocks for the entire
load — 5–30 s for a large model — before generating a single token. This is
honest (no fake progress, no partial response), but it means:

- Clients with a short HTTP timeout will see a timeout, not an answer. The
  gateway's own `request_timeout_seconds` defaults to 600 s; a client like
  OpenCode waits up to 5 minutes for response headers, which is fine. A client
  with a 30 s timeout is not.
- Two clients asking for the same unloaded model at once: the first triggers
  the load and waits; the second gets `409 model_busy` and must retry.

Turn auto-load off if you'd rather get an immediate `404 model_not_loaded` and
manage loading yourself from the dashboard.

---

## Reasoning models need room

Models like the Qwen3.x family emit their thinking into `reasoning_content`
before the actual answer lands in `content`. With a small `max_tokens`, the
model can spend its entire budget reasoning and return an **empty** `content`
field. The tokens were really generated — check `usage.completion_tokens` and
read `reasoning_content`.

Use `max_tokens: 2048` or more for these models.

---

## Streaming cancellation leaves a cancelled request behind

Aborting a stream (client disconnect, or `POST /api/instances/{id}/cancel`) is
recorded as a cancelled/499 request in the request log and the engine slot is
freed immediately. The engine may still print a couple of stderr lines for a
moment after. That's the engine cleaning up, not a hang.

---

## Engine stderr can be noisy

`llama-server` prints timing and slot information to stderr during prompt
processing. It's all captured on the **Logs** page and filtered per instance,
but a long prompt produces a lot of lines. Filter by component (`engine[...]`)
or level rather than reading the raw tail.

---

## Vision models need the `mmproj` file

A multimodal model only accepts images if its companion `mmproj-*.gguf` sits in
the same directory. The scanner attaches it automatically when present; when
absent, the model loads as text-only and image content is rejected. See
[MODEL_COMPAT.md](MODEL_COMPAT.md).

---

## What is *not* implemented

Being explicit about scope, so nothing here surprises you:

- **No model download.** The scanner reads GGUF files you already have. There
  is no catalog, no HF integration, no conversion UI.
- **No fine-tuning, LoRA or embedding training.**
- **No multi-user / remote access.** It binds to `127.0.0.1`. `allow_remote`
  exists but there's no auth, rate limiting or TLS behind it — don't expose it.
- **No model quantization or conversion.** If llama.cpp can't load your file,
  convert it first.
- **No GPU passthrough management, no container support, no non-Windows
  support** (the resource monitor uses Windows PDH counters).
- **No response caching.** Every request goes to the engine.
- **Metrics are per-gateway**, not per-token-cost or per-dollar.

---

## Data and privacy

- Model files are **never** written, renamed or deleted. The scanner treats
  your model directory as read-only.
- `data\settings.json` and `data\instances.json` are plain local JSON. Instance
  history holds metadata only (state, PID, timings) — never weights or prompts.
- Logs redact API keys and bearer tokens before storing or displaying. That
  redaction is case-insensitive and covered by tests.
- Nothing leaves the machine. There is no telemetry, no update check, no
  network call of any kind.
