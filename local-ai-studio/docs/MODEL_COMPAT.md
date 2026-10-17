# Model Compatibility

Local AI Studio runs **GGUF** models through llama.cpp. It doesn't convert,
download or quantize — it loads files you already have.

---

## What loads

Any GGUF that your llama.cpp build supports. The scanner reads the file header
(nothing is loaded into RAM) and reports:

- **architecture** — e.g. `qwen3`, `qwen35moe`, `llama`, `gemma`
- **quantization** — from the file name and header (`IQ2_M`, `Q4_K_M`, …)
- **size**, **context length**, **layer count**, **embedding size**
- a **compatibility** verdict: `supported`, `needs_mmproj`, or `error` with a
  reason

A model with `compatibility: error` won't load, and the reason tells you why
(commonly: a safetensors/shard directory with no single GGUF, which llama.cpp
can't read directly).

### Sharded GGUFs

Files named `Model-00001-of-00003.gguf` are grouped by directory and treated as
one model, keyed on the common stem. All shards must sit in the same folder.

### Directory layout

Scanning is per-directory, not per-file. Any of these works:

```
 models\Org\ModelName\ModelName-Q4_K_M.gguf
 models\Org\Qwen3.8-27B-Uncensored-GGUF\Qwen3.8-27B-Uncensored-IQ2_M.gguf
```

The model's **display name** is the GGUF file stem. That name is what you put
in the `model` field of a request, and it's what shows in the dashboard.

---

## Quantization and memory

Lower quantization = smaller file = less VRAM for weights = more room for
context. Roughly:

| Quant | Quality | Size for a 27B |
|---|---|---|
| Q8_0 | near-lossless | ~28 GB |
| Q6_K | very good | ~21 GB |
| Q4_K_M | good, common default | ~16 GB |
| Q3_K_M | usable, some degradation | ~12 GB |
| IQ2_M / Q2_K | aggressive, noticeable loss | ~7–10 GB |

**Weights are only half the story.** The KV cache and compute buffers also
need VRAM, and the KV cache scales with context length. On a 16 GB card a
10.9 GB IQ2_M model still only gets ~8k of context. See
[LIMITATIONS.md](LIMITATIONS.md#context-length-is-bounded-by-vram-not-by-the-model).

The dashboard's **estimate** (and `POST /api/models/estimate`) tells you the
plan — weights, KV, compute buffers, and a verdict — before you load.

---

## Vision models (`mmproj`)

A multimodal model needs its companion projector file to accept images:

```
 models\Org\VisionModel\
   ├── VisionModel-Q4_K_M.gguf
   └── mmproj-VisionModel-F16.gguf      ← same directory
```

The scanner attaches `mmproj-*.gguf` found in the same directory as the main
model. When attached:

- the model loads with the projector (`use_mmproj: true`, the default)
- OpenAI-style image content works:

```json
{"role": "user", "content": [
  {"type": "text", "text": "What is in this image?"},
  {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}
]}
```

Without the `mmproj` file the model still loads as text-only; image content is
rejected by the engine. A base64 data URL or an `http(s)` URL both work — the
URL is forwarded to the engine as-is.

---

## Reasoning models

Some models (notably the Qwen3.x family here) emit a `reasoning_content` field
before the final `content`. This is normal:

- The gateway forwards both fields untouched.
- The dashboard's Chat page shows the reasoning separately.
- Use `max_tokens: 2048` or more, or the model can spend its whole budget
  reasoning and return an empty answer.

Set `reasoning_format` if you want to control how the engine reports it; it's
forwarded like any other OpenAI parameter.

---

## Architecture support

Support comes entirely from the llama.cpp build in `engine\llama.cpp`. This
project pins **b11712 (llama.cpp 0.7.0-dev)** with the Vulkan backend, which
covers the common families: Llama, Qwen (incl. MoE), Gemma, Mistral, Phi,
Gemma, GLM, and others in that era's GGML registry.

If a model's architecture isn't in your build, the scanner reports it as
unsupported. Two options:

```powershell
# grab a newer official Vulkan build
.\scripts\setup.ps1 -EngineTag <newer-tag>

# or compile from source with GGML_VULKAN=ON
.\scripts\setup.ps1 -BuildFromSource
```

Then rescan from the dashboard (**Models → Rescan**) or
`POST /api/models/scan`.

---

## Troubleshooting a specific model

1. **`compatibility: error`** — read the reason in `GET /api/models/{id}`. Most
   often it's a shard directory without a proper GGUF, or a corrupted header.
2. **Fails at `estimating_memory`** — the planner refused the plan. The message
   names what didn't fit. Lower `context`, or pass `"force": true`.
3. **Fails at `starting_engine` / `loading_model`** — the engine rejected it.
   Check **Logs**, filtered to that instance, for the engine's own error.
4. **Loads but garbled output** — usually a wrong chat template for that
   architecture, or too low a quant. Try a different quant of the same model
   first; it's the most common culprit.
