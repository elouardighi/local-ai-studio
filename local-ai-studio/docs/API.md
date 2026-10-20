# API Reference

Local AI Studio exposes two APIs on `http://127.0.0.1:1234`:

| API | Base | Purpose |
|---|---|---|
| **OpenAI-compatible** | `/v1` | Chat, completions, embeddings, model listing |
| **Management** | `/api` | Model discovery, load/unload, instances, settings, logs, metrics |

Everything binds to `127.0.0.1` by default. No authentication is required
unless you set an `api_key` in **Settings**, in which case send it as
`Authorization: Bearer <key>`.

Throughout, `<BASE>` means `http://127.0.0.1:1234`.

---

## OpenAI-compatible API (`/v1`)

### `GET /v1/models`

Lists every discovered model with its load state and metadata.

```powershell
curl http://127.0.0.1:1234/v1/models
```

```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive",
      "object": "model",
      "owned_by": "local",
      "metadata": {
        "model_id": "e07e0b1d5ae79a8a",
        "quantization": "IQ2_M",
        "size_human": "10.9 GB",
        "loaded": false,
        "state": "unloaded",
        "compatibility": "supported"
      }
    }
  ]
}
```

The `id` is what you put in the `model` field of a request. The internal
`metadata.model_id` (a short hash) also works.

The `metadata.recommended` block says what the model can really do on this
machine right now:

```json
"recommended": {
  "context": 18432,
  "output": 4608,
  "reason": "about 12,191 MB for the weights and buffers, leaving 3,019 MB of VRAM for the KV cache; the model was trained for 262,144 tokens"
}
```

`context` is the largest context that fits in the free VRAM with the model
fully offloaded — **not** the number the model advertises. A client that
believes it has 150k tokens will build a prompt the engine rejects, so this
is the number to put in a client's `limit.context`.

`max_context` is the model's own ceiling - the window it was trained with.
`context` never exceeds it. When the two are close, the model is as large
as it can be; when they are far apart, the VRAM is the limit.

`warning` is present when the context is too small to hold a long
conversation, which matters for anything that summarises its own history:

> only 2,048 tokens fit here. A long conversation will not work:
> summarising it needs more room than the model has. Use a model with a
> larger context for agent work.

### `POST /v1/chat/completions`

Standard OpenAI chat shape. Extra fields are forwarded to the engine untouched,
so anything llama.cpp supports (`temperature`, `top_p`, `stop`, `mirostat`,
`reasoning_format`, …) works.

```json
{
  "model": "Bonsai-27B",
  "messages": [
    {"role": "system", "content": "You are terse."},
    {"role": "user", "content": "Say hello."}
  ],
  "max_tokens": 512,
  "temperature": 0.7,
  "stream": false
}
```

**curl**

```bash
curl http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Bonsai-27B",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 64
  }'
```

**PowerShell** — `curl` here is an alias for `Invoke-WebRequest`, so use
`curl.exe` for the literal curl above, or build the body as an object:

```powershell
$body = @{
  model      = "Bonsai-27B"
  messages   = @(@{ role = "user"; content = "Say hello." })
  max_tokens = 64
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri "http://127.0.0.1:1234/v1/chat/completions" `
  -Method Post -ContentType "application/json" -Body $body
```

**Python** (`httpx`, already a dependency of the project)

```python
import httpx

r = httpx.post("http://127.0.0.1:1234/v1/chat/completions", json={
    "model": "Bonsai-27B",
    "messages": [{"role": "user", "content": "Say hello."}],
    "max_tokens": 64,
}, timeout=300)
r.raise_for_status()
data = r.json()
print(data["choices"][0]["message"]["content"])
print(data["usage"])   # real token counts, not estimates
```

**OpenAI Python SDK**

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:1234/v1", api_key="unused")
reply = client.chat.completions.create(
    model="Bonsai-27B",
    messages=[{"role": "user", "content": "Say hello."}],
)
print(reply.choices[0].message.content)
```

#### Streaming

Set `"stream": true`. The response is standard Server-Sent Events: one
`data: <json>` chunk per token batch, terminated by `data: [DONE]`. The gateway
injects `stream_options: {"include_usage": true}` unless you set it yourself, so
the final chunk carries real token counts (it has an empty `choices` array,
exactly as OpenAI streams it).

**curl**

```bash
curl -N http://127.0.0.1:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Bonsai-27B","messages":[{"role":"user","content":"Count to five."}],"stream":true}'
```

**Python**

```python
import httpx, json

with httpx.stream("POST", "http://127.0.0.1:1234/v1/chat/completions",
                  json={"model": "Bonsai-27B",
                        "messages": [{"role": "user", "content": "Count to five."}],
                        "stream": True},
                  timeout=300) as r:
    for line in r.iter_lines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        choices = chunk.get("choices") or []
        if choices:
            delta = choices[0].get("delta", {})
            print(delta.get("content") or "", end="", flush=True)
            # reasoning models also emit delta.reasoning_content
        elif chunk.get("usage"):
            print("\nusage:", chunk["usage"])
```

**Handling reasoning models.** Some models (e.g. the Qwen3.x family here) emit
their thinking into `delta.reasoning_content` first and the final answer into
`delta.content`. If a response looks empty, read `reasoning_content`, and use
`max_tokens: 2048` or higher so the model has room to finish thinking *and*
answering.

#### Errors

Errors are JSON with an OpenAI-style envelope, plus an `X-Request-Id` header on
successes for correlating with the **Requests** page:

| Status | `type` | Meaning |
|---|---|---|
| 400 | `invalid_request_error` | Malformed body (e.g. empty `messages`) |
| 404 | `model_not_found` | No model by that name is registered |
| 404 | `model_not_loaded` | Model exists on disk but isn't loaded, and auto-load is off |
| 409 | `model_busy` | Model is loading/busy; retry shortly |
| 409 | `model_busy` | Another request is loading the same model |
| 503 | `model_load_failed` | Auto-load was attempted and failed (reason included) |
| 504 | `engine_timeout` | Engine didn't respond within `request_timeout_seconds` |

```json
{
  "error": {
    "message": "model 'Nope' exists on disk but is not loaded. Load it from the dashboard or POST /api/models/load.",
    "type": "model_not_loaded",
    "code": "model_not_loaded",
    "model_id": "e07e0b1d5ae79a8a"
  }
}
```

**Auto-load.** When **Settings → Auto-load** is on, naming a model that exists
on disk but isn't loaded makes the gateway load it and *wait*, then answer — so
coding agents and OpenAI clients can simply name a model and use it. The first
request takes as long as the load does (~20–30 s for a large model). If you'd
rather get an immediate error, turn auto-load off and you'll receive the 404
above instead.

### `POST /v1/completions` and `POST /v1/embeddings`

Same request/response contract as OpenAI, proxied to the loaded engine.

### `GET /v1/health`, `GET /health`

```json
{"status": "ok", "service": "local-ai-studio", "loaded_instances": 0, "active_requests": 0}
```

### `GET /metrics`

Prometheus text format: request totals, token totals, error counts and active
request gauges.

---

## Management API (`/api`)

### Models

#### `GET /api/models`
Discovery results with richer metadata than `/v1/models`: file path, GGUF
architecture, quantization, byte size, context length, per-layer memory
estimates, and a `compatibility` verdict.

```powershell
(Invoke-RestMethod http://127.0.0.1:1234/api/models).models |
  Select-Object name, quantization, size_human, instance_state
```

#### `POST /api/models/scan`

Re-scan the model directories. Returns the new count and any errors.

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:1234/api/models/scan
```

#### `GET /api/models/{model_id}`

Full detail for one model, including the memory plan the loader would use.

#### `POST /api/models/load`

Load a model. This is the same path the dashboard's Load button uses.

```json
{
  "model_id": "Bonsai-27B",
  "context": 8192,
  "gpu_mode": "auto",
  "gpu_layers": null,
  "kv_on_gpu": true,
  "batch": 512,
  "ubatch": 512,
  "use_mmproj": true,
  "unload_idle": false,
  "allow_duplicate": false,
  "force": false
}
```

Only `model_id` is required.

| Field | Default | Notes |
|---|---|---|
| `context` | `settings.default_ctx` | KV-cache size in tokens. Bigger = more VRAM. |
| `gpu_mode` | `auto` | `auto` picks layers from free VRAM · `max` offloads everything · `manual` uses `gpu_layers` · `cpu` disables the GPU |
| `gpu_layers` | — | Layer count, only used with `gpu_mode: manual` (or as a cap in `auto`) |
| `kv_on_gpu` | `true` | Put the KV cache on the GPU (much faster, uses VRAM) |
| `batch` / `ubatch` | settings | Token batch sizes |
| `use_mmproj` | `true` | Attach the vision projector if the model has one |
| `unload_idle` | `false` | Evict an idle model first if the concurrency limit would refuse the load |
| `allow_duplicate` | `false` | Load a second instance of a model that's already loaded |
| `force` | `false` | Load even when the memory planner warns of paging risk |

```powershell
$body = @{ model_id = "Bonsai-27B"; context = 8192 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:1234/api/models/load `
  -ContentType "application/json" -Body $body
```

The response contains the full `instance` record. Loading takes 5–30 s
depending on model size — it starts the engine, streams the weights, and only
returns once the model is `ready`.

**Before it starts anything**, the memory planner estimates weights, KV cache
and compute buffers against your currently free VRAM and RAM. If the plan would
page to disk, the load is **refused with a reason** rather than allowed to
thrash your machine. Pass `"force": true` to override.

#### `POST /api/models/unload`

Accepts either an `instance_id` or a `model_id`.

```powershell
$body = @{ model_id = "Bonsai-27B" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:1234/api/models/unload `
  -ContentType "application/json" -Body $body
```

Frees the VRAM immediately (verified: 12.5 GB returned on a large-model unload).

### Instances

#### `GET /api/instances`

Every instance record: `state`, `pid`, `port`, `gpu_layers`, `context`, timings,
and the current token counters.

Instance states follow a validated lifecycle:

```
discovered → validating → estimating_memory → queued
  → starting_engine → loading_model → ready ⇄ busy
  → unloading → unloaded
```

plus failure states `validation_failed`, `out_of_memory`, `engine_failed`,
`cancelled` and `crashed`. Transitions are checked — an illegal jump raises
rather than silently corrupting the record.

#### `POST /api/instances/{instance_id}/unload`
#### `POST /api/instances/{instance_id}/restart`
#### `POST /api/instances/{instance_id}/cancel`

Cancel aborts an in-flight generation without unloading the model; the engine
slot is freed and the next request works immediately.

#### `GET /api/instances/{instance_id}/logs`

Per-instance log slice (engine stdout/stderr).

### Settings

#### `GET /api/settings` / `PUT /api/settings`

`PUT` accepts a partial object and validates it; unknown or invalid values
return a 400 rather than being silently dropped.

```powershell
# Turn on auto-load
Invoke-RestMethod -Method Put -Uri http://127.0.0.1:1234/api/settings `
  -ContentType "application/json" -Body '{"auto_load_on_request": true}'
```

Key settings:

| Key | Default | Meaning |
|---|---|---|
| `model_dirs` | `…\.lmstudio\models` | Where to scan for GGUF files (read-only) |
| `api_host` / `api_port` | `127.0.0.1` / `1234` | Bind address |
| `api_key` | `null` | If set, required as a Bearer token |
| `engine_path` | `engine\llama.cpp\llama-server.exe` | The engine binary |
| `engine_backend` | `vulkan` | `vulkan` or `cpu` |
| `engine_port_start` / `_end` | `18400` / `18499` | Internal engine port range |
| `default_ngl` | `99` | Fallback GPU layer count |
| `default_ctx` | see settings | Fallback context length |
| `max_instances` | `1` | Concurrent loaded models |
| `auto_load_on_request` | `false` | Load-on-demand for `/v1` requests |
| `idle_unload_minutes` | `0` (never) | Auto-unload after idle time |
| `request_timeout_seconds` | `600` | Upstream engine read timeout |

#### `POST /api/settings/reset`

Restores defaults (keeps your model directories).

### Logs and events

#### `GET /api/logs?limit=&level=&component=&search=&instance_id=`

Tail the log store. Every line carries a real `ts`, a normalized `level`, a
`component` and the message (with API keys and bearer tokens redacted).

#### `GET /api/system`

Service summary: engine path/backend/device, model count, loaded instances,
active requests, system memory and the latest GPU sample.

#### `GET /api/system/gpu-devices`, `GET /api/system/metrics`
#### `GET /api/requests`

`/api/requests` shows recent requests with durations, status codes and token
counts, plus running totals.

### Live events

#### `WebSocket /api/events`

Connects and immediately receives a `snapshot` (recent events, instances, latest
resource sample), then a live stream of typed events:

| `type` | Payload |
|---|---|
| `log` | A log record (every component — settings, gateway, scanner, engine) |
| `instance_state` | A full instance record after a state change |
| `resources` | 1 Hz CPU / RAM / pagefile / VRAM sample |
| `snapshot` | Initial catch-up payload |
| `events_dropped` | Marker when a slow client falls behind |

#### `GET /api/events` (SSE)

The same stream over Server-Sent Events for clients that can't do WebSocket.

---

## Library API (`/api/library`)

The library API is the richer view of the model collection: it searches,
reports file integrity, spots duplicates and runs the checks that must pass
before a model is loaded.

### `GET /api/library/models`

Search the library. Every filter is optional and they combine with AND.

| Parameter | Meaning |
|---|---|
| `q` | Free text over name, path, architecture, quantization and parameter label |
| `architecture` | Exact architecture (`qwen35moe`, `gemma4`, …) or `all` |
| `quantization` | Exact quantization (`IQ2_M`, `Q4_K_M`, …) or `all` |
| `format` | `GGUF` or `safetensors`, or `all` |
| `min_size` / `max_size` | Size bounds: `10gb`, `500 MB`, `1.5g` or a byte count |
| `integrity` | `ok`, `unverified`, `incomplete` or `corrupt` |
| `compatibility` | `supported`, `unverified`, `convertible` or `error` |
| `duplicates_only` | Only models that exist more than once |
| `sort` | `size`, `name`, `architecture` or `quantization` |
| `order` | `asc` or `desc` |

```powershell
(Invoke-RestMethod "http://127.0.0.1:1234/api/library/models?architecture=qwen35&max_size=10gb&sort=name").models |
  Select-Object name, quantization, size_human, @{n='state';e={$_.integrity.status}}
```

The response also carries `facets` — the architectures, quantizations, formats
and file states that exist — so a UI can fill its filter controls from the
data instead of hard-coding them.

Each model record gained three fields:

- `integrity` — `status` (`ok`, `unverified`, `incomplete`, `corrupt`), a
  `reason` when something is wrong, and the `expected_bytes` the file should
  have. A file is `incomplete` when its tensor data is shorter than the
  tensor directory says it should be, which is what an interrupted download
  looks like.
- `duplicate_count` / `duplicates` — other models with the same content.
  Duplicates are found by fingerprinting each file (size plus its first and
  last mebibyte), so the same weights in two folders are recognised without
  reading ten gigabytes twice.
- `variant_count` / `variants` - other quantizations of the same base model.
  These are *not* duplicates and should not be deleted.

**A model that is not found is looked for again.** The scan results follow
the folder list, and a lookup that comes up empty triggers one more scan
before it gives up - so a model that was just downloaded, or that appeared
in a folder imported a moment ago, is found rather than reported as
missing. When it really is gone, the error says so:

```
model not found: 4f4103c593a8ad67. It is not in any scanned folder any more
- press Rescan, or import the folder it lives in.
```

### `GET /api/library/folders` · `POST` · `DELETE`

The folders the library scans. `POST` imports one:

```json
{"path": "C:\\Users\\SAKISAKI\\models"}
```

The folder is added to `model_dirs` and the library is rescanned. Nothing
inside the folder is moved, renamed or deleted. `DELETE` stops scanning a
folder; the folder and its files are left alone.

### `GET /api/library/browse?path=`

Lists the sub-folders of a path so the dashboard can offer a folder picker.
System directories (Windows, Program Files) are hidden, and a path that
cannot be read is reported rather than guessed at.

### `POST /api/library/preflight`

Runs the checks that must pass before a model is loaded.

```json
{"model_id": "e07e0b1d5ae79a8a", "context": 8192, "gpu_layers": 99, "gpu_mode": "auto"}
```

```json
{
  "verdict": "risky",
  "checks": [
    {"name": "file", "status": "ok", "detail": "11,659,235,456 bytes on disk"},
    {"name": "integrity", "status": "ok", "detail": "the file is complete"},
    {"name": "architecture", "status": "ok", "detail": "'qwen35moe' is implemented by llama.cpp 0.7.0-dev (b11712)"},
    {"name": "quantization", "status": "warn", "detail": "very aggressive - large quality loss"},
    {"name": "vision projector", "status": "ok", "detail": "attached: mmproj-...-f16.gguf"},
    {"name": "context", "status": "ok", "detail": "8,192 of 262,144 tokens (the model maximum)"},
    {"name": "memory", "status": "ok", "detail": "about 14,372 MB of VRAM and 0.5 GB of RAM"}
  ]
}
```

`verdict` is `ready`, `risky` or `blocked`. A `blocked` verdict means one of
the checks failed — the file is incomplete, the architecture is not
implemented, or the requested context does not fit.

### `POST /api/library/rescan`

Rescans every folder and rebuilds the duplicate and variant groups.

### `GET /api/library/models/{model_id}/delete-plan`

What deleting a model would remove. Nothing is deleted here — this is what
the confirmation dialog shows.

```json
{
  "model_id": "e65c6f1bee67563c",
  "model_name": "py007_tinyllama-1.1b-chat-v0.3",
  "files": [
    {"path": "...\\tinyllama-1.1b-chat-v0.3.Q2_K.gguf", "size_bytes": 482149856}
  ],
  "total_bytes": 482149856,
  "projector": null,
  "loaded": false,
  "recoverable": true,
  "recoverable_reason": null
}
```

`recoverable` is false when the file is too big for the Recycle Bin on
that drive — Windows erases such files permanently, whatever the caller
asks for, so the user is told *before* deleting rather than after.

### `DELETE /api/library/models/{model_id}`

Deletes a model from disk.

| Parameter | Meaning |
|---|---|
| `confirm` | Must be true. Nothing happens without it. |
| `delete_mmproj` | Also delete the vision projector. Off by default — a projector is never removed by accident. |
| `permanent` | Erase the file instead of using the Recycle Bin. |

The confirmation may be sent in the body or as `?confirm=true`, because
some HTTP stacks drop the body of a DELETE.

```powershell
Invoke-RestMethod -Method Delete -Uri "http://127.0.0.1:1234/api/library/models/e65c6f1bee67563c?confirm=true"
```

```json
{
  "deleted": true,
  "model_name": "py007_tinyllama-1.1b-chat-v0.3",
  "files": [{"path": "...", "size_bytes": 482149856,
             "method": "moved to the Recycle Bin", "recoverable": true}],
  "freed_bytes": 482149856,
  "permanent": false
}
```

The rules that keep this safe:

- **The path comes from the library's own record.** A caller cannot ask for
  an arbitrary file to be removed.
- **A loaded model cannot be deleted.** Unload it first (409 `busy`).
- **Nothing happens without a confirmation** (400 `confirmation_required`).
- **The file goes to the Windows Recycle Bin** by default, so it can be put
  back. A permanent delete has to be asked for.
- **The vision projector is a separate choice** and is kept unless
  `delete_mmproj` is set.
- **Only files are removed**, never a directory.

### `GET /api/library/opencode`

What the OpenCode config looks like, and whether it is in step with the
library.

```json
{
  "enabled": true,
  "path": "C:\\Users\\SAKISAKI\\.config\\opencode\\opencode.jsonc",
  "exists": true,
  "provider": "localai",
  "models": ["Bonsai-27B", "Abl 27b", "..."],
  "count": 7,
  "library_models": ["Bonsai-27B", "Abl 27b", "..."],
  "missing": []
}
```

`missing` lists the models the library knows about that the config does
not yet offer.

### `POST /api/library/opencode/sync`

Puts the two back together. Missing models are added, and models this
library deleted are taken out. Models the user listed themselves are never
touched.

```json
{"synced": true, "added": ["New-Model"], "skipped": ["Bonsai-27B"], "removed": []}
```

This is what runs automatically when a model is downloaded or deleted, so
it is only needed when the config was changed by hand.

The limits written are the ones the server computed: the context that fits
in the free VRAM, and an output budget of a quarter of that.

---

## Downloads from Hugging Face (optional)

**Browsing is always available.** Searching the Hub and listing a
repository's files are read-only metadata requests, and they work whether or
not downloading is switched on - looking at what is out there should not
require opting in to fetch gigabytes.

**Downloading is off by default.** Nothing in this project fetches a file
from the internet until `allow_hf_download` is switched on in Settings, and
even then every file is planned and shown to you before a single byte is
transferred.

| Endpoint | Purpose | Needs downloads on? |
|---|---|---|
| `GET /api/library/hf/search?q=&filter=` | Find repositories by name | no |
| `GET /api/library/hf/files?repo=` | List a repository's files with size and hash | no |
| `POST /api/library/downloads/plan` | Work out what a download would do | no |
| `POST /api/library/downloads` | Start a planned download — needs `approve: true` | yes |
| `GET /api/library/downloads` | The download log | no |
| `POST /api/library/downloads/{id}/cancel` | Stop a download | - |

`filter` is passed straight to the Hub. It defaults to `gguf`, which keeps
the results to repositories that actually ship GGUF files - the only format
that can be loaded here. Pass `filter=` to see everything.

### Planning

```json
{"repo_id": "ggml-org/models", "filename": "tinyllamas/stories260K.gguf"}
```

```json
{
  "plan": {
    "plan_id": "85bdba4693c8",
    "size_bytes": 1185376,
    "size_human": "1.1 MB",
    "destination": "...\\data\\downloads\\ggml-org\\models\\tinyllamas\\stories260K.gguf",
    "hash_algorithm": "sha256",
    "hash": "270cba1bd5109f42d03350f60406024560464db173c0e387d91f0426d3bd256d",
    "free_bytes": 82735312896,
    "required_bytes": 538079995,
    "resume_bytes": 0,
    "warnings": []
  },
  "downloads_enabled": false
}
```

Planning is a read-only metadata request, so it works whether or not
downloading is switched on - you can always see exactly what a download
would do before deciding to allow it. The plan is the whole safety story
in one object: the size, where the file would land, the hash it will be
checked against, how much room there is, and what has to be true before the
transfer can start. A download is refused when the file is not a `.gguf`,
when it is over the size limit, or when the disk does not have room for it
plus a safety margin.

`downloads_enabled` tells the caller whether the transfer would be allowed
right now. It is `false` until `allow_hf_download` is switched on.

### Approving

```json
{"plan_id": "85bdba4693c8", "approve": true}
```

Without `approve: true` the request is answered with `403 approval_required`
and nothing is fetched. The plan is also short-lived: it expires after
fifteen minutes.

### What happens during the transfer

- The file is written as `<name>.part` and the transfer continues from there
  with an HTTP range request if the download is interrupted.
- The size must match, and the content must match the hash the Hub
  published. A file that fails is **deleted**, not installed.
- Only `huggingface.co` and its own CDN (`hf.co`) are ever contacted; every
  redirect is checked, and anything else is refused.
- A downloaded file is **never** loaded or executed. It joins the library
  like any other model and still has to be loaded explicitly.

---

## Ports

| Port | Who |
|---|---|
| `1234` | The gateway — the only port you talk to |
| `18400–18499` | Internal engine instances, one per loaded model |

Nothing outside `127.0.0.1` is exposed unless you set `allow_remote`, and even
then the bind address is explicit.
