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

## Ports

| Port | Who |
|---|---|
| `1234` | The gateway — the only port you talk to |
| `18400–18499` | Internal engine instances, one per loaded model |

Nothing outside `127.0.0.1` is exposed unless you set `allow_remote`, and even
then the bind address is explicit.
