# Local AI Studio

A local, LM Studio-style model studio for Windows: discover the GGUF models you
already have on disk, load them onto an AMD GPU through llama.cpp's Vulkan
backend, and talk to them over an OpenAI-compatible API on `127.0.0.1:1234`.

Everything runs on your machine. No Docker, no cloud, no telemetry.

```
┌─────────────────────────────────────────────────────────────┐
│  Browser dashboard  (http://127.0.0.1:1234)                 │
│  Dashboard · Library · Instances · Chat · Playground ·       │
│  Monitor · Logs · Settings                                  │
└───────────────┬─────────────────────────────────────────────┘
                │  /api/*  (management)   /v1/*  (OpenAI-compatible)
┌───────────────▼─────────────────────────────────────────────┐
│  FastAPI gateway  (127.0.0.1:1234)                          │
│  model library · instance manager · memory planner ·      │
│  resource monitor · event bus                               │
└───────────────┬─────────────────────────────────────────────┘
                │  internal ports 18400-18499
┌───────────────▼─────────────────────────────────────────────┐
│  llama-server.exe  (llama.cpp, Vulkan backend)              │
│  one process per loaded model, offloaded to the GPU         │
└─────────────────────────────────────────────────────────────┘
```

---

## Requirements

| | |
|---|---|
| OS | Windows 10/11 x64 |
| GPU | AMD RDNA3 (verified on RX 7800 XT, 16 GB) |
| Python | 3.12+ |
| Disk | ~200 MB for the engine, plus your models |
| RAM | Enough to hold whichever model you load |

Models are read from `C:\Users\SAKISAKI\.lmstudio\models` by default. The
scanner **never writes to, renames, or deletes** anything in your model
directory.

---

## Install

```powershell
cd C:\Users\SAKISAKI\Documents\gay\local-ai-studio
.\scripts\setup.ps1
```

`setup.ps1` checks the OS, Python, GPU and Vulkan support; downloads the
official prebuilt llama.cpp Vulkan build (currently `b11712`, llama.cpp
0.7.0-dev); creates an isolated `.venv`; installs dependencies; writes a
default `data\settings.json` **without overwriting an existing one**; and
verifies the result.

Options:

```powershell
.\scripts\setup.ps1 -EngineTag b11712     # pin a specific nightly build
.\scripts\setup.ps1 -BuildFromSource      # compile llama.cpp with GGML_VULKAN=ON
.\scripts\setup.ps1 -SkipEngine           # reuse an engine already in engine\
```

If setup can't find a Vulkan device, see
[docs/TROUBLESHOOTING.md](local-ai-studio/docs/TROUBLESHOOTING.md).

---

## Run

```powershell
.\scripts\start.ps1      # start the backend (detached) on 127.0.0.1:1234
.\scripts\stop.ps1       # stop the backend and every engine it started
.\scripts\diagnose.ps1   # write a diagnostic report to the console and logs\
```

Then open **http://127.0.0.1:1234** in a browser.

`start.ps1` returns in a couple of seconds and leaves the server running in the
background. `stop.ps1` verifies process identity before killing anything, so it
stops the launcher, the API server and every `llama-server.exe` it started — and
nothing else.

---

## Use

### From the dashboard

1. Open **http://127.0.0.1:1234**.
2. Go to **Model library**. Your GGUF files appear with size, quantization,
   architecture and a compatibility verdict. Search them by name, size,
   quantization or architecture, and filter by whether the file is complete.
3. Pick one and press **Load**. The memory planner decides how many layers fit
   in VRAM before anything starts, so a model that can't fit is refused with a
   reason instead of hanging or thrashing your pagefile. The load dialog
   also runs a full preflight check — file completeness, architecture,
   quantization, context and the memory plan — so you see the problems
   before the engine does.
4. Talk to it from **Chat**, poke the raw API from **Playground**, watch VRAM
   and RAM on **Monitor**, and read engine output on **Logs**.
5. **Unload** frees the VRAM immediately.

### Finding and importing models

The **Model library** page searches the folders you have registered. A model
that exists in two places is flagged as a duplicate (the same bytes, so one
copy can go); the same model in a different quantization is flagged as a
variant (neither copy should go). A file whose tensor data stops short is
reported as incomplete, with the number of bytes missing.

**Import folder** adds a folder to the library. There is a folder picker, or
you can type a path. Nothing inside an imported folder is moved, renamed or
deleted — the library only ever reads.

### Deleting a model

Every model card has a **Delete** button. It shows you exactly which files
would be removed and how much space that frees, and a vision projector is
only deleted when you tick the box for it.

Deleted files go to the **Windows Recycle Bin**, so a mistake is one click
to undo. (Windows erases anything too big for the bin for good — when that
would happen, the dialog says so *before* you press Delete.) A loaded model
has to be unloaded first.

This is the only operation in the project that writes to a model directory,
and it only ever touches the file of a model the library already knows
about.

### Downloading a model (optional, off by default)

**Download** opens the Hugging Face Hub. You can search it and look at
what's in any repository straight away - browsing is read-only, and so is
planning a download, so you always see the size, the destination and the
hash before deciding anything. Fetching a file is a separate thing: it
stays switched off until you press **Allow downloads** in the dialog (or
turn on *Allow downloads from Hugging Face* in Settings), and even then
nothing is fetched until you press **Download** to approve that specific
file.

Transfers resume if they are interrupted, are verified against the hash the
Hub published, and a file that does not match is discarded rather than
installed. A downloaded model is a file like any other: it is never loaded
or run without you asking for it.

### From any OpenAI-compatible client

Point the client at `http://127.0.0.1:1234/v1`. No API key is required unless
you set one in **Settings**.

```powershell
# List what's on disk
curl http://127.0.0.1:1234/v1/models

# Chat (auto-loads the model if it isn't loaded yet)
curl http://127.0.0.1:1234/v1/chat/completions `
  -H "Content-Type: application/json" `
  -d '{\"model\":\"Bonsai-27B\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello\"}]}'
```

See **[docs/API.md](local-ai-studio/docs/API.md)** for the full API with curl, PowerShell and
Python examples, including streaming.

### From OpenCode

OpenCode does **not** ask an OpenAI-compatible server which models it has.
For a custom provider it reads the model list straight out of its own
config file. Local AI Studio keeps that file in step for you:

- **a model you download is added to it** — automatically, as soon as the
  download finishes;
- **a model you delete is taken out of it** — so you are never offered a
  model that is not there.

The limits it writes are **honest ones**. A model advertises a context
length — 150k, 262k — that says something about the architecture, not about
the card it is going on. The server is asked what actually fits in the free
VRAM and that is written instead, so OpenCode does not build a prompt the
engine will only reject:

```jsonc
"py007_tinyllama-1.1b-chat-v0.3": {
  "name": "py007_tinyllama-1.1b-chat-v0.3",
  "limit": { "context": 2048, "output": 1024 }
}
```

The config is edited as text, so your comments, your formatting and every
other provider are left exactly as they were, and a backup is written each
time. The **OpenCode** bar on the Library page shows whether the two are in
step, and *Sync now* puts them back together. It can be switched off in
Settings, and the file it edits can be pointed somewhere else.

You can also do it by hand:

```powershell
.\scripts\sync-opencode.ps1            # add whatever is missing
.\scripts\sync-opencode.ps1 -DryRun    # show what would change
.\scripts\sync-opencode.ps1 -Force     # also rewrite the context limits
```

If you prefer to write the provider by hand, this is all it takes:

```jsonc
{
  "provider": {
    "localai": {
      "name": "Local AI Studio",
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:1234/v1" },
      "models": {
        "Bonsai-27B": { "name": "Bonsai 27B", "limit": { "context": 43008, "output": 8192 } }
      }
    }
  }
}
```

With **Auto-load** enabled (Settings → *Auto-load a model when a request names
it*), the first message to an unloaded model waits while it loads, then answers.
Switching models evicts the idle one automatically.

---

## Model compatibility

Local AI Studio runs standard **GGUF** models through llama.cpp. See
[docs/MODEL_COMPAT.md](local-ai-studio/docs/MODEL_COMPAT.md) for what works, what needs a
companion `mmproj` file, and how quantization affects memory.

---

## Project layout

```
local-ai-studio\
├─ backend\app\
│  ├─ api\           FastAPI routers: /api/* management, /api/library/*, /v1/* OpenAI-compatible
│  ├─ services\      scanner, library, instance manager, memory planner, monitor, event bus
│  ├─ config.py      settings model + persistence
│  └─ main.py        app factory, lifespan, middleware
├─ frontend\src\     vanilla-JS dashboard (no build step)
├─ engine\llama.cpp\ llama-server.exe (downloaded by setup.ps1)
├─ scripts\          setup.ps1 · start.ps1 · stop.ps1 · diagnose.ps1 · build-vulkan.ps1
├─ tests\            pytest suite (backend) + Node unit tests (frontend)
└─ data\             settings.json · instances.json · library.json (created at runtime)
```

---

## Tests

```powershell
# Backend + frontend-JS suites
.\.venv\Scripts\python.exe -m pytest tests -q

# Frontend JS unit tests directly
node --test frontend\test\*.test.mjs
```

The suite covers the memory planner, the model scanner, the instance lifecycle
state machine, the resource monitor, the OpenAI-compatible gateway end-to-end
(against a fake engine, never a real model), the event bus, the log store and
redaction, and the Logs-page normalizer.

---

## Known limitations

Read **[docs/LIMITATIONS.md](local-ai-studio/docs/LIMITATIONS.md)** before relying on this for
anything important. The short version:

- Context length is limited by VRAM, not by what the model advertises. A 27B
  model on a 16 GB card gets ~8k tokens, not the 150k in its metadata.
- One model at a time by default (`max_instances: 1`).
- CPU-only fallback exists but is slow.
- The memory planner's estimates are conservative heuristics, not guarantees.

---

## License

Use and modify freely for personal, local use.
