# Troubleshooting

The fastest first move is always the same:

```powershell
.\scripts\diagnose.ps1
```

It checks Python, the virtual environment, the engine binary, Vulkan, GPU
drivers, free VRAM/RAM, listening ports and the backend's health endpoint, then
writes a report to the console and to `logs\diagnose-*.log`. Include that file
when asking for help.

---

## Setup fails

### "Python not found" / "Python X is too old"

Setup needs Python **3.10+** on `PATH` (3.12 is what this was built and tested
on). Install from [python.org](https://www.python.org/downloads/) and make sure
"Add to PATH" is checked, then open a **new** terminal.

### Download of the engine fails

Setup fetches an official llama.cpp Vulkan build from GitHub
(`llama-<tag>-bin-win-vulkan-x64.zip`). If you're offline, behind a proxy, or
want a different build:

```powershell
.\scripts\setup.ps1 -EngineTag b11712   # pin a known-good nightly
.\scripts\setup.ps1 -SkipEngine         # reuse an engine already in engine\
```

To build from source instead (needs `git` and `cmake`):

```powershell
.\scripts\setup.ps1 -BuildFromSource
```

---

## No Vulkan device

`setup.ps1` runs `vulkaninfo --summary` and looks for a device line.

**"vulkaninfo not on PATH"** — the tool is missing, but a Vulkan *runtime* may
still be present. This is a warning, not necessarily a failure. The real test is
whether the engine reports a device:

```powershell
& ".\engine\llama.cpp\llama-server.exe" --list-devices
```

You should see a line containing `Vulkan` and your GPU name.

**"vulkaninfo produced no device summary" / no Vulkan device from the engine**

1. **Update the GPU driver.** This is the cause most of the time. Use the
   vendor's full installer (AMD Software: Adrenalin Edition), not just the
   Windows Update driver.
2. Make sure you downloaded the **Vulkan** build, not a CUDA or CPU one. Check
   `engine\llama.cpp\` contains `llama-server.exe` from a `-vulkan-` zip.
3. Confirm the GPU actually supports Vulkan — RDNA2/RDNA3 AMD and RTX-series
   NVIDIA do.
4. Re-run `.\scripts\setup.ps1` after fixing the driver.

**Works on CPU instead:** set `engine_backend` to `"cpu"` in Settings (or edit
`data\settings.json` and restart). Slow, but functional.

---

## Models won't load

### "concurrent instance limit reached"

`max_instances` is `1` by default and something is already loaded. Unload it
first, or raise `max_instances` in Settings.

### "memory plan risky … exceeds currently available RAM"

The planner refuses to start a model it expects to page to disk. This is
deliberate — it stops your machine from grinding to a halt.

- Lower `context` (the single biggest lever; see
  [LIMITATIONS.md](LIMITATIONS.md#context-length-is-bounded-by-vram-not-by-the-model)).
- Close other GPU/RAM users.
- Use a smaller or more aggressively quantized model.
- Override with `"force": true` in the load request if you accept the risk.

### "model already loaded"

The same model can't be loaded twice unless you pass `allow_duplicate: true`.

### Load hangs at `starting_engine` or `loading_model`

Almost always a driver or VRAM problem. Check the per-instance logs:

```powershell
Invoke-RestMethod "http://127.0.0.1:1234/api/instances/<instance_id>/logs"
```

and the **Logs** page in the dashboard, filtered to that instance.

---

## The API isn't responding

### `start.ps1` returns but nothing answers

Give it a couple of seconds, then:

```powershell
Invoke-RestMethod http://127.0.0.1:1234/health
```

If that fails, the backend log has the reason:

```powershell
Get-Content .\logs\backend.log -Tail 50
```

### Port 1234 is already in use

Something else holds the port (another server, or a previous instance that
didn't shut down cleanly).

```powershell
Stop-Process -Id (Get-NetTCPConnection -LocalPort 1234 -State Listen).OwningProcess -Force
.\scripts\start.ps1
```

`stop.ps1` also cleans up orphaned engines and stale PID files — try it first.

### "connection refused" from a client

The gateway binds `127.0.0.1` only. `localhost` usually works, but some clients
resolve `localhost` to IPv6 `::1` first, which we don't bind. Use
`http://127.0.0.1:1234/v1` explicitly.

---

## The dashboard is stale or won't load

Assets are served with `Cache-Control: no-cache`, so a hard refresh
(`Ctrl+F5`) always gets current code. If the **Live** badge shows
"reconnecting…", the backend restarted — the page reconnects on its own.

---

## Engine loads but output is garbled

Usually, in order of likelihood:

1. **Wrong chat template** for that architecture — try a different model.
2. **Quant too aggressive** — a Q2/IQ2 quant of a large model can be incoherent
   where Q4 is fine.
3. **Context too small** — the prompt got truncated, so the model lost the
   beginning of the conversation. Raise `context` if VRAM allows.
4. **Truncated output** — raise `max_tokens`. Reasoning models especially need
   room to think *and* answer; see
   [MODEL_COMPAT.md](MODEL_COMPAT.md#reasoning-models).

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
node --test frontend\test\*.test.mjs
```

The pytest suite runs against a **fake engine**, not a real model, so it never
touches your GPU. If a test fails after a change, the failure message includes
the response body.

The Node tests need `node` on `PATH`; they're skipped cleanly if it's absent.

---

## Getting help

Collect these first:

```powershell
.\scripts\diagnose.ps1                      # report → logs\diagnose-*.log
Get-Content .\logs\backend.log -Tail 200
.\.venv\Scripts\python.exe -m pytest tests -q 2>&1 | Tee-Object logs\tests.log
```

The diagnostic report includes your driver version, engine version, free
VRAM/RAM and the health payload — enough to reproduce most issues without
needing your models.
