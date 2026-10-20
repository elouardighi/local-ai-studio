# Changelog

All notable changes to Local AI Studio are recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and the project uses [semantic versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-10-20

The model library. The project can now find, inspect, check, fetch and
remove models instead of only listing what happens to be on disk.

### Added

- **Model library** — a searchable, filterable view of the local models.
  Search by name, path, architecture or quantization; filter by
  architecture, quantization, format, size range, file state and
  compatibility; sort by size, name, architecture or quantization. The
  filter controls are filled from the data that actually exists
  (`facets`), not hard-coded.
- **File integrity checking** — every model file's tensor directory is
  read to work out the size the file *should* have. A file whose tensor
  data stops short is reported as **incomplete**, with the number of bytes
  missing. A file that is not a model at all is reported as **corrupt**.
  A download interrupted half way leaves a `.part` file behind, and that
  is spotted too.
- **Duplicate and variant detection** — files are fingerprinted from their
  size and their first and last mebibyte, so the same weights in two
  folders are recognised without reading ten gigabytes twice. A model in
  another quantization is reported as a *variant* rather than a duplicate,
  because neither copy should be thrown away.
- **Import folders** — a folder picker for adding model directories to the
  library. Nothing inside an imported folder is moved, renamed or deleted.
- **Preflight checks** — before a model is loaded, seven checks run: the
  file is there, it is complete, the architecture is implemented by the
  installed engine, the quantization is usable, the vision projector is
  attached, the context fits the model, and the memory plan survives. The
  result is `ready`, `risky` or `blocked`, and the load dialog shows it.
- **Optional Hugging Face downloader** — off by default. Browsing the Hub
  is free; fetching a file needs the downloader switched on *and* an
  individual approval of that file. Transfers resume from a `.part` file
  with an HTTP range request, are checked against the free disk space, and
  are verified against the hash the Hub published. A file that does not
  match is discarded rather than installed.
- **Deleting models** — with a plan that shows the exact files and the
  space that would be freed. The file goes to the Windows Recycle Bin, so
  a mistake can be undone. A loaded model has to be unloaded first, and a
  vision projector is only removed when it is asked for.
- **OpenCode integration** — a downloaded model is added to the OpenCode
  config automatically, and a deleted model is taken out again. The limits
  written are honest ones: the context that fits in the free VRAM, not the
  number the model advertises. The config is edited as text, so comments,
  formatting and other providers are left alone.
- **Continuous integration** — the test suite runs on every push and pull
  request.

### Fixed

- The scanner cache no longer goes stale when the folder list changes.
  A model could be shown in the dashboard and then be unknown to the
  server, which made loading it fail with "model not found".
- A lookup that comes up empty now triggers one more scan, so a model
  that was just downloaded is found rather than reported as missing.
- The context requested for a model is capped at the model's own maximum.
  llama.cpp clamps the context but allocates the KV cache for the size it
  was *asked* for, so a 2,048-token model was reserving 100,000 tokens of
  VRAM - 8.4 GB of it wasted.
- The context is settled before the number of GPU layers is chosen. A
  large context left no room for the weights, which collapsed the offload
  to zero layers and put the whole model in system RAM.
- The memory planner judges the VRAM, not only the RAM. It reports what
  the GPU needs; it now also says when that does not fit.
- A context that does not fit is reduced to the largest that does,
  instead of the load being refused for a context the user did not
  think about.
- "Try increasing it" is no longer the advice given for a model whose
  window cannot be increased. The gateway now says the context cannot be
  raised, and that the model is too small for a long conversation.
- The `requirements.txt` now declares the test dependencies, which
  `setup.ps1` needs to verify an install.
- The dashboard's **Load** button no longer closes its own dialog before
  the load starts, which hid the progress readout.
- The model list is no longer stale after a model is added or removed.

### Changed

- The Models page is now **Library**.
- The instance record always shows the context the engine is really
  using.
- `GET /v1/models` reports the recommended context, the output budget and
  the model's own ceiling.

[0.2.0]: https://github.com/elouardighi/local-ai-studio/releases/tag/v0.2.0
[0.1.0]: https://github.com/elouardighi/local-ai-studio/releases/tag/v0.1.0
