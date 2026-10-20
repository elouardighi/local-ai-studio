"""Recursive model discovery over configured directories.

Read-only by policy: files are opened for reading only, nothing under a model
directory is ever created, renamed or deleted. Each discovered model is turned
into a record with verified metadata (read from the GGUF header itself) and an
honest compatibility verdict.

Grouping rules
--------------
* A directory containing exactly one non-``mmproj`` GGUF is one model.
* A ``mmproj-*.gguf`` file in the same directory is attached to it as its
  vision projector (llama.cpp ``--mmproj``).
* Several non-mmproj GGUFs in one directory are separate models (e.g. several
  quantizations of the same base) unless they are split shards (``-00001-of-``).
* Hugging Face style directories (config.json + safetensors) are detected and
  reported as convertible/unsupported - never converted silently.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from . import gguf
from . import integrity
from .supported_archs import ENGINE_TAG, classify_arch

# Directories never worth walking into.
SKIP_DIRS = {".git", ".cache", "__pycache__", "node_modules", "$recycle.bin"}


def model_id(path: Path) -> str:
    """Stable identifier for a model file (path-derived, survives restarts)."""
    return hashlib.sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:16]


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024.0
    return f"{n:.1f} TB"


def _iter_files(root: Path):
    """Walk a directory tree, skipping junk and inaccessible entries."""
    for dirpath, dirnames, filenames in os.walk(root, onerror=lambda e: None):
        dirnames[:] = [d for d in dirnames if d.lower() not in SKIP_DIRS]
        for name in filenames:
            yield Path(dirpath) / name


def _is_mmproj(name: str) -> bool:
    lower = name.lower()
    return "mmproj" in lower or lower.startswith("mmproj-")


def _is_split_shard(name: str) -> bool:
    return "-of-" in name.lower() and "-0000" in name.lower()


def scan_directory(root: Path, check_integrity: bool = True
                   ) -> list[dict[str, Any]]:
    """Scan one root directory and return model records."""
    records: list[dict[str, Any]] = []
    if not root.is_dir():
        return [{
            "error": f"directory not accessible: {root}",
            "path": str(root), "models": [],
        }]

    gguf_by_dir: dict[Path, list[Path]] = {}
    hf_dirs: dict[Path, dict[str, Any]] = {}

    for file in _iter_files(root):
        try:
            lower = file.name.lower()
        except OSError:
            continue
        if lower.endswith(".gguf"):
            gguf_by_dir.setdefault(file.parent, []).append(file)
        elif lower in ("config.json", "model.safetensors.index.json"):
            try:
                if lower == "config.json":
                    cfg = json.loads(file.read_text(encoding="utf-8"))
                else:
                    cfg = {}
            except (OSError, json.JSONDecodeError):
                cfg = {}
            hf_dirs[file.parent] = {
                "config": cfg,
                "has_index": (file.parent / "model.safetensors.index.json").exists(),
            }

    # --- GGUF models -----------------------------------------------------
    for directory, files in sorted(gguf_by_dir.items()):
        mains = [f for f in files if not _is_mmproj(f.name)]
        projectors = [f for f in files if _is_mmproj(f.name)]

        # Group split shards into a single logical model.
        # Names look like Model-00001-of-00003.gguf: strip the shard part.
        shards: dict[str, list[Path]] = {}
        standalone: list[Path] = []
        for f in sorted(mains):
            if _is_split_shard(f.name):
                base = re.sub(r"-\d{5}-of-\d{5}", "", f.name).lower()
                shards.setdefault(base, []).append(f)
            else:
                standalone.append(f)

        for base, parts in shards.items():
            parts = sorted(parts)
            # The GGUF header lives in the first shard; the rest add size.
            records.append(_build_gguf_record(
                parts[0], directory, projectors, extra_files=parts[1:],
                check_integrity=check_integrity))

        for main_file in standalone:
            records.append(_build_gguf_record(
                main_file, directory, projectors,
                check_integrity=check_integrity))

    # --- Hugging Face / Safetensors directories ---------------------------
    for directory, info in sorted(hf_dirs.items()):
        # Skip directories that also contain GGUF (already handled).
        if directory in gguf_by_dir:
            continue
        weights = list(directory.glob("*.safetensors"))
        total = sum(_safe_size(w) for w in weights)
        cfg = info.get("config", {}) or {}
        records.append({
            "id": model_id(directory),
            "name": cfg.get("_name_or_path") or directory.name,
            "path": str(directory),
            "format": "safetensors",
            "size_bytes": total,
            "size_human": _human_size(total),
            "architecture": cfg.get("model_type", "unknown"),
            "quantization": None,
            "is_mmproj": False,
            "capabilities": _hf_capabilities(cfg),
            "compatibility": {
                "status": "convertible",
                "engine": "llama.cpp",
                "reason": (
                    "Safetensors/Transformers layout detected. llama.cpp needs "
                    "a GGUF conversion (llama.cpp convert_hf_to_gguf.py). "
                    "Local AI Studio never converts in place - converted files "
                    "go to the configured converted-models directory."
                ),
                "supported_by_engine": False,
            },
            "memory_estimate_mb": None,
            "integrity": {
                "status": integrity.STATUS_UNVERIFIED,
                "reason": ("safetensors weights are not GGUF; completeness "
                           "can only be checked after a conversion"),
                "size_bytes": total,
                "expected_min_bytes": None,
                "expected_bytes": None,
                "signature": None,
                "part_files": [],
            },
            "files": [{"path": str(w), "size_bytes": _safe_size(w)} for w in weights],
            "has_chat_template": "<|im_start|>" in str(cfg) or "chat_template" in str(cfg),
        })

    return records


def _safe_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def _hf_capabilities(cfg: dict[str, Any]) -> dict[str, Any]:
    arch = str(cfg.get("model_type", "")).lower()
    return {
        "vision": any(k in cfg for k in ("vision_config", "visual_config"))
                   or "vl" in arch,
        "moe": bool(cfg.get("expert_config")) or "moe" in arch,
        "embedding": "embedding" in arch or arch in ("bert", "nomic-bert"),
    }


def _build_gguf_record(main_file: Path, directory: Path,
                       projectors: list[Path],
                       extra_files: list[Path] | None = None,
                       check_integrity: bool = True) -> dict[str, Any]:
    """Build a model record from a main GGUF file and its siblings."""
    probe = gguf.read_gguf_metadata(main_file)
    summary = gguf.summarize(probe) if "metadata" in probe else {}
    arch = summary.get("architecture") or ""
    facts = classify_arch(arch)
    size = probe.get("size_bytes", 0) + sum(_safe_size(f) for f in (extra_files or []))

    quant = gguf.quant_label(summary, main_file.name)

    # Attach a matching vision projector (same directory, similar stem).
    mmproj_path = None
    stem_core = main_file.stem.lower()
    for proj in projectors:
        proj_stem = proj.stem.lower().replace("mmproj-", "", 1)
        # fuzzy: share the first two tokens or >=60% of the core name
        if (proj_stem[:24] in stem_core or stem_core[:24] in proj_stem
                or _token_overlap(stem_core, proj_stem) >= 0.6):
            mmproj_path = str(proj)
            break
    if mmproj_path is None and len(projectors) == 1:
        mmproj_path = str(projectors[0])  # only candidate in the directory

    if probe.get("error"):
        compat_status = "error"
        reason = f"GGUF metadata unreadable: {probe['error']}"
    elif not arch:
        compat_status = "unverified"
        reason = "No general.architecture key found in GGUF metadata."
    elif facts["supported_by_engine"]:
        compat_status = "supported"
        reason = (f"Architecture '{arch}' is implemented by the installed "
                  f"llama.cpp build.")
    else:
        compat_status = "unverified"
        reason = (f"Architecture '{arch}' is not in the installed engine's "
                  f"known architecture list (tag {ENGINE_TAG}). "
                  f"Loading will be attempted; the engine has the final say.")

    capabilities = {
        "vision": bool(mmproj_path) or facts["is_vision"],
        "moe": facts["is_moe"],
        "embedding": facts["is_embedding"],
        "chat_template": summary.get("has_chat_template", False),
    }

    files = [{"path": str(main_file), "size_bytes": _safe_size(main_file)}]
    for extra in extra_files or []:
        files.append({"path": str(extra), "size_bytes": _safe_size(extra)})
    if mmproj_path:
        files.append({"path": mmproj_path,
                      "size_bytes": _safe_size(Path(mmproj_path))})

    # Is the file actually complete, and is it a copy of something we already
    # have? Both questions are answered from the file, never guessed.
    if check_integrity:
        report = integrity.check_integrity(main_file, probe)
    else:
        report = integrity.IntegrityReport(
            status=integrity.STATUS_UNVERIFIED,
            reason="integrity checking is turned off in Settings")

    return {
        "id": model_id(main_file),
        "name": summary.get("name") or main_file.stem,
        "path": str(main_file),
        "directory": str(directory),
        "format": "GGUF",
        "gguf_version": probe.get("gguf_version"),
        "tensor_count": probe.get("tensor_count"),
        "size_bytes": size,
        "size_human": _human_size(size),
        "architecture": arch or "unknown",
        "quantization": quant,
        "is_mmproj": False,
        "mmproj_path": mmproj_path,
        "parameters_label": summary.get("size_label"),
        "context_length": summary.get("context_length"),
        "block_count": summary.get("block_count"),
        "embedding_length": summary.get("embedding_length"),
        "head_count": summary.get("head_count"),
        "head_count_kv": summary.get("head_count_kv"),
        "expert_count": summary.get("expert_count"),
        "expert_used_count": summary.get("expert_used_count"),
        "full_attention_interval": summary.get("full_attention_interval"),
        "tokenizer": summary.get("tokenizer_pre") or summary.get("tokenizer_model"),
        "license": summary.get("license"),
        "capabilities": capabilities,
        "compatibility": {
            "status": compat_status,
            "engine": "llama.cpp",
            "backend": "vulkan",
            "reason": reason,
            "supported_by_engine": facts["supported_by_engine"],
        },
        "integrity": report.to_dict(),
        "memory_estimate_mb": None,  # filled in by the memory planner
        "files": files,
    }


def _token_overlap(a: str, b: str) -> float:
    ta = set(a.split("-"))
    tb = set(b.split("-"))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


class ModelScanner:
    """Scans configured roots and caches results until ``rescan()``."""

    def __init__(self, get_dirs, log=None, check_integrity=None) -> None:
        self._get_dirs = get_dirs
        self._log = log
        # A callable returning True/False, so the setting can be changed
        # without rebuilding the scanner. Integrity checking reads each
        # file's tensor directory, which is worth being able to switch off.
        self._check_integrity = check_integrity or (lambda: True)
        self._models: dict[str, dict[str, Any]] = {}
        self._last_scan: float | None = None
        self._errors: list[str] = []
        # The folder list the cache was built from. When it changes the
        # cache is stale and must not be trusted.
        self._signature: tuple[str, ...] = ()

    def _integrity_enabled(self) -> bool:
        try:
            return bool(self._check_integrity())
        except Exception:  # noqa: BLE001 - never break a scan over a setting
            return True

    def _ensure_current(self) -> None:
        """Rescan when the watched folders have changed since the last scan.

        The cache used to survive a change to the folder list, so a model
        could still be listed after its folder was removed, and then vanish
        at the next scan - leaving a page holding an id the server no longer
        knows. The cache now follows the folder list.
        """
        try:
            dirs = self._get_dirs()
        except Exception:  # noqa: BLE001 - never break a lookup over a setting
            return
        signature = tuple(str(Path(d)) for d in dirs)
        if signature != self._signature:
            self.scan(dirs)

    def scan(self, dirs: list[str] | None = None) -> list[dict[str, Any]]:
        import time
        started = time.time()
        dirs = list(self._get_dirs()) if dirs is None else list(dirs)
        found: dict[str, dict[str, Any]] = {}
        errors: list[str] = []
        check = self._integrity_enabled()
        for d in dirs:
            root = Path(os.path.expandvars(d)).expanduser()
            for result in scan_directory(root, check_integrity=check):
                if "error" in result:
                    errors.append(result["error"])
                    continue
                found[result["id"]] = result
        self._models = found
        self._signature = tuple(str(Path(d)) for d in dirs)
        self._last_scan = time.time()
        self._errors = errors
        if self._log:
            self._log.info(
                "scanner",
                f"scan complete: {len(found)} models in "
                f"{time.time() - started:.1f}s from {len(dirs)} directories",
            )
            for err in errors:
                self._log.warning("scanner", err)
        return self.list()

    def list(self) -> list[dict[str, Any]]:
        self._ensure_current()
        return sorted(self._models.values(),
                      key=lambda m: m.get("size_bytes", 0), reverse=True)

    def get(self, model_id: str) -> dict[str, Any] | None:
        self._ensure_current()
        return self._models.get(model_id)

    def find_by_alias_or_name(self, name: str) -> dict[str, Any] | None:
        """Resolve a caller-supplied model name to a discovered model.

        Accepts the display name, the GGUF file stem, or the internal id
        (the hash used by /api/models), case-insensitively.
        """
        name_l = name.lower()
        for m in self._models.values():
            if name_l in (m["name"].lower(), Path(m["path"]).stem.lower(),
                          m["id"].lower()):
                return m
        for m in self._models.values():
            if name_l in m["name"].lower():
                return m
        return None

    @property
    def last_scan(self) -> float | None:
        return self._last_scan

    @property
    def errors(self) -> list[str]:
        return list(self._errors)
