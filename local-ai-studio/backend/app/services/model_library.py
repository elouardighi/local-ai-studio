"""The model library: what is on disk, what is usable, and what is duplicated.

One service owns the whole view of the local model collection:

* **which folders are scanned** - the library is the only place folders are
  added or removed, and every change is written to the settings file so the
  scanner, the dashboard and the settings page never disagree;
* **search** - by name, architecture, quantization, size and health;
* **duplicates and variants** - the same weights in two places, or the same
  base model in two quantizations;
* **preflight** - the checks that run *before* a model is loaded, so an
  incompatible architecture or a truncated file is reported instead of
  discovered by a failed load.

Read-only with respect to model files: nothing here writes, renames or
deletes anything inside a model directory.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from . import file_removal, integrity, memory_planner, opencode_sync
from .integrity import family_key
from .model_scanner import ModelScanner
from .supported_archs import ENGINE_TAG, ENGINE_VERSION, classify_arch

__all__ = ["ModelLibrary", "family_key", "parse_size", "quant_compatibility",
           "QUANT_COMPATIBILITY"]

# ---------------------------------------------------------------------------
# Quantization compatibility
# ---------------------------------------------------------------------------

# Quantization types the pinned engine build implements, and what the user
# should know about each. "status" is about *usability*, not about whether
# llama.cpp can read the type: a type that is not listed is reported as
# unverified, and the engine still has the final say.
QUANT_COMPATIBILITY: dict[str, dict[str, str]] = {
    "F32": {"status": "ok", "note": "unquantized - largest file, best quality"},
    "F16": {"status": "ok", "note": "unquantized - twice the size of Q8_0"},
    "BF16": {"status": "ok", "note": "unquantized - for training-grade weights"},
    "Q8_0": {"status": "ok", "note": "near-lossless"},
    "Q6_K": {"status": "ok", "note": "very good quality"},
    "Q5_K_M": {"status": "ok", "note": "very good quality"},
    "Q5_K_S": {"status": "ok", "note": "very good quality"},
    "Q4_K_M": {"status": "ok", "note": "good - the common default"},
    "Q4_K_S": {"status": "ok", "note": "good"},
    "Q3_K_M": {"status": "warn", "note": "usable, noticeable quality loss"},
    "Q3_K_S": {"status": "warn", "note": "visible quality loss"},
    "Q2_K": {"status": "warn", "note": "aggressive - expect clear degradation"},
    "IQ4_XS": {"status": "ok", "note": "good quality at 4 bits"},
    "IQ4_NL": {"status": "ok", "note": "good quality at 4 bits"},
    "IQ3_S": {"status": "warn", "note": "aggressive"},
    "IQ3_M": {"status": "warn", "note": "aggressive"},
    "IQ3_XS": {"status": "warn", "note": "aggressive"},
    "IQ3_XXS": {"status": "warn", "note": "very aggressive"},
    "IQ2_M": {"status": "warn", "note": "very aggressive - large quality loss"},
    "IQ2_S": {"status": "warn", "note": "very aggressive - large quality loss"},
    "IQ2_XS": {"status": "warn", "note": "extremely aggressive"},
    "IQ2_XXS": {"status": "warn", "note": "extremely aggressive"},
    "IQ1_S": {"status": "warn", "note": "near-lossless loss - expect incoherence"},
    "IQ1_M": {"status": "warn", "note": "extremely aggressive"},
    "Q1_0": {"status": "warn", "note": "very new type - poorly tested, expect "
                                          "incoherent output on some models"},
    "Q2_0": {"status": "warn", "note": "very new type - poorly tested"},
    "MXFP4_MOE": {"status": "warn", "note": "FP4 format - needs FP4-capable "
                                            "hardware; the engine decides"},
    "NVFP4": {"status": "warn", "note": "FP4 format - needs FP4-capable "
                                        "hardware; the engine decides"},
    "TQ1_0": {"status": "warn", "note": "ternary - very low quality"},
    "TQ2_0": {"status": "warn", "note": "ternary - very low quality"},
}

# Quantization types the installed engine build does not implement. Empty
# today, but the table is where a future engine without a type would go.
QUANT_UNSUPPORTED: set[str] = set()

_SIZE_UNITS = {"b": 1, "k": 1024, "kb": 1024, "m": 1024 ** 2,
               "mb": 1024 ** 2, "g": 1024 ** 3, "gb": 1024 ** 3,
               "t": 1024 ** 4, "tb": 1024 ** 4}


def parse_size(text: str | None) -> int | None:
    """Parse ``"10gb"``, ``"500 MB"``, ``"1.5g"`` into bytes."""
    if not text:
        return None
    m = re.fullmatch(r"\s*([0-9]*\.?[0-9]+)\s*([a-z]*)\s*", str(text),
                      re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    unit = (m.group(2) or "b").lower()
    factor = _SIZE_UNITS.get(unit)
    if factor is None:
        return None
    return int(value * factor)


def _small_context_warning(context: int) -> str:
    """The warning shown for a model whose window is too small to work in."""
    return (f"only {context:,} tokens fit here. A long conversation will "
            f"not work: summarising it needs more room than the model "
            f"has. Use a model with a larger context for agent work.")


def _output_budget(context: int) -> int:
    """How many tokens a response may use, given the context.

    A quarter of the context, and never more than half of it - the other
    half is for the prompt. A model with a 128 token window cannot be given
    a 1024 token answer, nor one that leaves no room to ask anything.
    """
    context = max(1, int(context))
    budget = max(1, context // 4)
    budget = min(budget, max(1, context // 2))
    return int(min(8192, budget))


def quant_compatibility(quant: str | None) -> dict[str, Any]:
    """What we know about one quantization type."""
    if not quant or quant == "unknown":
        return {"status": "unverified",
                "note": "the quantization could not be determined"}
    if quant in QUANT_UNSUPPORTED:
        return {"status": "fail",
                "note": f"quantization {quant} is not implemented by this "
                        f"engine build"}
    entry = QUANT_COMPATIBILITY.get(quant)
    if entry is None:
        return {"status": "unverified",
                "note": f"quantization {quant} is not in the known list for "
                        f"llama.cpp {ENGINE_VERSION}; the engine has the final "
                        f"say"}
    return dict(entry)


# ---------------------------------------------------------------------------
# Folder browsing
# ---------------------------------------------------------------------------

# Directories that are never offered in the folder picker: they hold the
# operating system and installed programs, not models.
BLOCKED_FOLDER_NAMES = {"windows", "program files", "program files (x86)",
                        "programdata", "$recycle.bin", "system volume information",
                        "recovery", "perflogs"}


def _is_blocked(path: Path) -> bool:
    for part in path.parts:
        if part.lower() in BLOCKED_FOLDER_NAMES:
            return True
    return False


def _drives() -> list[dict[str, Any]]:
    """Available drive roots, for the picker when no path is given."""
    out = []
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:\\"
        if Path(root).is_dir():
            out.append({"name": root, "path": root, "type": "drive"})
    return out


# ---------------------------------------------------------------------------
# The library
# ---------------------------------------------------------------------------

class ModelLibrary:
    """Owns the folder list and the enriched view of the model collection."""

    def __init__(self, scanner: ModelScanner, log, events,
                 get_settings: Callable[[], Any],
                 update_settings: Callable[[dict[str, Any]], Any] | None = None,
                 state_file: Path | None = None,
                 get_sample: Callable[[], dict[str, Any]] | None = None,
                 get_instances: Callable[[], list[dict[str, Any]]] | None = None
                 ) -> None:
        self._scanner = scanner
        self._log = log
        self._events = events
        self._get_settings = get_settings
        self._update_settings = update_settings
        self._state_file = state_file
        # The latest resource sample, so the library can say how much VRAM
        # is really free right now.
        self._get_sample = get_sample or (lambda: {})
        # The instances that are currently loaded, so the VRAM they hold can
        # be counted as "would be free" for a different model.
        self._get_instances = get_instances or (lambda: [])
        self._imported: dict[str, dict[str, Any]] = {}
        # The models this library deleted. They are taken out of the
        # OpenCode config too, but only ever those - a model the user
        # listed themselves is never touched.
        self._deleted_models: list[str] = []
        self._duplicates: dict[str, list[str]] = {}
        self._variants: dict[str, list[str]] = {}
        self._load_state()

    # -- state ------------------------------------------------------------

    def _load_state(self) -> None:
        import json
        if not self._state_file:
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        imported = raw.get("imported_folders")
        if isinstance(imported, dict):
            self._imported = {k: v for k, v in imported.items()
                              if isinstance(v, dict)}
        deleted = raw.get("deleted_models")
        if isinstance(deleted, list):
            self._deleted_models = [str(n) for n in deleted][-200:]

    def _save_state(self) -> None:
        import json
        if not self._state_file:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"imported_folders": self._imported,
                       "deleted_models": self._deleted_models[-200:]}
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self._state_file)
        except OSError as exc:
            self._log.warning("library", f"could not persist library state: {exc}")

    def ensure_imported_folders(self) -> list[str]:
        """Make sure every imported folder is still being scanned.

        The folder list also lives in the settings, and the settings can be
        edited elsewhere - the Settings page, for one, which sends the whole
        form back and can carry a list that was loaded before a folder was
        imported. This puts the imported folders back so a download does not
        quietly drop out of the library.

        A folder that is no longer on disk is forgotten: it was removed by
        the user, and keeping it would only produce a scan error.
        """
        if not self._imported:
            return []
        current = [str(Path(p)) for p in self._get_settings().model_dirs]
        gone = [p for p in list(self._imported) if not Path(p).is_dir()]
        if gone:
            for path in gone:
                self._imported.pop(path, None)
            current = [p for p in current if p not in gone]
            self._save_state()
            self._log.info("library",
                           f"forgot {len(gone)} imported folder(s) that are "
                           f"no longer on disk")
        missing = [p for p in self._imported if p not in current]
        if not missing and not gone:
            return []
        if self._update_settings:
            self._update_settings({"model_dirs": current + missing})
        if missing:
            self._log.info("library",
                           f"restored {len(missing)} imported folder(s) to the "
                           f"scanned list")
        return missing

    # -- folders ----------------------------------------------------------

    def folders(self) -> list[dict[str, Any]]:
        """Every scanned folder, with its provenance."""
        out = []
        for path in self._get_settings().model_dirs:
            meta = self._imported.get(str(Path(path)), {})
            out.append({
                "path": str(path),
                "imported": bool(meta),
                "added_at": meta.get("added_at"),
                "exists": Path(path).is_dir(),
            })
        return out

    def add_folder(self, path: str) -> dict[str, Any]:
        """Import a folder into the library.

        The folder is added to the scanned set and the library is rescanned.
        Nothing inside the folder is moved, copied or modified.
        """
        raw = str(path or "").strip().strip('"')
        if not raw:
            raise ValueError("a folder path is required")
        resolved = Path(os.path.expandvars(raw)).expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:
            pass
        if not resolved.is_dir():
            raise ValueError(f"not a folder: {resolved}")
        if _is_blocked(resolved):
            raise ValueError(
                f"{resolved} is a system folder and is not scanned for models")

        settings = self._get_settings()
        current = [str(Path(p)) for p in settings.model_dirs]
        key = str(resolved)
        already = key in current
        if not already:
            updated = current + [key]
            if self._update_settings:
                self._update_settings({"model_dirs": updated})
        self._imported[key] = {"added_at": time.time(), "path": key}
        self._save_state()
        self._log.info("library", f"imported folder {key}")
        # Rescan even when the folder was already known: a download may
        # have just put a new model in it.
        self.rescan()
        # A new model is only useful if the client can see it. Keep the
        # OpenCode config in step.
        self.sync_opencode()
        return {"path": key, "already_present": already,
                "folders": self.folders()}

    def remove_folder(self, path: str) -> dict[str, Any]:
        """Stop scanning a folder. The folder and its files are untouched."""
        raw = str(path or "").strip().strip('"')
        try:
            key = str(Path(os.path.expandvars(raw)).expanduser().resolve())
        except OSError:
            key = str(Path(raw))
        settings = self._get_settings()
        current = [str(Path(p)) for p in settings.model_dirs]
        if key not in current:
            # Accept a non-resolved spelling too (a trailing slash, a
            # different case) so the UI cannot leave a stale entry behind.
            for candidate in current:
                try:
                    if str(Path(candidate).resolve()) == key:
                        key = candidate
                        break
                except OSError:
                    continue
        if key not in current:
            raise ValueError(f"folder is not in the library: {path}")
        remaining = [p for p in current if p != key]
        if self._update_settings:
            self._update_settings({"model_dirs": remaining})
        self._imported.pop(key, None)
        self._save_state()
        self._log.info("library", f"removed folder {key} from the library")
        self.rescan()
        return {"path": key, "folders": self.folders()}

    def browse(self, path: str | None = None) -> dict[str, Any]:
        """List the sub-folders of a path, for the folder picker.

        Only drive roots, the user's own profile and already-known folders
        are reachable; system directories are hidden rather than listed.
        """
        if not path:
            drives = _drives()
            return {"path": "", "parent": None, "entries": drives,
                    "roots": drives}
        try:
            resolved = Path(os.path.expandvars(path)).expanduser().resolve()
        except OSError as exc:
            raise ValueError(f"cannot open {path}: {exc}") from exc
        if not resolved.is_dir():
            raise ValueError(f"not a folder: {resolved}")
        if _is_blocked(resolved):
            raise ValueError(f"{resolved} is a system folder")

        entries: list[dict[str, Any]] = []
        try:
            for item in sorted(resolved.iterdir(), key=lambda p: p.name.lower()):
                if not item.is_dir():
                    continue
                if item.name.startswith(".") or _is_blocked(item):
                    continue
                entries.append({"name": item.name, "path": str(item),
                                "type": "folder"})
        except PermissionError:
            entries = []
        except OSError as exc:
            raise ValueError(f"cannot read {resolved}: {exc}") from exc

        parent = str(resolved.parent) if resolved.parent != resolved else None
        return {"path": str(resolved), "parent": parent, "entries": entries,
                "roots": _drives()}

    # -- models -----------------------------------------------------------

    def rescan(self) -> list[dict[str, Any]]:
        """Rescan every folder and rebuild the derived groups."""
        self._scanner.scan()
        self._rebuild_groups()
        return self.models()

    def _rebuild_groups(self) -> None:
        records = self._scanner.list()
        if not getattr(self._get_settings(), "duplicate_scan", True):
            self._duplicates = {}
            self._variants = {}
            return
        duplicates, variants = integrity.group_duplicates(records)
        self._duplicates = duplicates
        self._variants = variants

    def all_models(self) -> list[dict[str, Any]]:
        """Every discovered model, enriched with library information."""
        out = []
        for rec in self._scanner.list():
            item = dict(rec)
            item["duplicates"] = self._duplicates.get(rec["id"], [])
            item["variants"] = self._variants.get(rec["id"], [])
            item["duplicate_count"] = len(item["duplicates"])
            item["variant_count"] = len(item["variants"])
            item["quant_compatibility"] = quant_compatibility(
                rec.get("quantization"))
            item["family"] = integrity.family_key(rec.get("name"),
                                                  rec.get("path"))
            item["recommended"] = self._recommend(rec)
            out.append(item)
        return out

    def _recommend(self, record: dict[str, Any]) -> dict[str, Any]:
        """What this model can actually do on this machine right now.

        The context length a client should ask for is the one that fits in
        the free VRAM, not the one the model advertises. A client that
        believes it has 150k tokens will build a prompt the engine rejects.

        The VRAM that is free *right now* is not the right number to use:
        another model may be loaded and holding VRAM that would be released
        when this one is loaded. So the VRAM the other model holds is added
        back, and the recommendation describes this model standing alone.
        """
        free_mb = 0.0
        try:
            gpu = (self._get_sample() or {}).get("gpu") or {}
            free_mb = float(gpu.get("dedicated_free_mb") or 0.0)
        except Exception:  # noqa: BLE001 - a plan is best-effort
            free_mb = 0.0
        free_mb += self._held_by_others()
        maximum = int(record.get("context_length") or 0)

        if free_mb <= 0:
            # The VRAM is not known. The model's own ceiling is still a
            # fact, and a small window is a small window.
            context = maximum or 8192
            out: dict[str, Any] = {
                "context": context,
                "max_context": maximum or None,
                "output": _output_budget(context),
                "confidence": "low",
                "reason": "the free VRAM is not known; using the "
                          "advertised context",
            }
            if context < 8192:
                out["warning"] = _small_context_warning(context)
            return out

        plan = memory_planner.recommend_context(
            record, free_mb, mmproj=bool(record.get("mmproj_path")))
        context = int(plan.get("context") or 8192)
        out = {
            "context": context,
            "max_context": maximum or None,
            "output": _output_budget(context),
            "confidence": plan.get("confidence", "low"),
            "reason": plan.get("reason"),
        }
        if maximum and context >= maximum:
            out["reason"] = (f"the model's own maximum is {maximum:,} tokens"
                             + (f"; {plan.get('reason')}"
                                if plan.get("reason") else ""))
        if context < 8192:
            out["warning"] = _small_context_warning(context)
        return out

    def _held_by_others(self) -> float:
        """VRAM that a currently loaded model holds, which would be freed.

        Only the instances that are running count. The history holds
        records of models that were loaded earlier and are long gone -
        counting their estimates would invent VRAM that does not exist.
        """
        held = 0.0
        try:
            instances = self._get_instances()
        except Exception:  # noqa: BLE001
            return 0.0
        for inst in instances or []:
            if inst.get("state") not in ("ready", "busy", "loading_model",
                                         "starting_engine", "queued"):
                continue
            try:
                estimate = inst.get("estimate") or {}
                held += float(estimate.get("estimated_gpu_mb") or 0)
            except (TypeError, ValueError):
                continue
        return held

    def models(self, query: str | None = None,
               architecture: str | None = None,
               quantization: str | None = None,
               format: str | None = None,
               min_size: int | None = None,
               max_size: int | None = None,
               integrity_status: str | None = None,
               compatibility: str | None = None,
               duplicates_only: bool = False,
               sort: str = "size", order: str = "desc") -> dict[str, Any]:
        """Search the library.

        Every filter is optional and they combine with AND. ``query`` is
        matched against the name, the file name, the path, the architecture
        and the quantization, so one box finds a model by any of the things
        a user would know it by.
        """
        items = self.all_models()
        needle = (query or "").strip().lower()

        def matches(m: dict[str, Any]) -> bool:
            if needle:
                haystack = " ".join(str(m.get(k) or "") for k in
                                    ("name", "path", "architecture",
                                     "quantization", "parameters_label"))
                if needle not in haystack.lower():
                    return False
            if architecture and architecture != "all" \
                    and m.get("architecture") != architecture:
                return False
            if quantization and quantization != "all" \
                    and m.get("quantization") != quantization:
                return False
            if format and format != "all" and m.get("format") != format:
                return False
            size = int(m.get("size_bytes") or 0)
            if min_size is not None and size < min_size:
                return False
            if max_size is not None and size > max_size:
                return False
            if integrity_status and integrity_status != "all" \
                    and (m.get("integrity") or {}).get("status") != integrity_status:
                return False
            if compatibility and compatibility != "all" \
                    and (m.get("compatibility") or {}).get("status") != compatibility:
                return False
            if duplicates_only and not m.get("duplicate_count"):
                return False
            return True

        items = [m for m in items if matches(m)]

        key = {
            "name": lambda m: str(m.get("name") or "").lower(),
            "size": lambda m: int(m.get("size_bytes") or 0),
            "architecture": lambda m: str(m.get("architecture") or "").lower(),
            "quantization": lambda m: str(m.get("quantization") or "").lower(),
            "modified": lambda m: -float(m.get("size_bytes") or 0),
        }.get(sort, lambda m: int(m.get("size_bytes") or 0))
        items.sort(key=key, reverse=(order or "desc").lower() != "asc")

        return {
            "models": items,
            "count": len(items),
            "total": len(self.all_models()),
            "last_scan": self._scanner.last_scan,
            "scan_errors": self._scanner.errors,
            "facets": self.facets(),
        }

    def facets(self) -> dict[str, Any]:
        """The values available for the filter controls."""
        models = self.all_models()
        archs = sorted({str(m.get("architecture")) for m in models
                        if m.get("architecture")})
        quants = sorted({str(m.get("quantization")) for m in models
                         if m.get("quantization")})
        formats = sorted({str(m.get("format")) for m in models
                          if m.get("format")})
        states = sorted({(m.get("integrity") or {}).get("status")
                         for m in models
                         if (m.get("integrity") or {}).get("status")})
        return {"architectures": archs, "quantizations": quants,
                "formats": formats, "integrity_statuses": states}

    def get(self, model_id: str) -> dict[str, Any] | None:
        found = self._find(model_id)
        if found is None:
            # The model may simply not have been scanned yet - a folder was
            # added, a download had just finished, or the folder list
            # changed. The id is derived from the path, so a model that is
            # still on disk in a scanned folder is found by rescanning.
            self.rescan()
            found = self._find(model_id)
        return found

    def _find(self, model_id: str) -> dict[str, Any] | None:
        for m in self.all_models():
            if m["id"] == model_id:
                return m
        return None

    # -- OpenCode ---------------------------------------------------------

    def opencode_config(self) -> opencode_sync.OpenCodeConfig | None:
        """The OpenCode config to keep in step, if there is one."""
        if not getattr(self._get_settings(), "auto_sync_opencode", False):
            return None
        explicit = getattr(self._get_settings(), "opencode_config_path", "")
        path = opencode_sync.find_config(explicit or None)
        if path is None:
            return None
        provider = getattr(self._get_settings(), "opencode_provider",
                           opencode_sync.DEFAULT_PROVIDER)
        return opencode_sync.OpenCodeConfig(path, provider=provider)

    def opencode_status(self) -> dict[str, Any]:
        """What the OpenCode config looks like right now."""
        config = self.opencode_config()
        if config is None:
            return {"enabled": False,
                    "reason": "the automatic OpenCode sync is switched off"
                              if not getattr(self._get_settings(),
                                             "auto_sync_opencode", False)
                              else "no OpenCode config was found"}
        try:
            status = config.status()
        except opencode_sync.ConfigError as exc:
            return {"enabled": True, "error": str(exc)}
        status["enabled"] = True
        status["library_models"] = [m["name"] for m in self.all_models()]
        listed = set(status.get("models") or [])
        known = set(status["library_models"])
        status["missing"] = [n for n in status["library_models"]
                             if n not in listed]
        # Entries the library does not know about are left alone - they
        # were put there by hand, or the model was never in the library.
        status["extra"] = [n for n in (status.get("models") or [])
                           if n not in known]
        return status

    def sync_opencode(self, only: str | None = None) -> dict[str, Any]:
        """Add the library's models to the OpenCode config.

        ``only`` restricts it to a single model - that is what a download
        uses, so the config is touched once per download rather than
        rewritten every time something changes.
        """
        config = self.opencode_config()
        if config is None:
            return {"synced": False,
                    "reason": "no OpenCode config, or the sync is off"}
        added, skipped, removed = [], [], []
        try:
            # A model that is back in the library is not deleted any more -
            # it may have been downloaded again.
            present = {m["name"] for m in self.all_models()}
            still_gone = [n for n in self._deleted_models if n not in present]
            if len(still_gone) != len(self._deleted_models):
                self._deleted_models = still_gone
                self._save_state()

            # Models this library deleted are taken out again - but only
            # those, never one the user listed themselves.
            for name in list(self._deleted_models):
                if config.remove_model(name):
                    removed.append(name)
            if removed:
                self._deleted_models = [n for n in self._deleted_models
                                        if n not in removed]
                self._save_state()
            for model in self.all_models():
                name = model["name"]
                if only is not None and name != only:
                    continue
                if name in self._deleted_models:
                    # A model this library deleted is never put back, even
                    # if a scan still sees the file.
                    continue
                recommended = model.get("recommended") or {}
                spec = {
                    "name": name,
                    "limit": {
                        "context": int(recommended.get("context") or 8192),
                        "output": int(recommended.get("output") or 4096),
                    },
                }
                if config.add_model(name, spec):
                    added.append(name)
                else:
                    skipped.append(name)
        except opencode_sync.ConfigError as exc:
            self._log.warning("library", f"could not update the OpenCode "
                                          f"config: {exc}")
            return {"synced": False, "error": str(exc)}
        if added:
            self._log.info("library",
                           f"added {len(added)} model(s) to the OpenCode "
                           f"config: {', '.join(added)}")
        if removed:
            self._log.info("library",
                           f"removed {len(removed)} deleted model(s) from "
                           f"the OpenCode config: {', '.join(removed)}")
        return {"synced": True, "added": added, "skipped": skipped,
                "removed": removed}

    def unsync_opencode(self, name: str) -> dict[str, Any]:
        """Take a model out of the OpenCode config."""
        config = self.opencode_config()
        if config is None:
            return {"synced": False, "reason": "no OpenCode config"}
        try:
            removed = config.remove_model(name)
        except opencode_sync.ConfigError as exc:
            self._log.warning("library", f"could not update the OpenCode "
                                          f"config: {exc}")
            return {"synced": False, "error": str(exc)}
        if removed:
            self._log.info("library",
                           f"removed {name} from the OpenCode config")
        return {"synced": True, "removed": removed}

    # -- deletion ----------------------------------------------------------

    def delete_plan(self, model_id: str) -> dict[str, Any]:
        """What deleting this model would remove.

        Nothing is deleted here. The plan is what the confirmation dialog
        shows: the exact files, how much space they take, and what would be
        left behind.
        """
        model = self._scanner.get(model_id)
        if model is None:
            raise ValueError(f"model not found: {model_id}")
        mmproj = model.get("mmproj_path")
        files = []
        for entry in model.get("files") or []:
            path = str(entry.get("path") or "")
            if not path:
                continue
            if mmproj and Path(path) == Path(mmproj):
                continue           # the projector is a separate choice
            files.append({"path": path,
                          "size_bytes": int(entry.get("size_bytes") or 0)})
        if not files:
            files = [{"path": model["path"],
                      "size_bytes": int(model.get("size_bytes") or 0)}]
        projector = None
        if mmproj:
            try:
                projector = {"path": mmproj,
                             "size_bytes": os.path.getsize(mmproj)}
            except OSError:
                projector = {"path": mmproj, "size_bytes": 0}

        # Whether the files can be put back matters before the delete, not
        # after it: Windows erases anything too big for the Recycle Bin.
        recoverable = True
        reason = None
        for entry in files:
            ok, why = file_removal.will_be_recoverable(Path(entry["path"]))
            if not ok:
                recoverable = False
                reason = why
                break

        return {
            "model_id": model_id,
            "model_name": model.get("name"),
            "files": files,
            "total_bytes": sum(f["size_bytes"] for f in files),
            "projector": projector,
            "loaded": self._is_loaded(model_id),
            "recoverable": recoverable,
            "recoverable_reason": reason,
        }

    def _is_loaded(self, model_id: str) -> bool:
        try:
            instances = self._get_instances()
        except Exception:  # noqa: BLE001
            return False
        return any(i.get("model_id") == model_id
                   and i.get("state") in ("ready", "busy", "loading_model",
                                          "starting_engine")
                   for i in instances or [])

    def delete_model(self, model_id: str, delete_mmproj: bool = False,
                     permanent: bool = False) -> dict[str, Any]:
        """Delete a model from disk.

        The path always comes from the library's own record - the caller
        cannot ask for an arbitrary file to be removed. The file goes to
        the Recycle Bin unless a permanent delete was asked for.
        """
        plan = self.delete_plan(model_id)
        if plan["loaded"]:
            raise RuntimeError(
                f"{plan['model_name']} is loaded. Unload it before deleting it.")

        removed: list[dict[str, Any]] = []
        for entry in plan["files"]:
            path = Path(entry["path"])
            try:
                removed.append(file_removal.remove(path, permanent=permanent))
            except file_removal.RemovalError as exc:
                self._log.error("library", f"could not delete {path}: {exc}")
                if not removed:
                    raise
                break

        projector = None
        if delete_mmproj and plan.get("projector"):
            path = Path(plan["projector"]["path"])
            try:
                projector = file_removal.remove(path, permanent=permanent)
            except file_removal.RemovalError as exc:
                self._log.warning("library",
                                  f"could not delete {path}: {exc}")

        freed = sum(int(r.get("size_bytes") or 0) for r in removed)
        self._log.info(
            "library",
            f"deleted {plan['model_name']} ({freed / 2**20:.0f} MB) - "
            f"{removed[0]['method'] if removed else 'nothing removed'}")
        self.rescan()
        # A model that is gone should not be offered by the client either.
        name = str(plan["model_name"])
        if name not in self._deleted_models:
            self._deleted_models.append(name)
            self._deleted_models = self._deleted_models[-200:]
            self._save_state()
        self.unsync_opencode(name)
        return {
            "deleted": True,
            "model_id": model_id,
            "model_name": plan["model_name"],
            "files": removed,
            "projector": projector,
            "freed_bytes": freed,
            "permanent": permanent,
        }

    # -- preflight --------------------------------------------------------

    def preflight(self, model_id: str, context: int | None = None,
                  gpu_layers: int | None = None,
                  gpu_mode: str = "auto") -> dict[str, Any]:
        """Run the checks that must pass before a model is loaded.

        Returns a structured report. ``verdict`` is ``ready``, ``risky`` or
        ``blocked``; the caller decides what to do with a risky plan, but a
        blocked one should not be loaded at all.
        """
        model = self.get(model_id)
        if model is None:
            return {"ok": False, "verdict": "blocked", "checks": [],
                    "error": f"model not found: {model_id}. It is not in any "
                             f"scanned folder any more - press Rescan, or "
                             f"import the folder it lives in."}

        settings = self._get_settings()
        checks: list[dict[str, Any]] = []

        def add(name: str, status: str, detail: str) -> None:
            checks.append({"name": name, "status": status, "detail": detail})

        # 1. the file must be there
        path = Path(model.get("path") or "")
        if not path.is_file():
            add("file", "fail", f"file not found: {path}")
        else:
            add("file", "ok", f"{path.stat().st_size:,} bytes on disk")

        # 2. the file must be complete
        report = model.get("integrity") or {}
        status = report.get("status")
        if status == integrity.STATUS_CORRUPT:
            add("integrity", "fail",
                report.get("reason") or "the file is not a readable model")
        elif status == integrity.STATUS_INCOMPLETE:
            add("integrity", "fail",
                report.get("reason") or "the file is incomplete")
        elif status == integrity.STATUS_UNVERIFIED:
            add("integrity", "warn",
                report.get("reason") or "completeness could not be verified")
        else:
            add("integrity", "ok", "the file is complete")

        # 3. the architecture must be implemented by the installed engine
        arch = str(model.get("architecture") or "")
        facts = classify_arch(arch)
        if not arch or arch == "unknown":
            add("architecture", "warn",
                "no architecture in the GGUF metadata; the engine has the "
                "final say")
        elif facts["supported_by_engine"]:
            add("architecture", "ok",
                f"'{arch}' is implemented by llama.cpp {ENGINE_VERSION} "
                f"({ENGINE_TAG})")
        else:
            add("architecture", "warn",
                f"'{arch}' is not in the known architecture list for this "
                f"engine build; loading will be attempted and the engine "
                f"decides")

        # 4. the quantization must be usable
        quant = quant_compatibility(model.get("quantization"))
        add("quantization", quant["status"], quant["note"])

        # 5. a vision model needs its projector
        caps = model.get("capabilities") or {}
        if caps.get("vision"):
            if model.get("mmproj_path"):
                add("vision projector", "ok",
                    f"attached: {Path(model['mmproj_path']).name}")
            else:
                add("vision projector", "warn",
                    "no mmproj file found - the model loads as text-only")

        # 6. the requested context must fit the model
        ctx = int(context or settings.default_ctx)
        max_ctx = model.get("context_length")
        if max_ctx and ctx > int(max_ctx):
            add("context", "fail",
                f"{ctx:,} tokens were requested but the model was trained "
                f"for {int(max_ctx):,}")
        elif max_ctx:
            add("context", "ok",
                f"{ctx:,} of {int(max_ctx):,} tokens (the model maximum)")
        else:
            add("context", "warn",
                f"{ctx:,} tokens; the model does not advertise a maximum")

        # 7. the memory plan has to be survivable
        free_mb = 0.0
        try:
            gpu = (self._get_sample() or {}).get("gpu") or {}
            free_mb = float(gpu.get("dedicated_free_mb") or 0.0)
        except Exception:  # noqa: BLE001
            free_mb = 0.0
        free_mb += self._held_by_others()
        plan = memory_planner.estimate(
            model, ctx,
            gpu_layers if gpu_layers is not None else int(model.get("block_count") or 0),
            batch=int(settings.default_batch), kv_on_gpu=bool(settings.kv_on_gpu),
            mmproj=bool(model.get("mmproj_path")),
            gpu_free_mb=free_mb or None)
        verdict = plan.get("verdict")
        if verdict == "risk_of_paging":
            add("memory", "fail",
                "the plan would page to disk: " +
                "; ".join(plan.get("warnings") or []))
        elif verdict == "partial_offload":
            add("memory", "warn",
                "; ".join(plan.get("warnings") or []) or
                "part of the model will run on the CPU")
        else:
            add("memory", "ok",
                f"about {plan.get('estimated_gpu_mb', 0):,.0f} MB of VRAM and "
                f"{plan.get('estimated_ram_mb', 0) / 1024:.1f} GB of RAM")

        statuses = {c["status"] for c in checks}
        if "fail" in statuses:
            overall = "blocked"
        elif "warn" in statuses:
            overall = "risky"
        else:
            overall = "ready"

        return {
            "ok": overall != "blocked",
            "verdict": overall,
            "model_id": model_id,
            "model_name": model.get("name"),
            "checks": checks,
            "plan": plan,
            "engine": {"tag": ENGINE_TAG, "version": ENGINE_VERSION},
        }
