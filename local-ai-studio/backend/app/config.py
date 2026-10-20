"""Application configuration.

Settings live in ``data/settings.json`` (JSON, human-editable). The file is
never overwritten by setup scripts; missing keys fall back to defaults so an
older settings file keeps working after an upgrade.

All filesystem paths are resolved to absolute paths. Original model
directories are treated as read-only: the application never writes, renames or
deletes anything under a configured model directory.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator

# backend/app/config.py -> backend/app -> backend -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
ENGINE_DIR = PROJECT_ROOT / "engine" / "llama.cpp"
DEFAULT_SETTINGS_PATH = DATA_DIR / "settings.json"

DEFAULT_MODEL_DIR = r"C:\Users\SAKISAKI\.lmstudio\models"


class Settings(BaseModel):
    """User-facing configuration. Every field has a safe default."""

    # --- discovery -------------------------------------------------------
    model_dirs: list[str] = Field(default_factory=lambda: [DEFAULT_MODEL_DIR])
    converted_dir: str = str(DATA_DIR / "converted")
    scan_interval_seconds: int = 0  # 0 = manual/rescan-button only

    # --- public API gateway ---------------------------------------------
    api_host: str = "127.0.0.1"
    api_port: int = 1234
    api_key: str | None = None  # None = auth disabled (local-only default)
    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    allow_remote: bool = False  # must be explicitly enabled to bind non-loopback
    max_request_bytes: int = 16 * 1024 * 1024
    request_timeout_seconds: int = 600

    # --- engine ----------------------------------------------------------
    engine_path: str = str(ENGINE_DIR / "llama-server.exe")
    engine_backend: str = "vulkan"  # vulkan | cpu (auto-detected at startup)
    engine_port_start: int = 18400
    engine_port_end: int = 18499

    # --- defaults applied to new instances -------------------------------
    default_ngl: int = 99
    default_ctx: int = 8192
    default_threads: int = 0  # 0 = let the engine decide
    default_batch: int = 512
    default_ubatch: int = 512
    flash_attention: str = "auto"  # auto | on | off
    kv_on_gpu: bool = True
    max_instances: int = 1
    auto_load_on_request: bool = False
    idle_unload_minutes: int = 0  # 0 = never unload automatically
    advanced_args: list[str] = Field(default_factory=list)

    # --- memory guardrails ------------------------------------------------
    auto_adjust_offload: bool = False  # shrink ngl/ctx when VRAM is short
    reserve_ram_mb: int = 2048  # never plan to use the last of physical RAM
    warn_commit_ratio: float = 0.9  # warn when committed memory crosses this

    # --- monitoring -------------------------------------------------------
    sample_interval_seconds: float = 1.0
    history_max_samples: int = 900  # bounded in-memory metric history
    log_max_lines: int = 5000

    # --- model library ----------------------------------------------------
    integrity_check: bool = True   # verify that model files are complete
    duplicate_scan: bool = True    # group identical files and variants
    allow_hf_download: bool = False  # OFF: no outbound traffic by default
    download_dir: str = str(DATA_DIR / "downloads")
    max_download_bytes: int = 80 * 1024 ** 3
    verify_download_sha256: bool = True
    download_free_space_margin_mb: int = 512
    download_timeout_seconds: int = 60
    hf_api_base: str = "https://huggingface.co"
    # Extra hosts to allow besides huggingface.co and its own CDN (anything
    # under huggingface.co or hf.co is always allowed).
    hf_allowed_hosts: list[str] = Field(default_factory=lambda: [
        "huggingface.co",
        "cdn-lfs.hf.co",
        "cdn-lfs-us-1.hf.co",
        "cdn-lfs-eu-1.hf.co",
    ])

    # --- OpenCode ---------------------------------------------------------
    # OpenCode reads its model list from its own config file, so a model
    # that is downloaded later has to be added there to be usable.
    auto_sync_opencode: bool = True
    opencode_config_path: str = ""   # empty = look in the usual places
    opencode_provider: str = "localai"

    @field_validator("api_host")
    @classmethod
    def _validate_host(cls, v: str) -> str:
        if not v or " " in v:
            raise ValueError("api_host must be a non-empty hostname or IP")
        return v

    @field_validator("api_port", "engine_port_start", "engine_port_end")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError(f"port {v} out of range 1-65535")
        return v

    @field_validator("engine_backend")
    @classmethod
    def _validate_backend(cls, v: str) -> str:
        if v not in ("vulkan", "cpu"):
            raise ValueError("engine_backend must be 'vulkan' or 'cpu'")
        return v

    @field_validator("flash_attention")
    @classmethod
    def _validate_fa(cls, v: str) -> str:
        if v not in ("auto", "on", "off"):
            raise ValueError("flash_attention must be auto|on|off")
        return v

    @field_validator("advanced_args")
    @classmethod
    def _validate_advanced(cls, v: list[str]) -> list[str]:
        """Advanced engine arguments are restricted to a safe allowlist.

        The web UI must never be able to inject arbitrary flags (e.g.
        ``--api-key`` of another app, or a path outside approved roots).
        """
        allowed_prefixes = (
            "--temp", "--top-k", "--top-p", "--min-p", "--repeat-penalty",
            "--seed", "--no-mmap", "--mlock", "--cache-reuse", "--flash-attn",
            "--simple-io", "--reasoning-format", "--jinja", "--no-jinja",
            "--slot-save-path", "--defrag-thold", "--numa",
        )
        for arg in v:
            if not any(arg == p or arg.startswith(p + "=") for p in allowed_prefixes):
                raise ValueError(
                    f"advanced argument not allowed: {arg!r}. "
                    f"Allowed prefixes: {', '.join(allowed_prefixes)}"
                )
        return v

    @field_validator("model_dirs")
    @classmethod
    def _validate_dirs(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one model directory is required")
        return v

    @field_validator("download_dir")
    @classmethod
    def _validate_download_dir(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("download_dir must not be empty")
        return v.strip()

    @field_validator("max_download_bytes")
    @classmethod
    def _validate_max_download(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("max_download_bytes must be positive")
        return v

    @field_validator("hf_api_base")
    @classmethod
    def _validate_hf_base(cls, v: str) -> str:
        v = (v or "").strip().rstrip("/")
        if not v.startswith("https://") and not v.startswith("http://"):
            raise ValueError("hf_api_base must be an http(s):// URL")
        if "/" in v.split("://", 1)[1]:
            raise ValueError("hf_api_base must not contain a path")
        host = v.split("://", 1)[1].split(":")[0].lower()
        if v.startswith("http://") and host not in ("127.0.0.1", "localhost",
                                                   "::1"):
            # Plain http is only acceptable for a local test server.
            raise ValueError("hf_api_base must use https unless it points at "
                             "localhost")
        return v

    @field_validator("hf_allowed_hosts")
    @classmethod
    def _validate_hf_hosts(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("at least one allowed download host is required")
        for host in v:
            if not host or "/" in host or " " in host:
                raise ValueError(f"'{host}' is not a plain host name")
        return [h.lower() for h in v]

    # -- helpers -----------------------------------------------------------

    def bound_host(self) -> str:
        """Refuse to bind a public interface unless explicitly allowed."""
        if self.allow_remote:
            return self.api_host
        if self.api_host in ("127.0.0.1", "localhost", "::1"):
            return self.api_host
        return "127.0.0.1"  # silently fall back to loopback, never expose

    def to_public_dict(self) -> dict[str, Any]:
        """Settings as returned to the UI - the API key is masked."""
        data = self.model_dump()
        data["api_key"] = "***set***" if self.api_key else None
        return data


class SettingsStore:
    """Thread-safe settings persistence with atomic writes."""

    def __init__(self, path: Path = DEFAULT_SETTINGS_PATH) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._settings = self._load()

    def _load(self) -> Settings:
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return Settings()
        except (json.JSONDecodeError, OSError) as exc:
            # Never crash on a broken settings file: fall back and keep it.
            print(f"[config] settings file unreadable ({exc}); using defaults")
            return Settings()
        try:
            return Settings(**raw)
        except Exception as exc:  # noqa: BLE001 - report and fall back
            print(f"[config] settings invalid ({exc}); using defaults")
            return Settings()

    @property
    def settings(self) -> Settings:
        with self._lock:
            return self._settings

    def update(self, data: dict[str, Any]) -> Settings:
        """Merge partial update, validate, persist atomically."""
        with self._lock:
            merged = self._settings.model_dump()
            # api_key: masked value means "keep the existing secret"
            if data.get("api_key") == "***set***":
                data = {k: v for k, v in data.items() if k != "api_key"}
            merged.update(data)
            new = Settings(**merged)
            self._settings = new
            self._write(new)
            return new

    def replace(self, data: dict[str, Any]) -> Settings:
        with self._lock:
            new = Settings(**data)
            self._settings = new
            self._write(new)
            return new

    def _write(self, settings: Settings) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps(settings.model_dump(), indent=2), encoding="utf-8"
        )
        os.replace(tmp, self._path)  # atomic on NTFS


def ensure_dirs() -> None:
    """Create application-owned directories (never model directories)."""
    for d in (DATA_DIR, LOGS_DIR, DATA_DIR / "converted"):
        d.mkdir(parents=True, exist_ok=True)
