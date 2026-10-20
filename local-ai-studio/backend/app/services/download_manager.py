"""Resumable model downloads with explicit approval, disk checks and
integrity verification.

The rules this module exists to enforce:

* **Nothing is downloaded until the user says so.** A download always starts
  with a *plan* - a description of exactly what would be fetched, how big it
  is, where it would land, whether it fits on the disk and what hash it will
  be checked against. Starting the download requires that plan id and an
  explicit approval flag. There is no code path that downloads without it.
* **Nothing is executed.** A downloaded file is a file. It is never started,
  never loaded and never handed to the engine as a result of a download; a
  model still has to be loaded explicitly, through the normal load path.
* **Downloads resume.** A partial file is kept as ``<name>.part`` and the
  transfer continues from there with an HTTP range request.
* **Downloads are verified.** The size must match, and the content hash must
  match the one the Hub published. A file that fails verification is
  discarded rather than promoted.
* **Downloads stay inside the download directory.** The destination is
  derived from the plan, never taken from the request, and the Hub host is
  checked on every redirect.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import httpx

from . import integrity
from .hf_client import HuggingFaceClient, HuggingFaceError

# A plan is short-lived on purpose: the user is expected to look at it and
# then act, not to come back hours later.
PLAN_TTL_SECONDS = 15 * 60
MAX_REDIRECTS = 5


class DownloadError(RuntimeError):
    """A download could not be planned or continued."""


class ApprovalRequired(DownloadError):
    """The caller did not approve the download."""


@dataclass
class Plan:
    plan_id: str
    repo_id: str
    filename: str
    url: str
    destination: str
    base_dir: str
    size_bytes: int
    size_human: str
    hash_algorithm: str | None
    hash: str | None
    free_bytes: int
    required_bytes: int
    resume_bytes: int
    warnings: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    @property
    def expires_at(self) -> float:
        return self.created_at + PLAN_TTL_SECONDS

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["expires_at"] = self.expires_at
        data["expired"] = time.time() > self.expires_at
        return data


class DownloadManager:
    """Plans, runs and tracks model downloads."""

    def __init__(self, log, events, get_settings: Callable[[], Any],
                 client: HuggingFaceClient | None = None,
                 state_file: Path | None = None,
                 on_complete: Callable[[Path], None] | None = None) -> None:
        self._log = log
        self._events = events
        self._get_settings = get_settings
        self._client = client or HuggingFaceClient()
        self._state_file = state_file
        # Called with the download folder after a successful transfer, so
        # the library can pick the new model up.
        self._on_complete = on_complete
        self._plans: dict[str, Plan] = {}
        self._downloads: dict[str, dict[str, Any]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._load_state()

    # -- state ------------------------------------------------------------

    def _load_state(self) -> None:
        if not self._state_file:
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        for item in raw.get("downloads", []) or []:
            if isinstance(item, dict) and item.get("download_id"):
                self._downloads[item["download_id"]] = item

    def _save_state(self) -> None:
        if not self._state_file:
            return
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = {"downloads": list(self._downloads.values())[-50:]}
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self._state_file)
        except OSError as exc:
            self._log.warning("downloads", f"could not persist state: {exc}")

    # -- planning ---------------------------------------------------------

    def _download_dir(self) -> Path:
        settings = self._get_settings()
        raw = getattr(settings, "download_dir", "") or ""
        path = Path(os.path.expandvars(raw)).expanduser()
        if not path.is_absolute():
            path = Path(os.path.dirname(os.path.abspath(self._state_file or
                                                          Path.cwd()))) / path
        return path

    def _check_enabled(self) -> None:
        if not getattr(self._get_settings(), "allow_hf_download", False):
            raise DownloadError(
                "downloading is turned off. Enable 'Allow downloads from "
                "Hugging Face' in Settings first - nothing is fetched from "
                "the internet until you do.")

    async def plan(self, repo_id: str, filename: str,
                   target_dir: str | None = None) -> dict[str, Any]:
        """Describe a download without performing it.

        Planning is a read-only metadata request - it fetches the file list
        and works out where the file would go and whether it fits. No file
        bytes are transferred, so it is available whether or not downloading
        is switched on. The transfer itself is what needs both the switch
        and an explicit approval.
        """
        settings = self._get_settings()

        try:
            files = await self._client.list_files(repo_id)
        except HuggingFaceError as exc:
            raise DownloadError(str(exc)) from exc
        wanted = filename.lstrip("/")
        entry = next((f for f in files if f["path"] == wanted), None)
        if entry is None:
            entry = next((f for f in files if f["filename"] == wanted), None)
        if entry is None:
            raise DownloadError(
                f"'{filename}' is not in '{repo_id}'. "
                f"Available files: {', '.join(f['filename'] for f in files[:20])}")
        if not entry["is_model"]:
            raise DownloadError(
                f"'{entry['filename']}' is not a GGUF model file. Only .gguf "
                f"files can be downloaded into the library.")

        size = int(entry.get("size_bytes") or 0)
        max_bytes = int(getattr(settings, "max_download_bytes", 0) or 0)
        warnings: list[str] = []
        if size <= 0:
            raise DownloadError("the Hub did not report a size for this file")
        if max_bytes and size > max_bytes:
            raise DownloadError(
                f"'{entry['filename']}' is {size / 2**30:.1f} GB, which is "
                f"over the download limit of {max_bytes / 2**30:.0f} GB")

        # Destination: the library's own layout (owner / repo / file path),
        # so two files with the same name in different folders cannot collide.
        base = Path(target_dir) if target_dir else self._download_dir()
        base = Path(os.path.expandvars(str(base))).expanduser()
        owner, _, name = repo_id.partition("/")
        relative = entry["path"].replace("\\", "/").strip("/")
        destination = base / (owner or "huggingface") / (name or "model") \
            / relative

        # Disk space: the file plus a small margin for the filesystem.
        margin = int(getattr(settings, "download_free_space_margin_mb", 0)
                     or 0) * 1024 * 1024
        required = int(size * 1.02) + margin
        try:
            base.mkdir(parents=True, exist_ok=True)
            usage = shutil.disk_usage(base)
            free = int(usage.free)
        except OSError as exc:
            raise DownloadError(f"cannot prepare {base}: {exc}") from exc
        if free < required:
            raise DownloadError(
                f"not enough space on {base.drive or base}: {free / 2**30:.1f} GB "
                f"free, but the download needs about {required / 2**30:.1f} GB "
                f"(file {size / 2**30:.1f} GB plus a safety margin)")

        part = destination.with_name(destination.name + ".part")
        resume = 0
        if part.exists():
            resume = part.stat().st_size
            if resume > size:
                resume = 0
                with contextlib.suppress(OSError):
                    part.unlink()
            elif resume:
                warnings.append(
                    f"a partial download of {resume / 2**30:.2f} GB is already "
                    f"there and will be continued")

        if not entry.get("hash"):
            warnings.append(
                "the Hub did not publish a hash for this file - it will be "
                "checked by size only")
        if not entry.get("is_model"):
            warnings.append("this file is not a model file")

        plan = Plan(
            plan_id=uuid.uuid4().hex[:12],
            repo_id=repo_id,
            filename=entry["path"],
            url=self._client.resolve_url(repo_id, entry["path"]),
            destination=str(destination),
            base_dir=str(base),
            size_bytes=size,
            size_human=entry["size_human"],
            hash_algorithm=entry.get("hash_algorithm"),
            hash=entry.get("hash"),
            free_bytes=free,
            required_bytes=required,
            resume_bytes=resume,
            warnings=warnings,
        )
        self._plans[plan.plan_id] = plan
        self._log.info("downloads",
                       f"planned {repo_id}/{entry['filename']} "
                       f"({entry['size_human']}) -> {destination}")
        return plan.to_dict()

    # -- starting ---------------------------------------------------------

    async def start(self, plan_id: str, approve: bool = False
                    ) -> dict[str, Any]:
        """Start a planned download. Requires an explicit approval."""
        self._check_enabled()
        if not approve:
            raise ApprovalRequired(
                "this download was not approved. Review the plan and start it "
                "again with approval.")
        plan = self._plans.get(plan_id)
        if plan is None:
            raise DownloadError("unknown download plan - make a new plan")
        if time.time() > plan.expires_at:
            self._plans.pop(plan_id, None)
            raise DownloadError("this plan has expired - make a new plan")

        download_id = uuid.uuid4().hex[:12]
        record = {
            "download_id": download_id,
            "plan_id": plan.plan_id,
            "repo_id": plan.repo_id,
            "filename": plan.filename,
            "destination": plan.destination,
            "size_bytes": plan.size_bytes,
            "downloaded_bytes": plan.resume_bytes,
            "status": "starting",
            "error": None,
            "started_at": time.time(),
            "finished_at": None,
        }
        self._downloads[download_id] = record
        task = asyncio.create_task(self._run(plan, record))
        self._tasks[download_id] = task
        task.add_done_callback(lambda _t, d=download_id:
                               self._tasks.pop(d, None))
        self._events.publish("download_state", dict(record), download_id)
        return dict(record)

    def cancel(self, download_id: str) -> dict[str, Any]:
        record = self._downloads.get(download_id)
        if record is None:
            raise DownloadError(f"unknown download {download_id}")
        task = self._tasks.get(download_id)
        if task and not task.done():
            task.cancel()
        record["status"] = "cancelled"
        record["finished_at"] = time.time()
        self._events.publish("download_state", dict(record), download_id)
        self._save_state()
        return dict(record)

    def list_downloads(self) -> list[dict[str, Any]]:
        return sorted(self._downloads.values(),
                      key=lambda r: r.get("started_at") or 0, reverse=True)

    def get(self, download_id: str) -> dict[str, Any] | None:
        record = self._downloads.get(download_id)
        return dict(record) if record else None

    # -- the transfer -----------------------------------------------------

    def _check_host(self, url: str) -> None:
        """Refuse to fetch from anywhere but the Hub and its CDN.

        The Hub hands out signed URLs on several different CDN host names
        (and the set changes over time), so the rule is expressed as a
        domain family rather than a fixed list: anything under
        ``huggingface.co`` or ``hf.co`` is the Hub's own infrastructure.
        Extra hosts can be added in Settings for a mirror.
        """
        settings = self._get_settings()
        extra = [h.lower() for h in
                 getattr(settings, "hf_allowed_hosts",
                         ["huggingface.co", "cdn-lfs.hf.co"])]
        host = (urlparse(url).hostname or "").lower()
        if not host:
            raise DownloadError(f"cannot make sense of the URL {url!r}")
        if self._host_allowed(host, extra):
            return
        raise DownloadError(
            f"refusing to download from '{host}': only huggingface.co, its "
            f"CDN (hf.co) and the hosts listed in Settings "
            f"({', '.join(extra)}) are allowed")

    @staticmethod
    def _host_allowed(host: str, extra: list[str]) -> bool:
        if host in extra:
            return True
        for domain in ("huggingface.co", "hf.co"):
            if host == domain or host.endswith("." + domain):
                return True
        return False

    async def _open(self, client: httpx.AsyncClient, url: str,
                    headers: dict[str, str]) -> httpx.Response:
        """GET with redirects followed by hand, so every hop is checked."""
        current = url
        for _ in range(MAX_REDIRECTS):
            self._check_host(current)
            req = client.build_request("GET", current, headers=headers)
            resp = await client.send(req, stream=True)
            if resp.status_code in (301, 302, 303, 307, 308):
                location = resp.headers.get("location")
                await resp.aclose()
                if not location:
                    raise DownloadError("the server sent a redirect with no "
                                        "destination")
                current = urljoin(current, location)
                continue
            return resp
        raise DownloadError("too many redirects")

    async def _run(self, plan: Plan, record: dict[str, Any]) -> None:
        settings = self._get_settings()
        destination = Path(plan.destination)
        part = destination.with_name(destination.name + ".part")
        timeout = float(getattr(settings, "download_timeout_seconds", 60) or 60)
        headers = {"User-Agent": "local-ai-studio/0.1"}
        written = 0
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            resume = 0
            if part.exists():
                resume = part.stat().st_size
                if resume > plan.size_bytes:
                    resume = 0
            if resume:
                headers["Range"] = f"bytes={resume}-"

            record["status"] = "downloading"
            self._events.publish("download_state", dict(record),
                                 record["download_id"])

            timeout_cfg = httpx.Timeout(timeout, read=timeout)
            async with httpx.AsyncClient(timeout=timeout_cfg,
                                         follow_redirects=False) as client:
                resp = await self._open(client, plan.url, headers)
                if resp.status_code >= 400:
                    body = (await resp.aread())[:200].decode(errors="replace")
                    await resp.aclose()
                    raise DownloadError(
                        f"the server answered {resp.status_code}: {body}")

                mode = "ab"
                if resp.status_code == 206 and resume:
                    mode = "ab"
                elif resp.status_code == 200:
                    # The server ignored the range request - start over.
                    if resume:
                        self._log.info("downloads",
                                       "server ignored the range request; "
                                       "restarting the transfer")
                    resume = 0
                    mode = "wb"
                else:
                    mode = "wb"

                total = plan.size_bytes
                length = resp.headers.get("content-length")
                if length and length.isdigit() and resp.status_code == 200:
                    total = int(length)

                record["size_bytes"] = total
                record["downloaded_bytes"] = resume
                written = resume
                last_publish = 0.0

                with part.open(mode) as fh:
                    async for chunk in resp.aiter_bytes(1024 * 256):
                        if not chunk:
                            continue
                        fh.write(chunk)
                        written += len(chunk)
                        now = time.time()
                        if now - last_publish >= 0.5:
                            last_publish = now
                            record["downloaded_bytes"] = written
                            record["percent"] = round(
                                100.0 * written / total, 1) if total else None
                            record["speed_bps"] = int(
                                (written - resume) / max(0.001,
                                                        now - (record.get(
                                                            "started_at") or now)))
                            self._events.publish("download_state",
                                                 dict(record),
                                                 record["download_id"])
                await resp.aclose()

            record["downloaded_bytes"] = written

            # -- verification ------------------------------------------------
            actual = part.stat().st_size
            if actual != plan.size_bytes:
                raise DownloadError(
                    f"the download ended at {actual:,} bytes but the file "
                    f"should be {plan.size_bytes:,} bytes. The partial file "
                    f"was kept so the download can be continued.")

            if plan.hash and plan.hash_algorithm:
                if not getattr(settings, "verify_download_sha256", True):
                    record["verified"] = False
                    record["verification"] = "skipped by settings"
                else:
                    digest = await asyncio.to_thread(
                        integrity.hash_file, part, plan.hash_algorithm)
                    if not digest or digest.lower() != plan.hash.lower():
                        with contextlib.suppress(OSError):
                            part.unlink()
                        raise DownloadError(
                            "the downloaded file does not match the hash the "
                            "Hub published - it was discarded and nothing "
                            "was installed")
                    record["verified"] = True
                    record["hash"] = digest
            else:
                record["verified"] = None
                record["verification"] = "no hash published for this file"

            # -- promote -----------------------------------------------------
            if destination.exists():
                with contextlib.suppress(OSError):
                    destination.unlink()
            os.replace(part, destination)
            record["status"] = "completed"
            record["finished_at"] = time.time()
            record["percent"] = 100.0
            self._log.info("downloads",
                           f"downloaded {plan.repo_id}/{plan.filename} to "
                           f"{destination}")
            self._events.publish("download_state", dict(record),
                                 record["download_id"])
            self._events.publish("download_complete", {
                "repo_id": plan.repo_id,
                "filename": plan.filename,
                "destination": str(destination),
                "size_bytes": plan.size_bytes,
            }, record["download_id"])
            self._save_state()
            if self._on_complete:
                # Let the library pick the new model up. A failure here must
                # not turn a finished download into a failed one.
                with contextlib.suppress(Exception):
                    self._on_complete(Path(plan.base_dir))
        except asyncio.CancelledError:
            record["status"] = "cancelled"
            record["finished_at"] = time.time()
            self._events.publish("download_state", dict(record),
                                 record["download_id"])
            raise
        except Exception as exc:  # noqa: BLE001 - report every failure
            record["status"] = "failed"
            record["error"] = str(exc)
            record["finished_at"] = time.time()
            self._log.error("downloads",
                            f"download of {plan.repo_id}/{plan.filename} "
                            f"failed: {exc}")
            self._events.publish("download_state", dict(record),
                                 record["download_id"])
            self._save_state()
