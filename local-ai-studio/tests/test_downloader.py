"""Download manager tests: approval, resume, disk checks and verification.

Everything runs against a local HTTP server, so the tests never touch the
internet and work the same on a machine with no network at all.
"""
from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pytest

from backend.app.config import Settings
from backend.app.services.download_manager import (
    ApprovalRequired, DownloadError, DownloadManager,
)
from backend.app.services.event_bus import EventBus
from backend.app.services.hf_client import HuggingFaceClient
from backend.app.services.log_store import LogStore

PAYLOAD = b"model weights would go here" * 512
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # keep the test output quiet
        pass

    def _json(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        server = self.server
        server.requests.append(self.path)

        if parsed.path.startswith("/api/models/") and parsed.path.endswith(
                "/tree/main"):
            self._json([{
                "type": "file",
                "path": "model.gguf",
                "size": len(PAYLOAD),
                "lfs": {"oid": f"sha256:{PAYLOAD_SHA256}",
                        "size": len(PAYLOAD)},
            }])
            return

        if parsed.path.endswith("/resolve/main/model.gguf"):
            rng = self.headers.get("Range")
            server.ranges.append(rng)
            if rng and server.supports_range:
                start = int(rng.split("=")[1].split("-")[0])
                body = PAYLOAD[start:]
                self.send_response(206)
                self.send_header("Content-Range",
                                 f"bytes {start}-{len(PAYLOAD) - 1}/"
                                 f"{len(PAYLOAD)}")
            else:
                body = PAYLOAD
                self.send_response(200)
            if server.corrupt:
                body = bytes((b + 1) % 256 for b in body)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    httpd.requests = []
    httpd.ranges = []
    httpd.supports_range = True
    httpd.corrupt = False
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    httpd.base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield httpd
    finally:
        httpd.shutdown()
        httpd.server_close()


def _manager(tmp_path: Path, server, **overrides) -> DownloadManager:
    values = {
        "allow_hf_download": True,
        "download_dir": str(tmp_path / "downloads"),
        "hf_api_base": server.base_url,
        "hf_allowed_hosts": ["127.0.0.1"],
        "max_download_bytes": 1024 ** 3,
    }
    values.update(overrides)
    settings = Settings(**values)
    client = HuggingFaceClient(base_url=server.base_url, timeout=10)
    return DownloadManager(
        log=LogStore(), events=EventBus(), get_settings=lambda: settings,
        client=client, state_file=tmp_path / "downloads.json",
    )


# ---------------------------------------------------------------------------
# The feature is off by default
# ---------------------------------------------------------------------------

def test_downloads_are_refused_when_the_feature_is_off(tmp_path: Path, server):
    """Planning is allowed (it fetches no bytes); transferring is not."""
    import asyncio
    manager = _manager(tmp_path, server, allow_hf_download=False)
    # The plan works - it is metadata only, so the user can see what a
    # download would do before deciding to allow it.
    plan = asyncio.run(manager.plan("owner/model", "model.gguf"))
    assert plan["size_bytes"] == len(PAYLOAD)
    assert not Path(plan["destination"]).exists()

    # But nothing may be transferred.
    with pytest.raises(DownloadError) as exc:
        asyncio.run(manager.start(plan["plan_id"], approve=True))
    assert "turned off" in str(exc.value)
    assert not Path(plan["destination"]).exists()


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

def test_plan_describes_the_download_without_fetching_it(tmp_path: Path, server):
    import asyncio
    manager = _manager(tmp_path, server)
    plan = asyncio.run(manager.plan("owner/model", "model.gguf"))
    assert plan["size_bytes"] == len(PAYLOAD)
    assert plan["hash"] == PAYLOAD_SHA256
    assert plan["hash_algorithm"] == "sha256"
    assert plan["destination"].endswith("owner\\model\\model.gguf") \
        or plan["destination"].endswith("owner/model/model.gguf")
    # Planning must not have downloaded anything.
    assert not Path(plan["destination"]).exists()
    assert not Path(plan["destination"] + ".part").exists()


def test_plan_refuses_a_file_that_is_not_a_model(tmp_path: Path, server):
    import asyncio
    manager = _manager(tmp_path, server)
    with pytest.raises(DownloadError) as exc:
        asyncio.run(manager.plan("owner/model", "readme.md"))
    assert "not a GGUF" in str(exc.value) or "not in" in str(exc.value)


def test_plan_refuses_when_the_disk_is_too_small(tmp_path: Path, server,
                                                  monkeypatch):
    import asyncio
    import shutil
    manager = _manager(tmp_path, server)

    class _Usage:
        total = 100 * 1024 ** 3
        used = 99 * 1024 ** 3
        free = 1024  # one kilobyte left

    monkeypatch.setattr(shutil, "disk_usage", lambda _p: _Usage())
    with pytest.raises(DownloadError) as exc:
        asyncio.run(manager.plan("owner/model", "model.gguf"))
    assert "not enough space" in str(exc.value)


def test_plan_refuses_a_file_over_the_size_limit(tmp_path: Path, server):
    import asyncio
    manager = _manager(tmp_path, server, max_download_bytes=16)
    with pytest.raises(DownloadError) as exc:
        asyncio.run(manager.plan("owner/model", "model.gguf"))
    assert "over the download limit" in str(exc.value)


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------

def test_starting_a_download_requires_approval(tmp_path: Path, server):
    import asyncio
    manager = _manager(tmp_path, server)
    plan = asyncio.run(manager.plan("owner/model", "model.gguf"))
    with pytest.raises(ApprovalRequired):
        asyncio.run(manager.start(plan["plan_id"], approve=False))
    assert not Path(plan["destination"]).exists()


def test_an_unknown_plan_cannot_be_started(tmp_path: Path, server):
    import asyncio
    manager = _manager(tmp_path, server)
    with pytest.raises(DownloadError):
        asyncio.run(manager.start("nonsense", approve=True))


# ---------------------------------------------------------------------------
# The transfer
# ---------------------------------------------------------------------------

def _run_download(manager, plan) -> dict:
    """Start a download and wait for it to finish."""
    import asyncio

    async def go():
        record = await manager.start(plan["plan_id"], approve=True)
        for _ in range(200):
            current = manager.get(record["download_id"])
            if current["status"] in ("completed", "failed", "cancelled"):
                return current
            await asyncio.sleep(0.05)
        return manager.get(record["download_id"])

    return asyncio.run(go())


def test_an_approved_download_is_fetched_and_verified(tmp_path: Path, server):
    import asyncio
    manager = _manager(tmp_path, server)
    plan = asyncio.run(manager.plan("owner/model", "model.gguf"))
    record = _run_download(manager, plan)

    assert record["status"] == "completed", record.get("error")
    assert record["verified"] is True
    destination = Path(plan["destination"])
    assert destination.is_file()
    assert destination.read_bytes() == PAYLOAD
    # The partial file must not be left behind.
    assert not destination.with_name(destination.name + ".part").exists()


def test_a_download_resumes_from_a_partial_file(tmp_path: Path, server):
    import asyncio
    manager = _manager(tmp_path, server)
    plan = asyncio.run(manager.plan("owner/model", "model.gguf"))

    # Pretend a previous transfer got half way.
    destination = Path(plan["destination"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    part = destination.with_name(destination.name + ".part")
    part.write_bytes(PAYLOAD[:1000])

    record = _run_download(manager, plan)
    assert record["status"] == "completed", record.get("error")
    assert destination.read_bytes() == PAYLOAD
    # The server must have been asked for the remainder, not the whole file.
    assert server.ranges == ["bytes=1000-"]
    assert record["downloaded_bytes"] == len(PAYLOAD)


def test_a_download_without_a_partial_file_asks_for_everything(
        tmp_path: Path, server):
    import asyncio
    manager = _manager(tmp_path, server)
    plan = asyncio.run(manager.plan("owner/model", "model.gguf"))
    record = _run_download(manager, plan)
    assert record["status"] == "completed", record.get("error")
    assert all(r is None for r in server.ranges)


def test_a_corrupted_download_is_discarded(tmp_path: Path, server):
    import asyncio
    server.corrupt = True
    manager = _manager(tmp_path, server)
    plan = asyncio.run(manager.plan("owner/model", "model.gguf"))
    record = _run_download(manager, plan)

    assert record["status"] == "failed"
    assert "hash" in (record["error"] or "").lower()
    # Nothing may be promoted to the final name.
    assert not Path(plan["destination"]).exists()
    assert not Path(plan["destination"] + ".part").exists()


def test_a_download_never_runs_anything(tmp_path: Path, server):
    """The downloaded file is a file - it is never loaded or executed."""
    import asyncio
    manager = _manager(tmp_path, server)
    plan = asyncio.run(manager.plan("owner/model", "model.gguf"))
    record = _run_download(manager, plan)
    assert record["status"] == "completed"
    # No instance was created for the downloaded model.
    assert manager.list_downloads()[0]["status"] == "completed"
    assert "instance_id" not in record


def test_cancelling_a_download_marks_it_cancelled(tmp_path: Path, server):
    import asyncio
    manager = _manager(tmp_path, server)
    plan = asyncio.run(manager.plan("owner/model", "model.gguf"))
    record = asyncio.run(manager.start(plan["plan_id"], approve=True))
    cancelled = manager.cancel(record["download_id"])
    assert cancelled["status"] == "cancelled"


def test_a_redirect_to_an_unknown_host_is_refused(tmp_path: Path, server):
    """Only the Hub and its own CDN may ever be fetched from."""
    import asyncio
    manager = _manager(tmp_path, server, hf_allowed_hosts=["example.com"])
    plan = asyncio.run(manager.plan("owner/model", "model.gguf"))
    record = _run_download(manager, plan)
    assert record["status"] == "failed"
    assert "refusing to download" in (record["error"] or "")


def test_the_hub_cdn_family_is_allowed(tmp_path: Path, server):
    """The Hub hands out URLs on several CDN names; the rule is a family."""
    from backend.app.services.download_manager import DownloadManager as DM
    assert DM._host_allowed("huggingface.co", [])
    assert DM._host_allowed("cdn-lfs.hf.co", [])
    assert DM._host_allowed("us.aws.cdn.hf.co", [])
    assert DM._host_allowed("cdn-lfs-us-1.huggingface.co", [])
    assert not DM._host_allowed("evil.example.com", [])
    assert not DM._host_allowed("huggingface.co.evil.com", [])
    # An explicitly listed host is allowed too.
    assert DM._host_allowed("mirror.example.com", ["mirror.example.com"])


def test_a_finished_download_tells_the_library(tmp_path: Path, server):
    """The download folder is handed over so the new model is picked up."""
    import asyncio
    seen: list[str] = []
    manager = _manager(tmp_path, server)
    manager._on_complete = lambda folder: seen.append(str(folder))
    plan = asyncio.run(manager.plan("owner/model", "model.gguf"))
    record = _run_download(manager, plan)
    assert record["status"] == "completed", record.get("error")
    assert seen, "the library was not told about the download"
    assert seen[0] == str(tmp_path / "downloads")


def test_a_download_is_never_added_to_the_scanned_folders_by_itself(
        tmp_path: Path, server):
    """A download must not change what is scanned without being asked."""
    import asyncio
    manager = _manager(tmp_path, server)
    plan = asyncio.run(manager.plan("owner/model", "model.gguf"))
    record = _run_download(manager, plan)
    assert record["status"] == "completed"
    # The manager knows nothing about the settings' folder list.
    assert not hasattr(manager, "_scanner")
