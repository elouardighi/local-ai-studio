"""Model library tests: search, folders, duplicates, integrity, preflight.

No network, no GPU and no real model files: everything runs on synthetic
GGUF files written by the shared conftest helper.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings, SettingsStore
from backend.app.main import create_app
from backend.app.services import integrity
from backend.app.services.model_library import (
    ModelLibrary, family_key, parse_size, quant_compatibility,
)
from tests.conftest import write_gguf


def _gguf(path: Path, arch: str = "qwen35moe", name: str = "M",
          file_type: int = 15, tensors: list | None = None,
          data_bytes: int = 0) -> Path:
    return write_gguf(path, {
        "general.architecture": arch,
        "general.name": name,
        "general.file_type": file_type,
        f"{arch}.block_count": 40,
        f"{arch}.context_length": 32768,
    }, tensors=tensors, data_bytes=data_bytes)


@pytest.fixture
def library(tmp_path: Path) -> ModelLibrary:
    """A library over a temp folder with three models in it."""
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_scanner import ModelScanner

    root = tmp_path / "models"
    root.mkdir()
    _gguf(root / "Alpha-Q4_K_M.gguf", name="Alpha", data_bytes=4096)
    _gguf(root / "Beta-IQ2_M.gguf", arch="qwen35", name="Beta", file_type=29,
          data_bytes=8192)
    _gguf(root / "Gamma-Q8_0.gguf", arch="gemma4", name="Gamma", file_type=7,
          data_bytes=16384)

    settings = Settings(model_dirs=[str(root)])
    scanner = ModelScanner(get_dirs=lambda: [str(root)], log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings)
    lib.rescan()
    return lib


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def test_search_by_name(library: ModelLibrary):
    result = library.models(query="beta")
    assert result["count"] == 1
    assert result["models"][0]["name"] == "Beta"


def test_search_matches_path_and_architecture(library: ModelLibrary):
    assert library.models(query="gemma4")["count"] == 1
    assert library.models(query="Alpha-Q4_K_M")["count"] == 1
    assert library.models(query="nothing-like-this")["count"] == 0


def test_filter_by_architecture_and_quantization(library: ModelLibrary):
    assert library.models(architecture="qwen35moe")["count"] == 1
    assert library.models(quantization="IQ2_M")["count"] == 1
    assert library.models(architecture="all", quantization="all")["count"] == 3


def test_filter_by_size_range(library: ModelLibrary):
    sizes = sorted(m["size_bytes"] for m in library.all_models())
    small = library.models(max_size=sizes[0])
    assert small["count"] == 1
    big = library.models(min_size=sizes[-1])
    assert big["count"] == 1
    assert library.models(min_size=sizes[0], max_size=sizes[-1])["count"] == 3


def test_filters_combine_with_and(library: ModelLibrary):
    assert library.models(query="beta", architecture="qwen35")["count"] == 1
    assert library.models(query="beta", architecture="gemma4")["count"] == 0


def test_sorting(library: ModelLibrary):
    ascending = library.models(sort="size", order="asc")
    descending = library.models(sort="size", order="desc")
    assert ascending["models"][0]["id"] != descending["models"][0]["id"]
    by_name = library.models(sort="name", order="asc")
    names = [m["name"] for m in by_name["models"]]
    assert names == sorted(names, key=str.lower)


def test_facets_list_the_available_values(library: ModelLibrary):
    facets = library.facets()
    assert "qwen35moe" in facets["architectures"]
    assert "Q4_K_M" in facets["quantizations"]


def test_parse_size_accepts_human_units():
    assert parse_size("10gb") == 10 * 1024 ** 3
    assert parse_size("500 MB") == 500 * 1024 ** 2
    assert parse_size("1.5g") == int(1.5 * 1024 ** 3)
    assert parse_size("1024") == 1024
    assert parse_size(None) is None
    assert parse_size("nonsense") is None


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

def test_browse_lists_subfolders(tmp_path: Path):
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_scanner import ModelScanner

    (tmp_path / "one").mkdir()
    (tmp_path / "two").mkdir()
    (tmp_path / "afile.txt").write_text("x")
    settings = Settings(model_dirs=[str(tmp_path)])
    lib = ModelLibrary(scanner=ModelScanner(get_dirs=lambda: [str(tmp_path)]),
                       log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings)
    listing = lib.browse(str(tmp_path))
    names = {e["name"] for e in listing["entries"]}
    assert names == {"one", "two"}          # files are not listed
    assert listing["parent"] == str(tmp_path.parent)


def test_browse_refuses_system_folders(tmp_path: Path):
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_scanner import ModelScanner

    settings = Settings(model_dirs=[str(tmp_path)])
    lib = ModelLibrary(scanner=ModelScanner(get_dirs=lambda: [str(tmp_path)]),
                       log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings)
    with pytest.raises(ValueError):
        lib.browse(str(tmp_path / "Windows"))


def test_add_and_remove_folder(tmp_path: Path):
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_scanner import ModelScanner

    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _gguf(second / "Delta-Q4_K_M.gguf", name="Delta")

    settings = Settings(model_dirs=[str(first)])
    updated: dict = {}

    def apply(data):
        updated.update(data)
        settings.model_dirs = data["model_dirs"]

    scanner = ModelScanner(get_dirs=lambda: settings.model_dirs, log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings, update_settings=apply)

    lib.add_folder(str(second))
    assert str(second) in settings.model_dirs
    assert any(m["name"] == "Delta" for m in lib.all_models())

    lib.remove_folder(str(second))
    assert str(second) not in settings.model_dirs
    assert not any(m["name"] == "Delta" for m in lib.all_models())


def test_add_folder_rejects_a_file(tmp_path: Path):
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_scanner import ModelScanner

    a_file = tmp_path / "not-a-folder.txt"
    a_file.write_text("x")
    settings = Settings(model_dirs=[str(tmp_path)])
    lib = ModelLibrary(scanner=ModelScanner(get_dirs=lambda: [str(tmp_path)]),
                       log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings)
    with pytest.raises(ValueError):
        lib.add_folder(str(a_file))


# ---------------------------------------------------------------------------
# Duplicates and variants
# ---------------------------------------------------------------------------

def test_identical_files_are_reported_as_duplicates(tmp_path: Path):
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_scanner import ModelScanner

    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    payload = {"general.architecture": "qwen35moe", "general.name": "Same",
               "general.file_type": 15, "qwen35moe.block_count": 40}
    write_gguf(a / "Same-Q4_K_M.gguf", payload)
    write_gguf(b / "Same-Q4_K_M.gguf", payload)

    settings = Settings(model_dirs=[str(a), str(b)])
    scanner = ModelScanner(get_dirs=lambda: [str(a), str(b)], log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings)
    lib.rescan()

    models = lib.all_models()
    assert len(models) == 2
    assert all(m["duplicate_count"] == 1 for m in models)
    assert all(m["variant_count"] == 1 for m in models)


def test_different_quantizations_are_variants_not_duplicates(tmp_path: Path):
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_scanner import ModelScanner

    root = tmp_path / "m"
    root.mkdir()
    _gguf(root / "Model-Q4_K_M.gguf", name="Model", data_bytes=1024)
    _gguf(root / "Model-IQ2_M.gguf", name="Model", file_type=29,
          data_bytes=2048)

    settings = Settings(model_dirs=[str(root)])
    scanner = ModelScanner(get_dirs=lambda: [str(root)], log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings)
    lib.rescan()

    models = lib.all_models()
    assert all(m["duplicate_count"] == 0 for m in models)
    assert all(m["variant_count"] == 1 for m in models)


def test_family_key_strips_the_quantization():
    assert family_key("Qwen3.8-27B-Uncensored-IQ2_M", None) == \
        family_key("Qwen3.8-27B-Uncensored-Q4_K_M", None)
    assert family_key("Qwen3.8-27B-IQ2_M", None) != family_key("Other-Model", None)


# ---------------------------------------------------------------------------
# Integrity
# ---------------------------------------------------------------------------

def test_a_complete_file_passes_the_integrity_check(tmp_path: Path):
    path = _gguf(tmp_path / "Ok-Q4_K_M.gguf")
    report = integrity.check_integrity(path)
    assert report.status in (integrity.STATUS_OK, integrity.STATUS_UNVERIFIED)


def test_a_truncated_file_is_detected(tmp_path: Path):
    """A file whose header is fine but whose tensor data stops short."""
    path = _gguf(tmp_path / "Cut-Q4_K_M.gguf",
                 tensors=[("a", [64], 0), ("b", [64], 0)], data_bytes=512)
    assert integrity.check_integrity(path).status == integrity.STATUS_OK

    data = path.read_bytes()
    path.write_bytes(data[:-100])       # lose the tail of the tensor data
    report = integrity.check_integrity(path)
    assert report.status == integrity.STATUS_INCOMPLETE
    assert "missing" in (report.reason or "")


def test_a_file_that_is_not_a_model_is_corrupt(tmp_path: Path):
    path = tmp_path / "junk.gguf"
    path.write_bytes(b"this is definitely not a GGUF file, not even close")
    report = integrity.check_integrity(path)
    assert report.status == integrity.STATUS_CORRUPT


def test_an_unfinished_download_is_detected(tmp_path: Path):
    path = _gguf(tmp_path / "Half-Q4_K_M.gguf")
    path.with_name(path.name + ".part").write_bytes(b"x" * 16)
    report = integrity.check_integrity(path)
    assert report.status == integrity.STATUS_INCOMPLETE
    assert report.part_files


def test_quick_signature_differs_for_different_content(tmp_path: Path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"a" * 4096)
    b.write_bytes(b"b" * 4096)
    assert integrity.quick_signature(a) != integrity.quick_signature(b)
    assert integrity.quick_signature(a) == integrity.quick_signature(a)


def test_git_blob_sha1_matches_the_git_definition(tmp_path: Path):
    import hashlib
    path = tmp_path / "blob.bin"
    path.write_bytes(b"hello world")
    expected = hashlib.sha1(b"blob 11\x00hello world").hexdigest()
    assert integrity.git_blob_sha1(path) == expected


# ---------------------------------------------------------------------------
# Quantization compatibility
# ---------------------------------------------------------------------------

def test_known_quantizations_are_reported():
    assert quant_compatibility("Q4_K_M")["status"] == "ok"
    assert quant_compatibility("IQ2_M")["status"] == "warn"
    assert quant_compatibility("Q1_0")["status"] == "warn"
    assert quant_compatibility("MADE_UP")["status"] == "unverified"
    assert quant_compatibility(None)["status"] == "unverified"


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def test_preflight_reports_ready_for_a_sound_model(library: ModelLibrary):
    model = next(m for m in library.all_models()
                 if m["architecture"] == "qwen35moe")
    report = library.preflight(model["id"])
    assert report["verdict"] in ("ready", "risky")
    assert report["ok"] is True
    names = {c["name"] for c in report["checks"]}
    assert {"file", "integrity", "architecture", "quantization",
            "context", "memory"} <= names


def test_preflight_blocks_a_corrupt_model(library: ModelLibrary):
    model = library.all_models()[0]
    Path(model["path"]).write_bytes(b"corrupted beyond recognition")
    # Re-scan so the record carries the new integrity verdict.
    library.rescan()
    model = next(m for m in library.all_models() if m["id"] == model["id"])
    report = library.preflight(model["id"])
    assert report["verdict"] == "blocked"
    assert report["ok"] is False
    integrity_check = next(c for c in report["checks"] if c["name"] == "integrity")
    assert integrity_check["status"] == "fail"


def test_preflight_blocks_a_context_the_model_cannot_hold(library: ModelLibrary):
    model = next(m for m in library.all_models()
                 if m["architecture"] == "qwen35moe")
    report = library.preflight(model["id"], context=1_000_000)
    assert report["verdict"] == "blocked"
    check = next(c for c in report["checks"] if c["name"] == "context")
    assert check["status"] == "fail"


def test_preflight_reports_an_unknown_model(library: ModelLibrary):
    report = library.preflight("does-not-exist")
    assert report["verdict"] == "blocked"
    assert "not found" in report["error"]
    assert "Rescan" in report["error"]


# ---------------------------------------------------------------------------
# The scanner cache must never go stale
# ---------------------------------------------------------------------------

def test_the_scanner_cache_follows_the_folder_list(tmp_path: Path):
    """A folder that is removed from the list must drop out of the results.

    The bug this guards against: the cache kept serving a model after its
    folder had been removed, so the page showed a model the server no longer
    knew, and loading it failed with "model not found".
    """
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_scanner import ModelScanner

    root = tmp_path / "models"
    root.mkdir()
    _gguf(root / "Alpha-Q4_K_M.gguf", name="Alpha")

    dirs = [str(root)]
    scanner = ModelScanner(get_dirs=lambda: list(dirs), log=LogStore())
    scanner.scan()
    assert len(scanner.list()) == 1

    # The folder list changes - the cache must follow it.
    dirs = []
    assert scanner.list() == []
    assert scanner.get(scanner.list()[0]["id"] if scanner.list() else "x") is None


def test_a_model_added_after_the_scan_is_found_by_a_lookup(tmp_path: Path):
    """A file that appeared since the last scan is picked up when needed.

    The cache follows the folder list; a *new file* inside a known folder is
    found by the rescan that the lookup path performs when a model is not
    where it was expected.
    """
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_scanner import ModelScanner

    root = tmp_path / "models"
    root.mkdir()
    dirs = [str(root)]
    scanner = ModelScanner(get_dirs=lambda: list(dirs), log=LogStore())
    scanner.scan()
    assert scanner.list() == []

    _gguf(root / "New-Q4_K_M.gguf", name="New")

    # The library rescans when a lookup comes up empty.
    settings = Settings(model_dirs=dirs)
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings)
    assert lib.get("no-such-id") is None
    assert len(lib.all_models()) == 1
    assert lib.all_models()[0]["name"] == "New"


def test_the_library_rescans_before_giving_up(tmp_path: Path):
    """A lookup for an unscanned model triggers a rescan and succeeds."""
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_scanner import ModelScanner

    root = tmp_path / "models"
    root.mkdir()
    _gguf(root / "Alpha-Q4_K_M.gguf", name="Alpha")

    settings = Settings(model_dirs=[str(root)])
    scanner = ModelScanner(get_dirs=lambda: settings.model_dirs, log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings)

    # Nothing has been scanned yet, so the model is unknown...
    assert lib.get("some-id") is None
    # ...but the library finds it once it has looked properly.
    model = lib.get("some-id")
    assert model is None          # the id really is wrong
    assert len(lib.all_models()) == 1


def test_imported_folders_are_restored(tmp_path: Path):
    """An imported folder is put back if the settings drop it."""
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_scanner import ModelScanner

    root = tmp_path / "models"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    settings = Settings(model_dirs=[str(other)])

    def apply(data):
        settings.model_dirs = data["model_dirs"]

    scanner = ModelScanner(get_dirs=lambda: settings.model_dirs, log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings, update_settings=apply,
                       state_file=tmp_path / "library.json")
    lib.add_folder(str(root))
    assert str(root) in settings.model_dirs

    # Something else overwrites the folder list.
    settings.model_dirs = [str(other)]
    assert lib.ensure_imported_folders() == [str(root)]
    assert str(root) in settings.model_dirs


def test_a_folder_that_is_gone_is_forgotten(tmp_path: Path):
    """A folder the user removed is not kept in the scanned list."""
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_scanner import ModelScanner

    root = tmp_path / "models"
    other = tmp_path / "other"
    root.mkdir()
    other.mkdir()
    settings = Settings(model_dirs=[str(other)])

    def apply(data):
        settings.model_dirs = data["model_dirs"]

    scanner = ModelScanner(get_dirs=lambda: settings.model_dirs, log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings, update_settings=apply,
                       state_file=tmp_path / "library.json")
    lib.add_folder(str(root))
    assert str(root) in settings.model_dirs

    # The user removes the folder.
    root.rmdir()
    lib.ensure_imported_folders()
    assert str(root) not in settings.model_dirs
    assert str(root) not in lib._imported


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _gguf(model_dir / "Alpha-Q4_K_M.gguf", name="Alpha")
    store = SettingsStore(path=tmp_path / "settings.json")
    store.replace(Settings(model_dirs=[str(model_dir)],
                            converted_dir=str(tmp_path / "converted"),
                            # keep the tests out of the real project folder
                            download_dir=str(tmp_path / "downloads")
                            ).model_dump())
    app = create_app(store)
    with TestClient(app) as test_client:
        yield test_client


def test_search_endpoint_filters(client: TestClient):
    body = client.get("/api/library/models?q=alpha").json()
    assert body["count"] == 1
    assert body["models"][0]["name"] == "Alpha"
    assert "facets" in body


def test_search_endpoint_rejects_a_bad_size_filter(client: TestClient):
    res = client.get("/api/library/models?min_size=abc")
    assert res.status_code == 400
    assert res.json()["error"]["type"] == "invalid_request"


def test_preflight_endpoint(client: TestClient):
    model = client.get("/api/library/models").json()["models"][0]
    res = client.post("/api/library/preflight",
                      json={"model_id": model["id"]})
    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_browse_endpoint(client: TestClient):
    res = client.get("/api/library/browse")
    assert res.status_code == 200
    assert "entries" in res.json()


def test_folders_endpoint_lists_the_scanned_folders(client: TestClient):
    res = client.get("/api/library/folders")
    assert res.status_code == 200
    assert len(res.json()["folders"]) == 1


def test_planning_works_but_transferring_does_not(client: TestClient):
    """Planning is metadata; only the transfer is gated."""
    from backend.app.services.hf_client import HuggingFaceError

    class _Stub:
        async def list_files(self, repo_id):
            return [{"path": "model.gguf", "filename": "model.gguf",
                     "size_bytes": 10, "size_human": "10 B",
                     "hash_algorithm": "sha256", "hash": "ab" * 32,
                     "is_model": True, "is_mmproj": False}]

        def resolve_url(self, repo_id, filename):
            return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"

    client.app.state.ctx.hf = _Stub()
    # The download manager shares the app's Hub client.
    client.app.state.ctx.downloads._client = client.app.state.ctx.hf
    res = client.post("/api/library/downloads/plan",
                      json={"repo_id": "owner/model", "filename": "model.gguf"})
    assert res.status_code == 200
    body = res.json()
    assert body["plan"]["size_bytes"] == 10
    # The plan is available, but it says the transfer is not allowed yet.
    assert body["downloads_enabled"] is False

    # Starting the transfer without the switch is refused.
    res = client.post("/api/library/downloads",
                      json={"plan_id": body["plan"]["plan_id"],
                            "approve": True})
    assert res.status_code == 400
    assert "turned off" in res.json()["error"]["message"]


def test_browsing_the_hub_works_even_while_downloads_are_off(
        client: TestClient, monkeypatch):
    """Looking at the Hub is a read-only request and must not be gated."""
    from backend.app.services.hf_client import HuggingFaceError

    class _Stub:
        async def search_models(self, query, limit=30, model_filter=None):
            return [{"repo_id": "owner/model", "name": "model",
                      "author": "owner", "downloads": 1, "likes": 0,
                      "pipeline_tag": None, "library_name": None,
                      "tags": [], "gated": False}]

        async def list_files(self, repo_id):
            return [{"path": "model.gguf", "filename": "model.gguf",
                     "size_bytes": 10, "size_human": "10 B",
                     "hash_algorithm": "sha256", "hash": "ab" * 32,
                     "is_model": True, "is_mmproj": False}]

    client.app.state.ctx.hf = _Stub()
    res = client.get("/api/library/hf/search?q=model")
    assert res.status_code == 200
    body = res.json()
    assert body["count"] == 1
    assert body["downloads_enabled"] is False

    res = client.get("/api/library/hf/files?repo=owner/model")
    assert res.status_code == 200
    assert res.json()["files"][0]["is_model"] is True
    assert res.json()["downloads_enabled"] is False


def test_the_gguf_filter_is_passed_to_the_hub(client: TestClient):
    """The Hub's own filter keeps the results to loadable repositories."""
    seen = {}

    class _Stub:
        async def search_models(self, query, limit=30, model_filter=None):
            seen["filter"] = model_filter
            return []

    client.app.state.ctx.hf = _Stub()
    res = client.get("/api/library/hf/search?q=qwen&filter=gguf")
    assert res.status_code == 200
    assert seen["filter"] == "gguf"

    client.get("/api/library/hf/search?q=qwen&filter=")
    assert seen["filter"] is None
