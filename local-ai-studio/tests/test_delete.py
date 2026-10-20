"""Deleting a model: the rules that keep it safe.

Deletion is the one destructive thing this project does, so the rules are
pinned down here:

* the path always comes from the library's own record - a caller cannot ask
  for an arbitrary file to be removed;
* a loaded model cannot be deleted;
* nothing is deleted without an explicit confirmation;
* the file goes to the Recycle Bin, so it can be put back.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings, SettingsStore
from backend.app.main import create_app
from backend.app.services import file_removal
from tests.conftest import write_gguf


def _model(path: Path, name: str = "Alpha") -> Path:
    return write_gguf(path, {
        "general.architecture": "qwen35moe",
        "general.name": name,
        "general.file_type": 15,
        "qwen35moe.block_count": 40,
        "qwen35moe.context_length": 32768,
    })


@pytest.fixture
def library(tmp_path: Path):
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_library import ModelLibrary
    from backend.app.services.model_scanner import ModelScanner

    root = tmp_path / "models"
    root.mkdir()
    _model(root / "Alpha-Q4_K_M.gguf", "Alpha")
    _model(root / "Beta-Q4_K_M.gguf", "Beta")

    settings = Settings(model_dirs=[str(root)])
    scanner = ModelScanner(get_dirs=lambda: settings.model_dirs, log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings,
                       get_instances=lambda: [])
    lib.rescan()
    return lib


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def test_the_plan_lists_the_exact_files(library, tmp_path: Path):
    model = library.all_models()[0]
    plan = library.delete_plan(model["id"])
    assert len(plan["files"]) == 1
    assert plan["files"][0]["path"] == model["path"]
    assert plan["total_bytes"] > 0
    assert plan["loaded"] is False


def test_the_plan_keeps_the_projector_separate(library, tmp_path: Path):
    """A projector is a separate choice, never deleted by accident."""
    root = tmp_path / "models"
    _model(root / "mmproj-Alpha-f16.gguf", "proj")
    library.rescan()
    model = next(m for m in library.all_models() if m["name"] == "Alpha")
    plan = library.delete_plan(model["id"])
    assert plan["projector"] is not None
    assert all(f["path"] != plan["projector"]["path"] for f in plan["files"])


# ---------------------------------------------------------------------------
# Deleting
# ---------------------------------------------------------------------------

def test_deleting_removes_the_file(library, tmp_path: Path, monkeypatch):
    removed: list[tuple[str, bool]] = []

    def fake_remove(path: Path, permanent: bool = False):
        removed.append((str(path), permanent))
        return {"path": str(path), "size_bytes": 1,
                "method": "deleted permanently" if permanent
                else "moved to the Recycle Bin"}

    monkeypatch.setattr(file_removal, "remove", fake_remove)
    model = library.all_models()[0]
    result = library.delete_model(model["id"])
    assert result["deleted"] is True
    assert removed == [(model["path"], False)]
    assert result["freed_bytes"] > 0


def test_a_permanent_delete_is_only_permanent_when_asked(
        library, monkeypatch):
    removed: list[tuple[str, bool]] = []

    def fake_remove(path: Path, permanent: bool = False):
        removed.append((str(path), permanent))
        return {"path": str(path), "size_bytes": 1, "method": "x"}

    monkeypatch.setattr(file_removal, "remove", fake_remove)
    library.delete_model(library.all_models()[0]["id"], permanent=True)
    assert removed[0][1] is True


def test_a_loaded_model_cannot_be_deleted(library, monkeypatch):
    called: list[str] = []

    def fake_remove(path: Path, permanent: bool = False):
        called.append(str(path))
        return {"path": str(path), "size_bytes": 1, "method": "x"}

    monkeypatch.setattr(file_removal, "remove", fake_remove)
    model = library.all_models()[0]
    library._get_instances = lambda: [{
        "model_id": model["id"], "state": "ready", "instance_id": "i1"}]
    with pytest.raises(RuntimeError) as exc:
        library.delete_model(model["id"])
    assert "loaded" in str(exc.value)
    assert not called


def test_the_model_is_gone_from_the_library_afterwards(
        library, tmp_path: Path, monkeypatch):
    def fake_remove(path: Path, permanent: bool = False):
        Path(path).unlink(missing_ok=True)
        return {"path": str(path), "size_bytes": 1, "method": "x"}

    monkeypatch.setattr(file_removal, "remove", fake_remove)
    model = library.all_models()[0]
    library.delete_model(model["id"])
    assert not any(m["id"] == model["id"] for m in library.all_models())


# ---------------------------------------------------------------------------
# Over HTTP
# ---------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    _model(model_dir / "Alpha-Q4_K_M.gguf", "Alpha")
    store = SettingsStore(path=tmp_path / "settings.json")
    store.replace(Settings(model_dirs=[str(model_dir)],
                            converted_dir=str(tmp_path / "converted")
                            ).model_dump())
    app = create_app(store)
    with TestClient(app) as test_client:
        yield test_client


def test_the_delete_endpoint_needs_a_confirmation(client: TestClient,
                                                  monkeypatch):
    called: list[str] = []
    monkeypatch.setattr(file_removal, "remove",
                        lambda p, permanent=False: called.append(str(p)))
    model = client.get("/api/library/models").json()["models"][0]
    res = client.request("DELETE", f"/api/library/models/{model['id']}",
                         json={})
    assert res.status_code == 400
    assert res.json()["error"]["type"] == "confirmation_required"
    assert not called


def test_the_delete_endpoint_works_with_a_confirmation(client: TestClient,
                                                       monkeypatch):
    calls: list[tuple[str, bool]] = []

    def fake_remove(path: Path, permanent: bool = False):
        calls.append((str(path), permanent))
        return {"path": str(path), "size_bytes": 10, "method": "recycled"}

    monkeypatch.setattr(file_removal, "remove", fake_remove)
    model = client.get("/api/library/models").json()["models"][0]
    res = client.request("DELETE", f"/api/library/models/{model['id']}",
                         json={"confirm": True})
    assert res.status_code == 200
    body = res.json()
    assert body["deleted"] is True
    assert body["freed_bytes"] > 0
    assert calls and calls[0][1] is False      # the Recycle Bin by default


def test_the_confirmation_can_travel_in_the_query_string(client: TestClient,
                                                          monkeypatch):
    """Some HTTP stacks drop the body of a DELETE."""
    monkeypatch.setattr(file_removal, "remove",
                        lambda p, permanent=False:
                        {"path": str(p), "size_bytes": 1, "method": "recycled"})
    model = client.get("/api/library/models").json()["models"][0]
    res = client.request("DELETE",
                         f"/api/library/models/{model['id']}?confirm=true")
    assert res.status_code == 200


def test_the_delete_plan_endpoint(client: TestClient):
    model = client.get("/api/library/models").json()["models"][0]
    res = client.get(f"/api/library/models/{model['id']}/delete-plan")
    assert res.status_code == 200
    plan = res.json()
    assert plan["files"][0]["path"] == model["path"]
    assert plan["total_bytes"] > 0


def test_deleting_an_unknown_model_is_a_404(client: TestClient):
    res = client.request("DELETE", "/api/library/models/nope",
                         json={"confirm": True})
    assert res.status_code == 404


# ---------------------------------------------------------------------------
# The Recycle Bin
# ---------------------------------------------------------------------------

def test_the_recycle_bin_is_used_by_default(tmp_path: Path):
    """A deleted model has to be recoverable."""
    target = tmp_path / "a-model.gguf"
    target.write_bytes(b"x" * 32)
    result = file_removal.remove(target)
    assert "Recycle Bin" in result["method"]
    assert not target.exists()


def test_a_permanent_delete_erases_the_file(tmp_path: Path):
    target = tmp_path / "a-model.gguf"
    target.write_bytes(b"x" * 32)
    result = file_removal.remove(target, permanent=True)
    assert "permanently" in result["method"]
    assert not target.exists()


def test_a_missing_file_is_reported(tmp_path: Path):
    with pytest.raises(file_removal.RemovalError):
        file_removal.remove(tmp_path / "not-there.gguf")


def test_a_file_too_big_for_the_bin_is_reported_as_permanent(
        tmp_path: Path, monkeypatch):
    """Windows erases oversized files for good - the user has to know."""
    target = tmp_path / "huge.gguf"
    target.write_bytes(b"x" * 1024)
    monkeypatch.setattr(file_removal, "_recycle_bin_limit_bytes",
                        lambda p: 512)          # less than the file
    result = file_removal.remove(target)
    assert result["recoverable"] is False
    assert "permanently" in result["method"]
    assert "Recycle Bin" in result["reason"]


def test_a_file_that_fits_is_reported_as_recoverable(tmp_path: Path,
                                                    monkeypatch):
    target = tmp_path / "small.gguf"
    target.write_bytes(b"x" * 1024)
    monkeypatch.setattr(file_removal, "_recycle_bin_limit_bytes",
                        lambda p: 1024 ** 3)
    result = file_removal.remove(target)
    assert result["recoverable"] is True
    assert "Recycle Bin" in result["method"]


def test_the_plan_says_whether_the_file_can_be_put_back(library, tmp_path: Path,
                                                         monkeypatch):
    monkeypatch.setattr(file_removal, "will_be_recoverable",
                        lambda p: (False, "it is too big"))
    plan = library.delete_plan(library.all_models()[0]["id"])
    assert plan["recoverable"] is False
    assert "too big" in plan["recoverable_reason"]
