"""The OpenCode config stays in step with the library.

OpenCode reads its model list from its own config file, so a model that is
downloaded has to be added there to be usable, and a model that is deleted
has to be taken out again.

These tests use a small stand-in config, and check that the surgery leaves
everything else alone.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.config import Settings
from backend.app.services import opencode_sync
from backend.app.services.opencode_sync import OpenCodeConfig

CONFIG = """{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "rea": { "type": "local", "command": ["node.exe", "rea.mjs"] }
  },
  "provider": {
    "bionic": {
      "name": "bionic",
      "options": { "baseURL": "http://localhost:1234/v1" },
      "models": {
        "prism-ml/bonsai-27b": { "name": "prism-ml/bonsai-27b" }
      }
    },
    "localai": {
      "name": "Local AI Studio",
      "options": { "baseURL": "http://127.0.0.1:1234/v1" },
      "models": {
        "Bonsai-27B": {
          "name": "Bonsai 27B",
          "limit": { "context": 150000, "output": 20000 }
        }
      }
    }
  }
}
"""


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    path = tmp_path / "opencode.jsonc"
    path.write_text(CONFIG, encoding="utf-8")
    return path


@pytest.fixture
def config(config_path: Path) -> OpenCodeConfig:
    return OpenCodeConfig(config_path)


# ---------------------------------------------------------------------------
# Adding
# ---------------------------------------------------------------------------

def test_a_model_is_added(config: OpenCodeConfig):
    assert config.add_model("New-Model", {
        "name": "New-Model", "limit": {"context": 2048, "output": 1024}})
    models = config.provider_models()
    assert "New-Model" in models
    assert models["New-Model"]["limit"] == {"context": 2048, "output": 1024}


def test_an_existing_model_is_left_alone(config: OpenCodeConfig):
    assert config.add_model("Bonsai-27B", {
        "name": "Bonsai 27B", "limit": {"context": 1, "output": 1}}) is False
    models = config.provider_models()
    assert models["Bonsai-27B"]["limit"]["context"] == 150000


def test_the_rest_of_the_file_is_untouched(config: OpenCodeConfig,
                                            config_path: Path):
    config.add_model("New-Model", {"name": "New-Model",
                                   "limit": {"context": 2048, "output": 1024}})
    parsed = json.loads(opencode_sync.strip_jsonc(config_path.read_text("utf-8")))
    assert parsed["mcp"]["rea"]["command"] == ["node.exe", "rea.mjs"]
    assert parsed["provider"]["bionic"]["models"] == {
        "prism-ml/bonsai-27b": {"name": "prism-ml/bonsai-27b"}}
    assert parsed["provider"]["localai"]["options"]["baseURL"] == \
        "http://127.0.0.1:1234/v1"
    assert parsed["$schema"] == "https://opencode.ai/config.json"


def test_extra_keys_in_an_entry_survive(config: OpenCodeConfig,
                                        config_path: Path):
    config.add_model("Bonsai-27B", {
        "name": "Bonsai 27B", "limit": {"context": 1, "output": 1},
        "reasoning": True}, replace=True)
    models = config.provider_models()
    assert models["Bonsai-27B"].get("reasoning") is True


# ---------------------------------------------------------------------------
# Removing
# ---------------------------------------------------------------------------

def test_a_model_is_removed(config: OpenCodeConfig):
    assert config.remove_model("Bonsai-27B") is True
    assert "Bonsai-27B" not in config.provider_models()


def test_removing_a_model_that_is_not_there_is_not_an_error(
        config: OpenCodeConfig):
    assert config.remove_model("Not-Listed") is False


def test_removing_leaves_the_other_provider_alone(config: OpenCodeConfig,
                                                  config_path: Path):
    config.remove_model("Bonsai-27B")
    parsed = json.loads(opencode_sync.strip_jsonc(config_path.read_text("utf-8")))
    assert parsed["provider"]["bionic"]["models"] == {
        "prism-ml/bonsai-27b": {"name": "prism-ml/bonsai-27b"}}


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------

def test_only_the_limits_are_changed(config: OpenCodeConfig):
    assert config.set_limit("Bonsai-27B", 43008, 8192) is True
    models = config.provider_models()
    assert models["Bonsai-27B"]["limit"] == {"context": 43008, "output": 8192}
    assert models["Bonsai-27B"]["name"] == "Bonsai 27B"


# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

def test_a_config_that_would_not_parse_is_not_written(config: OpenCodeConfig,
                                                       config_path: Path,
                                                       monkeypatch):
    before = config_path.read_text(encoding="utf-8")

    def broken(specs):
        return specs

    monkeypatch.setattr(opencode_sync, "render_models",
                        lambda models, indent: "{ this is not json")
    with pytest.raises(opencode_sync.ConfigError):
        config.add_model("New-Model", {"name": "New-Model"})
    assert config_path.read_text(encoding="utf-8") == before


def test_the_status_says_what_is_missing(config: OpenCodeConfig):
    status = config.status()
    assert status["exists"] is True
    assert status["count"] == 1
    assert "Bonsai-27B" in status["models"]


def test_finding_a_config(tmp_path: Path, monkeypatch):
    path = tmp_path / "opencode.jsonc"
    path.write_text(CONFIG, encoding="utf-8")
    assert opencode_sync.find_config(str(path)) == path
    assert opencode_sync.find_config(str(tmp_path / "nope.jsonc")) is None


# ---------------------------------------------------------------------------
# The library drives it
# ---------------------------------------------------------------------------

def test_the_library_adds_and_removes(tmp_path: Path, monkeypatch):
    """A download adds the model; a delete takes it out again."""
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_library import ModelLibrary
    from backend.app.services.model_scanner import ModelScanner
    from tests.conftest import write_gguf

    config_file = tmp_path / "opencode.jsonc"
    config_file.write_text(CONFIG, encoding="utf-8")

    models = tmp_path / "models"
    models.mkdir()
    write_gguf(models / "Alpha-Q4_K_M.gguf", {
        "general.architecture": "qwen35moe", "general.name": "Alpha",
        "general.file_type": 15, "qwen35moe.block_count": 40,
    })

    settings = Settings(model_dirs=[str(models)],
                        auto_sync_opencode=True,
                        opencode_config_path=str(config_file))
    scanner = ModelScanner(get_dirs=lambda: settings.model_dirs, log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings,
                       get_instances=lambda: [])
    lib.rescan()

    # The library adds its models to the config.
    lib.sync_opencode()
    listed = OpenCodeConfig(config_file).provider_models()
    assert "Alpha" in listed
    assert listed["Alpha"]["limit"]["context"] > 0

    # And a deleted model is taken out again.
    from backend.app.services import file_removal
    monkeypatch.setattr(file_removal, "remove",
                        lambda p, permanent=False:
                        {"path": str(p), "size_bytes": 1, "method": "x"})
    model = next(m for m in lib.all_models() if m["name"] == "Alpha")
    lib.delete_model(model["id"])
    assert "Alpha" not in OpenCodeConfig(config_file).provider_models()
    # The other provider is untouched.
    parsed = json.loads(opencode_sync.strip_jsonc(config_file.read_text("utf-8")))
    assert parsed["provider"]["bionic"]["models"] == {
        "prism-ml/bonsai-27b": {"name": "prism-ml/bonsai-27b"}}


def test_the_sync_can_be_switched_off(tmp_path: Path):
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_library import ModelLibrary
    from backend.app.services.model_scanner import ModelScanner

    settings = Settings(model_dirs=[str(tmp_path)],
                        auto_sync_opencode=False)
    scanner = ModelScanner(get_dirs=lambda: settings.model_dirs, log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings)
    assert lib.opencode_config() is None
    assert lib.sync_opencode()["synced"] is False


def _remove_for_real(path: Path, permanent: bool = False) -> dict:
    """Stand in for the file removal, but actually take the file away."""
    Path(path).unlink(missing_ok=True)
    return {"path": str(path), "size_bytes": 1, "method": "x"}


def test_a_deleted_model_is_remembered_and_removed(tmp_path: Path,
                                                   monkeypatch):
    """A model that was deleted is taken out of the config, even if the
    delete happened while the sync was switched off."""
    from backend.app.services import file_removal
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_library import ModelLibrary
    from backend.app.services.model_scanner import ModelScanner
    from tests.conftest import write_gguf

    config_file = tmp_path / "opencode.jsonc"
    config_file.write_text(CONFIG, encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    write_gguf(models / "Alpha-Q4_K_M.gguf", {
        "general.architecture": "qwen35moe", "general.name": "Alpha",
        "general.file_type": 15, "qwen35moe.block_count": 40,
    })

    settings = Settings(model_dirs=[str(models)],
                        auto_sync_opencode=True,
                        opencode_config_path=str(config_file))
    scanner = ModelScanner(get_dirs=lambda: settings.model_dirs, log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings,
                       get_instances=lambda: [],
                       state_file=tmp_path / "library.json")
    lib.rescan()
    lib.sync_opencode()
    assert "Alpha" in OpenCodeConfig(config_file).provider_models()

    monkeypatch.setattr(file_removal, "remove",
                        lambda p, permanent=False: _remove_for_real(p))
    model = next(m for m in lib.all_models() if m["name"] == "Alpha")
    lib.delete_model(model["id"])
    assert "Alpha" not in OpenCodeConfig(config_file).provider_models()

    # The deletion is remembered, so a later sync keeps it out.
    lib.sync_opencode()
    assert "Alpha" not in OpenCodeConfig(config_file).provider_models()


def test_a_model_that_is_downloaded_again_comes_back(tmp_path: Path,
                                                      monkeypatch):
    """Deleting a model and then downloading it again must put it back."""
    from backend.app.services import file_removal
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_library import ModelLibrary
    from backend.app.services.model_scanner import ModelScanner
    from tests.conftest import write_gguf

    config_file = tmp_path / "opencode.jsonc"
    config_file.write_text(CONFIG, encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    write_gguf(models / "Alpha-Q4_K_M.gguf", {
        "general.architecture": "qwen35moe", "general.name": "Alpha",
        "general.file_type": 15, "qwen35moe.block_count": 40,
    })

    settings = Settings(model_dirs=[str(models)],
                        auto_sync_opencode=True,
                        opencode_config_path=str(config_file))
    scanner = ModelScanner(get_dirs=lambda: settings.model_dirs, log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings,
                       get_instances=lambda: [],
                       state_file=tmp_path / "library.json")
    lib.rescan()
    lib.sync_opencode()
    assert "Alpha" in OpenCodeConfig(config_file).provider_models()

    # Delete it - the file really goes, so the model leaves the library.
    monkeypatch.setattr(file_removal, "remove", _remove_for_real)
    model = next(m for m in lib.all_models() if m["name"] == "Alpha")
    lib.delete_model(model["id"])
    assert "Alpha" not in OpenCodeConfig(config_file).provider_models()

    # It comes back - a model that is in the library again is not deleted.
    write_gguf(models / "Alpha-Q4_K_M.gguf", {
        "general.architecture": "qwen35moe", "general.name": "Alpha",
        "general.file_type": 15, "qwen35moe.block_count": 40,
    })
    lib.rescan()
    lib.sync_opencode()
    assert "Alpha" in OpenCodeConfig(config_file).provider_models()


def test_a_model_the_user_listed_is_never_removed(tmp_path: Path):
    from backend.app.services.event_bus import EventBus
    from backend.app.services.log_store import LogStore
    from backend.app.services.model_library import ModelLibrary
    from backend.app.services.model_scanner import ModelScanner

    config_file = tmp_path / "opencode.jsonc"
    config_file.write_text(CONFIG, encoding="utf-8")
    settings = Settings(model_dirs=[str(tmp_path)],
                        auto_sync_opencode=True,
                        opencode_config_path=str(config_file))
    scanner = ModelScanner(get_dirs=lambda: settings.model_dirs, log=LogStore())
    lib = ModelLibrary(scanner=scanner, log=LogStore(), events=EventBus(),
                       get_settings=lambda: settings,
                       get_instances=lambda: [])
    lib.sync_opencode()
    # "Bonsai-27B" is in the config but not in the library - it was not
    # deleted by us, so it stays.
    assert "Bonsai-27B" in OpenCodeConfig(config_file).provider_models()
