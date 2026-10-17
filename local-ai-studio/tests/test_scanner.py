"""Model scanner tests: discovery, grouping, compatibility verdicts."""
from __future__ import annotations

import json
from pathlib import Path

from backend.app.services.model_scanner import ModelScanner, scan_directory
from tests.conftest import write_gguf


def _model(path: Path, arch: str = "qwen35moe", name: str = "M") -> Path:
    return write_gguf(path, {
        "general.architecture": arch,
        "general.name": name,
        "general.file_type": 15,
        f"{arch}.block_count": 40,
    })


def test_discovers_gguf_and_pairs_mmproj(tmp_path: Path):
    model_dir = tmp_path / "Org" / "MyModel"
    model_dir.mkdir(parents=True)
    _model(model_dir / "MyModel-Q4_K_M.gguf", name="MyModel")
    _model(model_dir / "mmproj-MyModel-f16.gguf", arch="clip", name="proj")

    records = scan_directory(tmp_path)
    mains = [r for r in records if not r.get("is_mmproj")]
    assert len(mains) == 1
    assert mains[0]["format"] == "GGUF"
    assert mains[0]["mmproj_path"] and mains[0]["mmproj_path"].endswith(
        "mmproj-MyModel-f16.gguf")
    assert mains[0]["capabilities"]["vision"] is True


def test_unknown_architecture_is_unverified_not_supported(tmp_path: Path):
    model_dir = tmp_path / "future"
    model_dir.mkdir()
    _model(model_dir / "weird.gguf", arch="brand_new_arch")
    records = [r for r in scan_directory(tmp_path) if not r.get("is_mmproj")]
    assert records[0]["compatibility"]["status"] == "unverified"
    assert records[0]["compatibility"]["supported_by_engine"] is False


def test_known_architecture_is_supported(tmp_path: Path):
    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    _model(model_dir / "ok.gguf", arch="qwen35moe")
    records = [r for r in scan_directory(tmp_path) if not r.get("is_mmproj")]
    assert records[0]["compatibility"]["status"] == "supported"
    assert records[0]["capabilities"]["moe"] is True


def test_huggingface_directory_is_reported_convertible(tmp_path: Path):
    hf = tmp_path / "hf-model"
    hf.mkdir()
    (hf / "config.json").write_text(json.dumps({"model_type": "llama"}))
    (hf / "model.safetensors").write_bytes(b"x" * 1024)
    records = scan_directory(tmp_path)
    assert records[0]["format"] == "safetensors"
    assert records[0]["compatibility"]["status"] == "convertible"


def test_split_shards_group_into_one_model(tmp_path: Path):
    d = tmp_path / "big"
    d.mkdir()
    _model(d / "Big-00001-of-00002.gguf")
    _model(d / "Big-00002-of-00002.gguf")
    records = [r for r in scan_directory(tmp_path) if not r.get("is_mmproj")]
    assert len(records) == 1
    assert len(records[0]["files"]) == 2


def test_inaccessible_directory_is_reported_not_raised(tmp_path: Path):
    records = scan_directory(tmp_path / "does-not-exist")
    assert records and "error" in records[0]


def test_scanner_caches_and_resolves_aliases(tmp_path: Path):
    model_dir = tmp_path / "alias"
    model_dir.mkdir()
    _model(model_dir / "CoolModel.gguf", name="Cool Model")

    scanner = ModelScanner(get_dirs=lambda: [str(tmp_path)])
    models = scanner.scan()
    assert len(models) == 1
    assert scanner.get(models[0]["id"]) is not None
    assert scanner.find_by_alias_or_name("Cool Model") is not None
    assert scanner.find_by_alias_or_name("coolmodel") is not None
    assert scanner.find_by_alias_or_name("nope") is None
