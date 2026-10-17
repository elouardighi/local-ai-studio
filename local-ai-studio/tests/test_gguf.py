"""GGUF metadata reader tests."""
from __future__ import annotations

import struct
from pathlib import Path

from backend.app.services import gguf
from tests.conftest import write_gguf


def test_reads_basic_metadata(tmp_path: Path):
    path = write_gguf(tmp_path / "m.gguf", {
        "general.architecture": "qwen35moe",
        "general.file_type": 15,
        "qwen35moe.block_count": 40,
    })
    result = gguf.read_gguf_metadata(path)
    assert "error" not in result, result
    assert result["metadata"]["general.architecture"] == "qwen35moe"
    assert result["metadata"]["general.file_type"] == 15
    assert result["gguf_version"] == 3


def test_large_string_array_does_not_misalign(tmp_path: Path):
    """A 248k-token vocab once broke naive parsers by not consuming the array."""
    vocab = [f"tok{i}" for i in range(5000)]
    path = write_gguf(tmp_path / "vocab.gguf", {
        "general.architecture": "qwen35",
        "tokenizer.ggml.tokens": vocab,
        "qwen35.block_count": 12,          # key AFTER the big array
        "general.name": "tail",            # and another after that
    })
    result = gguf.read_gguf_metadata(path)
    assert "error" not in result, result
    assert result["metadata"]["qwen35.block_count"] == 12
    assert result["metadata"]["general.name"] == "tail"
    assert result["metadata"]["tokenizer.ggml.tokens"]["count"] == 5000


def test_rejects_non_gguf_file(tmp_path: Path):
    path = tmp_path / "not.gguf"
    path.write_bytes(b"PK\x03\x04 definitely a zip, not gguf")
    result = gguf.read_gguf_metadata(path)
    assert "error" in result


def test_missing_file_reports_error(tmp_path: Path):
    result = gguf.read_gguf_metadata(tmp_path / "missing.gguf")
    assert "error" in result


def test_summarize_extracts_ui_fields(gguf_model: Path):
    summary = gguf.summarize(gguf.read_gguf_metadata(gguf_model))
    assert summary["architecture"] == "qwen35moe"
    assert summary["block_count"] == 40
    assert summary["expert_count"] == 256
    assert summary["full_attention_interval"] == 4
    assert summary["is_mmproj"] is False


def test_quant_label_from_metadata_and_filename():
    assert gguf.quant_label({"file_type_code": 29}, "x.gguf") == "IQ2_M"
    assert gguf.quant_label({"file_type_code": 15}, "x.gguf") == "Q4_K_M"
    # Unknown code falls back to the filename.
    assert gguf.quant_label({}, "Model-IQ2_M.gguf") == "IQ2_M"
    assert gguf.quant_label({}, "mystery.gguf") == "unknown"


def test_truncated_file_does_not_raise(tmp_path: Path):
    path = write_gguf(tmp_path / "cut.gguf", {"general.architecture": "llama"})
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 2])
    result = gguf.read_gguf_metadata(path)
    assert "error" in result  # reported, not raised
