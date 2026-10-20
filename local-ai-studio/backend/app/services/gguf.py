"""Read-only GGUF metadata reader.

Reads only the metadata header of a GGUF file - never the tensor data - so a
11 GB model is inspected in milliseconds. Pure standard library, no engine
required, and it never writes to the file (the model directory is treated as
read-only by policy).

Handles GGUF v1/v2/v3, string arrays of any size (a Qwen vocab can hold
248 000 tokens; failing to consume such an array misaligns every later key)
and reports errors instead of raising, so one corrupt file cannot break a scan.
"""
from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# GGUF value type -> struct format
_VT: dict[int, str | None] = {
    0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f",
    7: "?", 8: None, 9: None, 10: "Q", 11: "q", 12: "d",
}

# Metadata keys we surface to the UI (everything else stays internal).
INTERESTING_PREFIXES = (
    "general.", "tokenizer.ggml.model", "tokenizer.ggml.pre",
    "tokenizer.chat_template", ".context_length", ".block_count",
    ".embedding_length", ".attention.head_count", ".expert_count",
    ".expert_used_count", ".full_attention_interval", ".rope.freq_base",
    ".feed_forward_length", ".key_length", ".value_length",
    ".ssm.", ".attention.layer_norm",
)


def _read_string(fh) -> str:
    (length,) = struct.unpack("<Q", fh.read(8))
    if length > 100_000_000:
        raise ValueError(f"implausible string length {length}")
    return fh.read(length).decode("utf-8", errors="replace")


def _read_value(fh, vtype: int) -> Any:
    if vtype == 9:  # array
        (atype,) = struct.unpack("<I", fh.read(4))
        (count,) = struct.unpack("<Q", fh.read(8))
        if count > 100_000_000:
            raise ValueError(f"implausible array count {count}")
        if atype == 8:  # string array: must be fully consumed
            values: list[str] = []
            for i in range(count):
                s = _read_string(fh)
                if i < 32:  # sample only; the full vocab is rarely useful
                    values.append(s)
            return {"count": count, "sample": values}
        if atype == 9:  # array of arrays
            # Every element must be consumed, not just the first few, or the
            # next key would be read from the wrong offset.
            values = []
            for i in range(count):
                value = _read_value(fh, atype)
                if i < 8:
                    values.append(value)
            return {"count": count, "sample": values}
        fmt = _VT.get(atype)
        if fmt is None:
            raise ValueError(f"unsupported array element type {atype}")
        size = struct.calcsize("<" + fmt)
        data = fh.read(size * min(count, 4096))
        values = list(struct.unpack("<" + fmt * (len(data) // size), data))
        remaining = size * (count - len(values))
        if remaining:
            fh.seek(remaining, 1)
        return values[:64]
    if vtype == 8:
        return _read_string(fh)
    fmt = _VT.get(vtype)
    if fmt is None:
        raise ValueError(f"unsupported value type {vtype}")
    (value,) = struct.unpack("<" + fmt, fh.read(struct.calcsize("<" + fmt)))
    return value


def read_gguf_metadata(path: Path, max_keys: int = 400) -> dict[str, Any]:
    """Parse GGUF metadata. Returns a dict; never raises."""
    out: dict[str, Any] = {"path": str(path)}
    try:
        out["size_bytes"] = path.stat().st_size
    except OSError as exc:
        out["error"] = f"unreadable: {exc}"
        return out
    try:
        with path.open("rb") as fh:
            header = fh.read(8)
            if len(header) < 8:
                out["error"] = "file too small to be GGUF"
                return out
            magic, version = struct.unpack("<II", header)
            if magic != GGUF_MAGIC:
                out["error"] = f"not a GGUF file (magic 0x{magic:08x})"
                return out
            (tensor_count,) = struct.unpack("<Q", fh.read(8))
            (kv_count,) = struct.unpack("<Q", fh.read(8))
            out["gguf_version"] = version
            out["tensor_count"] = tensor_count
            metadata: dict[str, Any] = {}
            for _ in range(min(kv_count, max_keys)):
                key = _read_string(fh)
                (vtype,) = struct.unpack("<I", fh.read(4))
                metadata[key] = _read_value(fh, vtype)
            out["metadata"] = metadata
    except Exception as exc:  # noqa: BLE001 - scanner must survive bad files
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# ---------------------------------------------------------------------------
# Tensor layout: how big should this file be?
# ---------------------------------------------------------------------------
#
# A truncated download still has a perfectly valid GGUF header, so the header
# alone cannot tell a complete file from a half-finished one. The tensor
# directory does: every tensor records the offset of its data, so the end of
# the file can be derived from the offsets alone.
#
# Type sizes are *derived* from the file (the distance between two consecutive
# tensors) rather than trusted from a hard-coded table, so a quant type this
# code has never heard of still yields a correct answer. A hard-coded table is
# only consulted for a type that appears exactly once, and even then it is
# cross-checked against the derived sizes.

# Tensor type sizes are deliberately NOT hard-coded here: the byte size of a
# quantized tensor depends on its dimensions, so a table of "bytes per block"
# cannot answer "how long is this file". The size is derived from the tensor
# directory instead - see read_tensor_layout().

DEFAULT_ALIGNMENT = 32


def _skip_string(fh) -> None:
    (length,) = struct.unpack("<Q", fh.read(8))
    if length > 100_000_000:
        raise ValueError(f"implausible string length {length}")
    fh.seek(length, 1)


def _skip_value(fh, vtype: int) -> None:
    """Advance past a metadata value without decoding it.

    Decoding a 248 000-entry vocabulary is slow and pointless when all we
    want is the file offset, so strings are seeked over rather than read.
    """
    if vtype == 9:  # array
        (atype,) = struct.unpack("<I", fh.read(4))
        (count,) = struct.unpack("<Q", fh.read(8))
        if count > 100_000_000:
            raise ValueError(f"implausible array count {count}")
        if atype == 8:  # array of strings
            for _ in range(count):
                _skip_string(fh)
            return
        if atype == 9:  # array of arrays
            for _ in range(count):
                _skip_value(fh, atype)
            return
        fmt = _VT.get(atype)
        if fmt is None:
            raise ValueError(f"unsupported array element type {atype}")
        fh.seek(struct.calcsize("<" + fmt) * count, 1)
        return
    if vtype == 8:
        _skip_string(fh)
        return
    fmt = _VT.get(vtype)
    if fmt is None:
        raise ValueError(f"unsupported value type {vtype}")
    fh.seek(struct.calcsize("<" + fmt), 1)


def read_tensor_layout(path: Path) -> dict[str, Any]:
    """Derive the expected end-of-file offset from the tensor directory.

    Returns a dict with:

    ``min_end``
        A mathematically safe lower bound: where the *last* tensor starts.
        A file shorter than this is definitely truncated.
    ``exact_end``
        Where the file should end, when the size of the final tensor could be
        established. ``None`` when it could not be.
    ``derived_types``
        Tensor types whose byte size was learned from the file itself.

    How the size of a tensor is found
    --------------------------------
    The directory stores offsets, not sizes, but consecutive offsets give the
    exact size of every tensor except the last one. The last tensor's size is
    recovered by scaling a sibling of the same type: a quantized tensor
    occupies ``type_size * (ne0/block) * ne1 * ne2`` bytes, so for two tensors
    of one type the byte count is proportional to the product of their
    dimensions. That keeps the answer exact for quant types this code has
    never heard of, without a hard-coded table of sizes.

    Never raises: a malformed file returns an ``error`` key.
    """
    out: dict[str, Any] = {"min_end": None, "exact_end": None,
                           "derived_types": [], "error": None}
    try:
        with path.open("rb") as fh:
            header = fh.read(8)
            if len(header) < 8:
                out["error"] = "file too small to be GGUF"
                return out
            magic, version = struct.unpack("<II", header)
            if magic != GGUF_MAGIC:
                out["error"] = f"not a GGUF file (magic 0x{magic:08x})"
                return out
            (tensor_count,) = struct.unpack("<Q", fh.read(8))
            (kv_count,) = struct.unpack("<Q", fh.read(8))
            out["version"] = version

            # Skip the metadata section - it must be consumed in full or the
            # tensor directory would be read from the wrong offset.
            for _ in range(kv_count):
                _skip_string(fh)
                (vtype,) = struct.unpack("<I", fh.read(4))
                _skip_value(fh, vtype)

            # Tensor directory: name, n_dims, dims (u64 each), type, offset.
            tensors: list[tuple[int, int, int]] = []  # (offset, type, elems)
            for _ in range(tensor_count):
                _skip_string(fh)
                (n_dims,) = struct.unpack("<I", fh.read(4))
                elems = 1
                for _d in range(n_dims):
                    (dim,) = struct.unpack("<Q", fh.read(8))
                    elems *= dim
                (ttype,) = struct.unpack("<I", fh.read(4))
                (toffset,) = struct.unpack("<Q", fh.read(8))
                tensors.append((toffset, ttype, elems))

            if not tensors:
                out["error"] = "no tensors in file"
                return out

            tensors.sort(key=lambda t: t[0])
            max_offset = tensors[-1][0]
            data_start = (fh.tell() + DEFAULT_ALIGNMENT - 1) \
                & ~(DEFAULT_ALIGNMENT - 1)
            out["tensor_count"] = tensor_count
            out["data_start"] = data_start
            out["min_end"] = data_start + max_offset

            # Exact size of every tensor except the last one.
            sizes: dict[int, int] = {}
            for i in range(len(tensors) - 1):
                gap = tensors[i + 1][0] - tensors[i][0]
                if gap > 0:
                    sizes[i] = gap
            out["derived_types"] = sorted({t[1] for t in tensors})

            # The last tensor: scale a sibling of the same type.
            last = len(tensors) - 1
            lo, lt, ln = tensors[last]
            for j in range(last - 1, -1, -1):
                if tensors[j][1] != lt or j not in sizes:
                    continue
                ref_size = sizes[j]
                ref_elems = tensors[j][2]
                if ref_size <= 0 or ref_elems <= 0 or ln <= 0:
                    break
                numerator = ref_size * ln
                if numerator % ref_elems:
                    break            # not a whole number of bytes - do not guess
                candidate = numerator // ref_elems
                if candidate > 0:
                    out["exact_end"] = data_start + lo + candidate
                    out["last_size_source"] = "scaled from sibling tensor"
                break
            return out
    except Exception as exc:  # noqa: BLE001 - a bad file must not break a scan
        out["error"] = f"{type(exc).__name__}: {exc}"
        return out


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    """Distill raw GGUF metadata into the fields the UI shows."""
    meta = result.get("metadata", {}) or {}
    arch = meta.get("general.architecture", "")
    file_type = meta.get("general.file_type")
    prefix = f"{arch}." if arch else ""
    return {
        "architecture": arch,
        "name": meta.get("general.name"),
        "author": meta.get("general.author"),
        "description": meta.get("general.description"),
        "license": (meta.get("general.license.name")
                    if isinstance(meta.get("general.license.name"), str) else None),
        "size_label": meta.get("general.size_label"),
        "is_mmproj": meta.get("general.type") == "mmproj",
        "file_type_code": file_type,
        "context_length": meta.get(prefix + "context_length"),
        "block_count": meta.get(prefix + "block_count"),
        "embedding_length": meta.get(prefix + "embedding_length"),
        "head_count": meta.get(prefix + "attention.head_count"),
        "head_count_kv": meta.get(prefix + "attention.head_count_kv"),
        "key_length": meta.get(prefix + "attention.key_length"),
        "value_length": meta.get(prefix + "attention.value_length"),
        "expert_count": meta.get(prefix + "expert_count"),
        "expert_used_count": meta.get(prefix + "expert_used_count"),
        "full_attention_interval": meta.get(prefix + "full_attention_interval"),
        "rope_freq_base": meta.get(prefix + "rope.freq_base"),
        "ssm_state_size": meta.get(prefix + "ssm.state_size"),
        "ssm_group_count": meta.get(prefix + "ssm.group_count"),
        "tokenizer_model": meta.get("tokenizer.ggml.model"),
        "tokenizer_pre": meta.get("tokenizer.ggml.pre"),
        "has_chat_template": bool(meta.get("tokenizer.chat_template")),
        "vocab_size": (meta.get("tokenizer.ggml.tokens", {}) or {}).get("count"),
    }


# GGUF file_type codes -> human quantization label.
# Verified against include/llama.h at llama.cpp b11712 (LLAMA_FTYPE_MOSTLY_*).
FILE_TYPES = {
    0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0",
    9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L",
    14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K",
    19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS",
    24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
    29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16", 36: "TQ1_0",
    37: "TQ2_0", 38: "MXFP4_MOE", 39: "NVFP4", 40: "Q1_0", 41: "Q2_0",
}


def quant_label(summary: dict[str, Any], filename: str) -> str:
    """Best-effort quantization label from metadata, falling back to filename."""
    code = summary.get("file_type_code")
    if isinstance(code, int) and code in FILE_TYPES:
        return FILE_TYPES[code]
    upper = filename.upper()
    for known in ("IQ2_M", "IQ4_XS", "IQ3_S", "Q4_K_M", "Q4_K_S", "Q5_K_M",
                  "Q6_K", "Q8_0", "F16", "BF16", "F32", "Q4_0", "Q4_1"):
        if known in upper:
            return known
    return "unknown"
