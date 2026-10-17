"""Read-only GGUF metadata prober.

Parses only the metadata header of a GGUF file (never the weights) and prints
key/value pairs such as general.architecture, general.name and quantization.
Pure stdlib so it works before any engine is installed.
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

GGUF_MAGIC = 0x46554747  # "GGUF" little-endian

# value types -> (struct format, name)
VT = {
    0: ("B", "uint8"),
    1: ("b", "int8"),
    2: ("H", "uint16"),
    3: ("h", "int16"),
    4: ("I", "uint32"),
    5: ("i", "int32"),
    6: ("f", "float32"),
    7: ("?", "bool"),
    8: (None, "string"),
    9: (None, "array"),
    10: ("Q", "uint64"),
    11: ("q", "int64"),
    12: ("d", "float64"),
}


def read_string(fh) -> str:
    (length,) = struct.unpack("<Q", fh.read(8))
    if length > 1_000_000:
        raise ValueError(f"implausible string length {length}")
    return fh.read(length).decode("utf-8", errors="replace")


def read_value(fh, vtype: int, depth: int = 0):
    if vtype == 9:  # array
        (atype,) = struct.unpack("<I", fh.read(4))
        (count,) = struct.unpack("<Q", fh.read(8))
        if count > 10_000_000:
            raise ValueError(f"implausible array count {count}")
        fmt, _ = VT.get(atype, (None, "?"))
        if atype == 8:
            # string arrays (e.g. a 248K-token vocab) must be fully consumed or
            # every following key is misaligned; sample the first entries only.
            values = []
            for i in range(count):
                s = read_string(fh)
                if i < 32:
                    values.append(s)
            return {"count": count, "sample": values}
        if atype == 9:
            return [read_value(fh, atype, depth + 1) for _ in range(min(count, 64))]
        if fmt is None:
            raise ValueError(f"unsupported array type {atype}")
        size = struct.calcsize("<" + fmt)
        data = fh.read(size * min(count, 4096))
        values = list(struct.unpack("<" + fmt * (len(data) // size), data))
        skip = size * (count - len(values))
        if skip:
            fh.seek(skip, 1)
        return values[:64]
    if vtype == 8:
        return read_string(fh)
    fmt = VT.get(vtype, (None, "?"))[0]
    if fmt is None:
        return f"<unsupported type {vtype}>"
    (value,) = struct.unpack("<" + fmt, fh.read(struct.calcsize("<" + fmt)))
    return value


def probe(path: Path, max_keys: int = 200) -> dict:
    out: dict = {"path": str(path), "size_bytes": path.stat().st_size}
    try:
        with path.open("rb") as fh:
            magic, version = struct.unpack("<II", fh.read(8))
            if magic != GGUF_MAGIC:
                out["error"] = f"not a GGUF file (magic=0x{magic:08x})"
                return out
            (tensor_count,) = struct.unpack("<Q", fh.read(8))
            (kv_count,) = struct.unpack("<Q", fh.read(8))
            out["gguf_version"] = version
            out["tensor_count"] = tensor_count
            out["kv_count"] = kv_count
            kv: dict = {}
            for _ in range(min(kv_count, max_keys)):
                key = read_string(fh)
                (vtype,) = struct.unpack("<I", fh.read(4))
                kv[key] = read_value(fh, vtype)
            out["metadata"] = kv
    except Exception as exc:  # noqa: BLE001 - report, never crash the scanner
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    roots = sys.argv[1:]
    results = []
    for root in roots:
        p = Path(root)
        files = [p] if p.is_file() else sorted(p.rglob("*.gguf"))
        for f in files:
            results.append(probe(f))
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
