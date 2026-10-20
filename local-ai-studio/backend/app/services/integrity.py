"""File integrity and duplicate detection for model files.

Three questions, answered from the files themselves:

* **Is this file complete?** A download that stopped half-way still has a
  valid GGUF header, so the header is not enough. The tensor directory gives
  a safe lower bound for the end of the file, and a truncated file is caught
  by comparing the two.
* **Is this file a duplicate of another one?** The same weights downloaded
  twice, in two different folders, waste tens of gigabytes. Files are
  fingerprinted from their size plus their first and last mebibyte, which is
  enough to recognise identical content without reading 11 GB.
* **Is this file just another quantization of a model we already have?** That
  is not a duplicate - it is a variant - but it is worth showing next to the
  other one.

Everything here is read-only. Nothing is written, renamed or deleted, and no
model directory is ever modified.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from . import gguf

# Integrity verdicts.
STATUS_OK = "ok"                 # complete, verified
STATUS_INCOMPLETE = "incomplete"  # truncated: shorter than the tensor data
STATUS_CORRUPT = "corrupt"        # header unreadable / not a model at all
STATUS_UNVERIFIED = "unverified"  # could not be checked (unknown tensor type)

_MIB = 1024 * 1024
_HASH_CHUNK = 4 * 1024 * 1024
# The tensor data starts on a 32-byte boundary, but a file may have been
# written with a different alignment. A few bytes of slack keeps that from
# being reported as a truncated file.
_ALIGNMENT_SLACK = 64


@dataclass
class IntegrityReport:
    """What we know about one model file's completeness."""

    status: str = STATUS_UNVERIFIED
    reason: str | None = None
    size_bytes: int = 0
    expected_min_bytes: int | None = None
    expected_bytes: int | None = None
    signature: str | None = None
    part_files: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Fingerprints
# ---------------------------------------------------------------------------

def quick_signature(path: Path) -> str | None:
    """Content fingerprint that does not read the whole file.

    Hashes the size together with the first and the last mebibyte. Two files
    with the same signature are the same bytes with very high probability; a
    truncated file has a different tail and therefore a different signature.
    """
    try:
        size = path.stat().st_size
        h = hashlib.sha256()
        h.update(str(size).encode("ascii"))
        with path.open("rb") as fh:
            h.update(fh.read(_MIB))
            if size > _MIB:
                fh.seek(-_MIB, os.SEEK_END)
                h.update(fh.read(_MIB))
        return h.hexdigest()
    except OSError:
        return None


def hash_file(path: Path, algorithm: str = "sha256") -> str | None:
    """Full-file hash. Used for download verification, not during a scan."""
    try:
        h = hashlib.new(algorithm)
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(_HASH_CHUNK), b""):
                h.update(block)
        return h.hexdigest()
    except (OSError, ValueError):
        return None


def git_blob_sha1(path: Path) -> str | None:
    """Git blob hash of a file: sha1("blob <size>\\0" + content).

    Hugging Face serves small files as regular git blobs, and the tree API
    reports their object id. Computing the same digest locally lets us verify
    those downloads too, not just the LFS ones.
    """
    try:
        size = path.stat().st_size
        h = hashlib.sha1()
        h.update(f"blob {size}\0".encode("ascii"))
        with path.open("rb") as fh:
            for block in iter(lambda: fh.read(_HASH_CHUNK), b""):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Completeness
# ---------------------------------------------------------------------------

def check_integrity(path: Path,
                    probe: dict[str, Any] | None = None) -> IntegrityReport:
    """Decide whether a model file is complete.

    ``probe`` may be a previously parsed GGUF header (from the scanner) to
    avoid reading the header twice.
    """
    report = IntegrityReport()
    try:
        stat = path.stat()
    except OSError as exc:
        report.status = STATUS_CORRUPT
        report.reason = f"file not accessible: {exc}"
        return report

    report.size_bytes = stat.st_size

    # An interrupted download leaves a ".part" sibling behind. This is the
    # clearest sign of an unfinished transfer, so it is checked first.
    part = path.with_name(path.name + ".part")
    if part.exists():
        try:
            report.part_files.append(str(part))
        except OSError:
            pass

    # The content fingerprint is needed to spot duplicate files, and it is
    # cheap (two mebibytes), so it is always computed.
    report.signature = quick_signature(path)

    if probe is None:
        probe = gguf.read_gguf_metadata(path)
    if probe.get("error"):
        detail = str(probe["error"])
        report.status = STATUS_CORRUPT
        if "magic" in detail:
            report.reason = f"not a GGUF file ({detail})"
        else:
            report.reason = f"GGUF header unreadable: {detail}"
        return report

    layout = gguf.read_tensor_layout(path)
    if layout.get("error"):
        if report.part_files:
            report.status = STATUS_INCOMPLETE
            report.reason = ("an unfinished .part file sits next to this "
                             "model; the download never completed")
        else:
            report.status = STATUS_UNVERIFIED
            report.reason = (f"could not verify completeness "
                             f"({layout['error']})")
        return report

    min_end = layout.get("min_end")
    exact_end = layout.get("exact_end")
    report.expected_min_bytes = min_end
    report.expected_bytes = exact_end

    if min_end and stat.st_size < min_end:
        missing = min_end - stat.st_size
        report.status = STATUS_INCOMPLETE
        report.reason = (
            f"file is {stat.st_size:,} bytes but the tensor data needs at "
            f"least {min_end:,} - {missing:,} bytes are missing "
            f"(interrupted download?)")
        return report

    if exact_end and stat.st_size + _ALIGNMENT_SLACK < exact_end:
        missing = exact_end - stat.st_size
        report.status = STATUS_INCOMPLETE
        report.reason = (
            f"file is {stat.st_size:,} bytes but should be about "
            f"{exact_end:,} - {missing:,} bytes are missing "
            f"(interrupted download?)")
        return report

    if report.part_files:
        report.status = STATUS_INCOMPLETE
        report.reason = ("an unfinished .part file sits next to this model; "
                         "the download never completed")
        return report

    if exact_end is None:
        report.status = STATUS_UNVERIFIED
        report.reason = ("the file parses and the tensor data is present, but "
                         "the end of the file could not be verified "
                         "(unknown tensor type)")
        return report

    report.status = STATUS_OK
    report.reason = None
    return report


# ---------------------------------------------------------------------------
# Duplicates and variants
# ---------------------------------------------------------------------------

_QUANT_TOKEN = re.compile(
    r"(?:^|[-_ .])("
    r"iq\d+(?:_[a-z0-9]+)*|"
    r"q\d+(?:_[a-z0-9]+)*|"
    r"f16|f32|bf16|fp8|fp16|int[248]|"
    r"mxfp4|nvfp4|tq[12]_\d+"
    r")(?:$|[-_ .])",
    re.IGNORECASE,
)


def family_key(name: str | None, filename: str | None) -> str:
    """Key that groups the different quantizations of one base model.

    ``"Qwen3.8-27B-Uncensored-IQ2_M"`` and ``"Qwen3.8-27B-Uncensored-Q4_K_M"``
    both reduce to ``"qwen3.8-27b-uncensored"``, so they are recognised as
    variants of the same model rather than as unrelated files.
    """
    base = (name or filename or "").strip()
    if not base:
        return ""
    # The quantization is stripped *before* the separators are normalised:
    # "IQ2_M" is one token, and turning it into "IQ2-M" first would hide it.
    base = _QUANT_TOKEN.sub(" ", base)
    base = base.replace("_", "-")
    base = re.sub(r"[-_ .]{2,}", "-", base)
    base = re.sub(r"[-_ .]+$", "", base)
    return base.strip(" -_.").lower()


def group_duplicates(records: Iterable[dict[str, Any]]
                     ) -> tuple[dict[str, list[str]],
                                dict[str, list[str]]]:
    """Group models into duplicates and variants.

    ``records`` are model records carrying an ``integrity`` block. Returns
    ``(duplicates, variants)`` mapping a model id to the ids of its twins.

    * *duplicate* - the same content in two places (same fingerprint). One of
      them can be deleted without losing anything.
    * *variant* - the same base model in a different quantization. Neither
      should be deleted, but the user probably wants to know.
    """
    by_signature: dict[str, list[str]] = {}
    by_family: dict[str, list[str]] = {}
    for rec in records:
        mid = rec.get("id")
        if not mid:
            continue
        sig = (rec.get("integrity") or {}).get("signature")
        if sig:
            by_signature.setdefault(sig, []).append(mid)
        key = family_key(rec.get("name"), rec.get("path"))
        if key:
            by_family.setdefault(key, []).append(mid)

    duplicates: dict[str, list[str]] = {}
    for group in by_signature.values():
        if len(group) > 1:
            for mid in group:
                duplicates[mid] = [o for o in group if o != mid]

    variants: dict[str, list[str]] = {}
    for group in by_family.values():
        if len(group) > 1:
            for mid in group:
                variants[mid] = [o for o in group if o != mid]
    return duplicates, variants
