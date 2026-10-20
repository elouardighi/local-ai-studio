"""Removing a file, safely.

Deletion is the one operation in this project that destroys something the
user cares about, so it is deliberately conservative:

* **The Recycle Bin is the default.** A deleted model goes where the
  Windows shell puts deleted files, so it can be put back. A permanent
  delete has to be asked for explicitly.
* **Only files are removed.** Never a directory, never a symlink pointing
  somewhere else.
* **The caller never supplies the path.** The path always comes from a
  model record the library already knows about.

The Recycle Bin is reached through the shell's own API, so the file ends up
in the same place the Explorer's *Delete* would put it - with its original
name, and restorable.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import os
from pathlib import Path

# Shell constants for SHFileOperationW.
FO_DELETE = 0x0003
FOF_SILENT = 0x0004
FOF_NOCONFIRMATION = 0x0010
FOF_ALLOWUNDO = 0x0040       # send to the Recycle Bin instead of erasing
FOF_NOERRORUI = 0x0400


class _SHFILEOPSTRUCTW(ctypes.Structure):
    _fields_ = [
        ("hwnd", wt.HWND),
        ("wFunc", wt.UINT),
        ("pFrom", ctypes.c_wchar_p),
        ("pTo", ctypes.c_wchar_p),
        ("fFlags", ctypes.c_int),
        ("fAnyOperationsAborted", wt.BOOL),
        ("hNameMappings", ctypes.c_void_p),
        ("lpszProgressTitle", ctypes.c_wchar_p),
    ]


class RemovalError(RuntimeError):
    """A file could not be removed."""


def _recycle_bin_limit_bytes(path: Path) -> int | None:
    """How much the Recycle Bin will hold for this file's drive.

    Windows silently deletes anything too big for the bin *permanently*,
    even when the caller asked for an undoable delete. The capacity is
    either set explicitly in the registry or defaults to a tenth of the
    volume, so both are checked.
    """
    import shutil
    import winreg

    drive = path.drive or "C:"
    limit_mb: int | None = None
    applies: bool | None = None
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER,
                            r"Software\Microsoft\Windows\CurrentVersion"
                            r"\Explorer\BitBucket") as key:
            value, _ = winreg.QueryValueEx(key, "MaxCapacity")
            limit_mb = int(value)
            number, _ = winreg.QueryValueEx(key, "N")
            applies = (int(number) == _drive_number(drive))
    except (OSError, FileNotFoundError, ValueError):
        limit_mb = None
        applies = None

    if limit_mb and applies:
        return limit_mb * 1024 * 1024
    try:
        usage = shutil.disk_usage(drive + "\\")
    except OSError:
        return None
    return int(usage.total * 0.10)


def _drive_number(drive: str) -> int:
    """The drive number the registry's ``N`` value refers to."""
    letter = (drive or "C:").rstrip(":\\").upper()[:1]
    return ord(letter) - ord("A") + 1


def will_be_recoverable(path: Path) -> tuple[bool, str | None]:
    """Whether the Recycle Bin will take this file, and why not if it won't.

    A model that is too big for the bin is removed for good, and the user
    has to be told that *before* they press Delete.
    """
    try:
        size = path.stat().st_size
    except OSError:
        return False, "the file cannot be read"
    limit = _recycle_bin_limit_bytes(path)
    if limit is None:
        return True, None
    if size > limit:
        return False, (f"the file is {size / 2**20:,.0f} MB but the Recycle "
                       f"Bin on this drive holds {limit / 2**20:,.0f} MB, so "
                       f"Windows would delete it permanently")
    return True, None


def recycle(path: Path) -> bool:
    """Move a file to the Windows Recycle Bin.

    Returns True when the shell reported success. The file is gone from its
    old place either way, so the caller should not treat a False as "the
    file is still there" - it means "it was not put in the Recycle Bin".
    """
    target = str(path)
    if not os.path.isfile(target):
        raise RemovalError(f"not a file: {target}")
    op = _SHFILEOPSTRUCTW()
    op.wFunc = FO_DELETE
    op.pFrom = target + "\0\0"          # double null: a list of paths
    op.fFlags = (FOF_ALLOWUNDO | FOF_NOCONFIRMATION | FOF_SILENT
                 | FOF_NOERRORUI)
    try:
        code = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(op))
    except (AttributeError, OSError) as exc:
        raise RemovalError(f"the shell could not be reached: {exc}") from exc
    if code != 0:
        raise RemovalError(f"the shell refused the delete (code {code})")
    if op.fAnyOperationsAborted:
        raise RemovalError("the delete was aborted")
    return True


def remove_permanently(path: Path) -> bool:
    """Erase a file for good. Only used when the user says so."""
    target = str(path)
    if not os.path.isfile(target):
        raise RemovalError(f"not a file: {target}")
    try:
        os.remove(target)
    except OSError as exc:
        raise RemovalError(f"could not delete {target}: {exc}") from exc
    return True


def remove(path: Path, permanent: bool = False) -> dict[str, object]:
    """Remove one file and report what happened."""
    size = 0
    try:
        size = path.stat().st_size
    except OSError:
        pass
    if permanent:
        remove_permanently(path)
        return {"path": str(path), "size_bytes": size,
                "method": "deleted permanently", "recoverable": False}
    recoverable, why = will_be_recoverable(path)
    recycle(path)
    if recoverable:
        return {"path": str(path), "size_bytes": size,
                "method": "moved to the Recycle Bin", "recoverable": True}
    # Windows deletes anything too big for the bin for good, whatever the
    # caller asked for. Say so rather than promise a recovery that will
    # not happen.
    return {"path": str(path), "size_bytes": size,
            "method": "deleted permanently (too large for the Recycle Bin)",
            "recoverable": False, "reason": why}
