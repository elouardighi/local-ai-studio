"""Bounded, redacting application log store.

Every component (scanner, process manager, gateway, engine stdout/stderr)
writes here. The store keeps a fixed number of lines so memory stays bounded,
and redacts secrets before they ever reach a browser or an exported file.
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from typing import Any, Callable

# Patterns that must never appear in logs shown to a user or written to disk.
# Case-insensitive: HTTP header names and CLI flags vary in casing
# ("Authorization: Bearer ..." is the common real-world form).
_REDACTIONS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"(api[_-]?key[\"'\s:=]+)([^\s\"']+)", re.IGNORECASE),
     r"\1***redacted***"),
    (re.compile(r"(authorization[\"'\s:=]+bearer\s+)([^\s\"']+)",
                re.IGNORECASE), r"\1***redacted***"),
    (re.compile(r"(x-api-key[\"'\s:=]+)([^\s\"']+)", re.IGNORECASE),
     r"\1***redacted***"),
    (re.compile(r"(--api-key\s+)(\S+)", re.IGNORECASE), r"\1***redacted***"),
)


def redact(text: str) -> str:
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class LogStore:
    def __init__(self, max_lines: int = 5000,
                 on_add: Callable[[dict[str, Any]], None] | None = None) -> None:
        self._lines: deque[dict[str, Any]] = deque(maxlen=max_lines)
        self._lock = threading.Lock()
        # Optional sink (the event bus) so *every* log line streams live to
        # the dashboard, not just engine stdout/stderr.
        self._on_add = on_add

    def add(self, level: str, component: str, message: str,
            instance_id: str | None = None, **extra: Any) -> dict[str, Any]:
        record = {
            "ts": time.time(),
            "level": level.upper(),
            "component": component,
            "instance_id": instance_id,
            "message": redact(str(message))[:8000],
        }
        if extra:
            record["extra"] = extra
        with self._lock:
            self._lines.append(record)
        if self._on_add is not None:
            try:
                self._on_add(record)
            except Exception:  # noqa: BLE001 - a bus hiccup must not kill logging
                pass
        return record

    # Convenience wrappers -------------------------------------------------
    def info(self, component: str, message: str, **kw: Any) -> dict[str, Any]:
        return self.add("info", component, message, **kw)

    def warning(self, component: str, message: str, **kw: Any) -> dict[str, Any]:
        return self.add("warning", component, message, **kw)

    def error(self, component: str, message: str, **kw: Any) -> dict[str, Any]:
        return self.add("error", component, message, **kw)

    def debug(self, component: str, message: str, **kw: Any) -> dict[str, Any]:
        return self.add("debug", component, message, **kw)

    def tail(self, limit: int = 200, level: str | None = None,
             component: str | None = None, search: str | None = None,
             instance_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            lines = list(self._lines)
        if level:
            lines = [l for l in lines if l["level"] == level.upper()]
        if component:
            lines = [l for l in lines if l["component"] == component]
        if instance_id:
            lines = [l for l in lines if l.get("instance_id") == instance_id]
        if search:
            needle = search.lower()
            lines = [l for l in lines if needle in l["message"].lower()]
        return lines[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
