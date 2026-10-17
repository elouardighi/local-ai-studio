"""Tracks inference requests through the gateway.

Keeps live counters (active requests per instance) plus a bounded history of
finished requests with real latency and, when the engine reports token counts,
real throughput. No number here is estimated: token counts come from the
engine's own usage block, latency is measured in-process.
"""
from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from typing import Any


class RequestTracker:
    def __init__(self, max_history: int = 500) -> None:
        self._lock = threading.Lock()
        self._active: dict[str, dict[str, Any]] = {}
        self._history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self._totals = {
            "requests": 0, "errors": 0, "tokens_prompt": 0,
            "tokens_completion": 0, "streaming": 0,
        }

    def begin(self, endpoint: str, model: str | None,
              instance_id: str | None, streaming: bool) -> str:
        request_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._active[request_id] = {
                "request_id": request_id,
                "endpoint": endpoint,
                "model": model,
                "instance_id": instance_id,
                "streaming": streaming,
                "started_at": time.time(),
            }
            self._totals["requests"] += 1
            if streaming:
                self._totals["streaming"] += 1
        return request_id

    def finish(self, request_id: str, status: int,
               prompt_tokens: int | None = None,
               completion_tokens: int | None = None,
               error: str | None = None) -> dict[str, Any] | None:
        with self._lock:
            meta = self._active.pop(request_id, None)
            if meta is None:
                return None
            record = {
                **meta,
                "finished_at": time.time(),
                "latency_ms": round((time.time() - meta["started_at"]) * 1000, 1),
                "status": status,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "error": error[:500] if error else None,
            }
            if completion_tokens and record["latency_ms"] > 0:
                record["tps"] = round(
                    completion_tokens / (record["latency_ms"] / 1000), 1)
            self._history.append(record)
            if status >= 400 or error:
                self._totals["errors"] += 1
            if prompt_tokens:
                self._totals["tokens_prompt"] += prompt_tokens
            if completion_tokens:
                self._totals["tokens_completion"] += completion_tokens
            return record

    def increment_active(self, instance_id: str) -> None:
        with self._lock:
            for meta in self._active.values():
                if meta.get("instance_id") == instance_id:
                    meta["active_for_instance"] = meta.get(
                        "active_for_instance", 0) + 1

    def active_count(self, instance_id: str | None = None) -> int:
        with self._lock:
            if instance_id is None:
                return len(self._active)
            return sum(1 for m in self._active.values()
                       if m.get("instance_id") == instance_id)

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._history)[-limit:]

    def totals(self) -> dict[str, Any]:
        with self._lock:
            return {**self._totals, "active": len(self._active)}
