"""In-process publish/subscribe event bus feeding the WebSocket and SSE APIs.

Design notes
------------
* Each subscriber gets its own bounded queue. A slow browser tab can never
  block inference: when its queue is full the oldest event is dropped and a
  ``events_dropped`` marker is sent instead.
* A small ring buffer keeps the most recent events so a newly opened dashboard
  can render "recent activity" immediately.
"""
from __future__ import annotations

import asyncio
import contextlib
import time
from collections import deque
from typing import Any, AsyncIterator

MAX_QUEUE = 512
HISTORY = 200

# Event types that are streamed live but never added to the replay history.
# Resource samples arrive every second and would drown every other event.
HISTORY_EXCLUDED = {"resources"}


class EventBus:
    def __init__(self, history: int = HISTORY) -> None:
        self._subscribers: set[asyncio.Queue] = set()
        self._history: deque[dict[str, Any]] = deque(maxlen=history)

    def publish(self, event_type: str, data: Any = None,
                instance_id: str | None = None) -> dict[str, Any]:
        """Emit an event to every subscriber. Safe to call from sync code."""
        event = {
            "type": event_type,
            "data": data,
            "instance_id": instance_id,
            "ts": time.time(),
        }
        if event_type not in HISTORY_EXCLUDED:
            self._history.append(event)
        for queue in list(self._subscribers):
            self._offer(queue, event)
        return event

    @staticmethod
    def _offer(queue: asyncio.Queue, event: dict[str, Any]) -> None:
        try:
            queue.put_nowait(event)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                queue.get_nowait()  # drop oldest, keep newest
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(
                    {"type": "events_dropped", "data": "subscriber too slow",
                     "ts": time.time()}
                )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._history)[-limit:]

    async def subscribe(self) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=MAX_QUEUE)
        self._subscribers.add(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)
