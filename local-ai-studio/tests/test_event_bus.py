"""Event bus tests: history, subscription, and telemetry exclusion."""
from __future__ import annotations

import asyncio

from backend.app.services.event_bus import EventBus


def test_publish_records_history_and_returns_event():
    bus = EventBus()
    ev = bus.publish("instance_state", {"state": "ready"}, "i1")
    assert ev["type"] == "instance_state"
    assert ev["instance_id"] == "i1"
    assert ev["ts"] > 0
    assert bus.recent() == [ev]


def test_resource_samples_are_not_kept_in_history():
    """Per-second telemetry would drown the actual events in the replay."""
    bus = EventBus()
    bus.publish("instance_state", {"state": "ready"}, "i1")
    for _ in range(50):
        bus.publish("resources", {"cpu_percent": 1.0})
    bus.publish("log", {"message": "x"})
    types = [e["type"] for e in bus.recent()]
    assert "resources" not in types
    assert types == ["instance_state", "log"]


def test_resource_samples_still_reach_subscribers():
    async def scenario() -> list[str]:
        bus = EventBus()
        seen: list[str] = []

        async def consume() -> None:
            async for event in bus.subscribe():
                seen.append(event["type"])

        task = asyncio.create_task(consume())
        await asyncio.sleep(0)          # let the subscriber attach
        bus.publish("resources", {"cpu_percent": 2.0})
        bus.publish("instance_state", {"state": "busy"}, "i1")
        await asyncio.sleep(0.05)
        task.cancel()
        return seen

    assert asyncio.run(scenario()) == ["resources", "instance_state"]
