"""Log store tests: redaction, bounds, and live streaming to the event bus.

The Logs page bug hunt exposed that only engine output reached the live
dashboard: ``LogStore`` wrote every line to memory but never published to the
event bus, so ``settings``/``gateway``/``scanner`` lines only appeared on
reload. ``on_add`` closes that gap, and these tests pin the behaviour.
"""
from __future__ import annotations

from backend.app.services.event_bus import EventBus
from backend.app.services.log_store import LogStore, redact


def test_redacts_api_keys_and_bearer_tokens():
    assert "***redacted***" in redact("api_key: sk-secret123")
    assert "sk-secret123" not in redact("api_key: sk-secret123")
    assert "***redacted***" in redact("Authorization: Bearer abc.def.ghi")
    assert "abc.def.ghi" not in redact("Authorization: Bearer abc.def.ghi")
    assert "***redacted***" in redact("llama-server --api-key hunter2")


def test_add_returns_record_and_tail_filters():
    store = LogStore()
    store.add("info", "gateway", "request done")
    store.add("error", "gateway", "boom")
    store.add("info", "scanner", "scan complete")

    assert store.tail(limit=10)[0]["level"] == "INFO"  # normalized
    assert len(store.tail(level="error")) == 1
    assert len(store.tail(component="scanner")) == 1
    assert len(store.tail(search="boom")) == 1


def test_store_is_bounded():
    store = LogStore(max_lines=5)
    for i in range(50):
        store.add("info", "c", f"line {i}")
    lines = store.tail(limit=100)
    assert len(lines) == 5
    assert lines[-1]["message"] == "line 49"


def test_on_add_streams_every_line_to_the_bus():
    """Any component's log line must reach live subscribers, not just engine
    output - this is what made the dashboard Logs page look half-dead."""
    bus = EventBus()
    store = LogStore(on_add=lambda rec: bus.publish(
        "log", rec, rec.get("instance_id")))

    store.add("info", "settings", "settings updated: default_ctx")
    store.add("warning", "gateway", "slow request", instance_id="i9")

    recent = bus.recent()
    assert [e["type"] for e in recent] == ["log", "log"]
    assert recent[0]["data"]["component"] == "settings"
    assert recent[0]["data"]["ts"] > 0          # a real timestamp, not None
    assert recent[1]["instance_id"] == "i9"     # instance id carried through


def test_a_broken_sink_never_breaks_logging():
    """The bus must not be able to take down the log store."""
    def boom(_record):
        raise RuntimeError("bus down")

    store = LogStore(on_add=boom)
    record = store.add("info", "c", "still logged")
    assert record["message"] == "still logged"
    assert store.tail(limit=1) == [record]      # line kept regardless


def test_app_context_wires_log_store_to_event_bus():
    """End-to-end: ctx.log lines appear on the event bus (what the
    WebSocket handler forwards to the browser)."""
    from backend.app.services.context import AppContext

    ctx = AppContext()
    ctx.log.info("settings", "settings updated: scan_interval_seconds")

    recent = ctx.events.recent()
    log_events = [e for e in recent if e["type"] == "log"]
    assert log_events, "ctx.log lines must publish to the event bus"
    assert log_events[-1]["data"]["component"] == "settings"
    assert isinstance(log_events[-1]["data"]["ts"], float)
