"""Application context: one place that owns every service instance."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ENGINE_DIR, SettingsStore
from .event_bus import EventBus
from .instance_manager import InstanceManager
from .log_store import LogStore
from .memory_planner import get_system_memory
from .model_scanner import ModelScanner
from .process_manager import ProcessManager
from .request_tracker import RequestTracker
from .resource_monitor import ResourceMonitor


class AppContext:
    """Holds the long-lived services. Created once at FastAPI startup."""

    def __init__(self, settings_store: SettingsStore | None = None) -> None:
        self.settings_store = settings_store or SettingsStore()
        self.events = EventBus()
        self.log = LogStore(
            max_lines=self.settings_store.settings.log_max_lines,
            # Stream every log line to the dashboard, whatever component
            # wrote it (settings, gateway, scanner, instances, engine).
            on_add=lambda record: self.events.publish(
                "log", record, record.get("instance_id")))
        self.tracker = RequestTracker()

        def _log(level: str, message: str, instance_id: str | None = None
                 ) -> None:
            # LogStore publishes to the bus itself; no extra publish here.
            self.log.add(level, "process", message,
                         instance_id=instance_id)

        self.processes = ProcessManager(log=_log, on_exit=self._on_engine_exit)
        self.scanner = ModelScanner(
            get_dirs=lambda: self.settings_store.settings.model_dirs,
            log=self.log,
        )
        self.instances = InstanceManager(
            scanner=self.scanner, processes=self.processes,
            events=self.events, log=self.log,
            get_settings=lambda: self.settings_store.settings,
            state_file=Path(self.settings_store.settings.converted_dir).parent
            / "instances.json",
        )
        # The monitor needs a live view of instances for per-PID attribution.
        self.monitor = ResourceMonitor(
            get_settings=lambda: self.settings_store.settings,
            events=self.events, log=self.log,
            history_max=self.settings_store.settings.history_max_samples,
        )
        # Give the instance manager access to live GPU readings.
        self.instances.monitor = self.monitor  # type: ignore[attr-defined]

    # -- engine exit handling ----------------------------------------------

    async def _on_engine_exit(self, instance_id: str, code: int | None) -> None:
        """React to an engine that died on its own (crash / OOM kill)."""
        inst = self.instances.get(instance_id)
        if inst is None:
            return
        if inst.state in ("unloading", "unloaded", "crashed"):
            return  # we asked for this
        from .instance_manager import IllegalTransition
        inst.last_error = f"engine process exited unexpectedly (code {code})"
        try:
            inst.transition("crashed")
        except IllegalTransition:
            inst.state = "crashed"
        inst.pid = None
        self.instances._publish(inst)  # noqa: SLF001 - same package
        self.log.error("instances",
                       f"{inst.model_name} crashed (exit code {code})",
                       instance_id=instance_id)
        self.events.publish("instance_state", inst.__dict__, instance_id)
        self.instances._persist()  # noqa: SLF001

    # -- startup / shutdown -------------------------------------------------

    async def startup(self) -> None:
        self.log.info("app", "starting Local AI Studio backend")
        self.instances.load_history()
        self.monitor.probe_gpu_via_engine(ENGINE_DIR)
        self.scanner.scan()
        self.monitor.start(instances_ref=self.instances.list)
        self.log.info("app", f"ready: {len(self.scanner.list())} models found")

    async def shutdown(self) -> None:
        self.log.info("app", "shutting down")
        await self.instances.shutdown()
        await self.monitor.stop()

    # -- summary ------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        s = self.settings_store.settings
        loaded = self.instances.loaded()
        sample = self.monitor.last_sample()
        return {
            "service": "local-ai-studio",
            "status": "ok",
            "api_base": f"http://{s.bound_host()}:{s.api_port}/v1",
            "api_key_required": bool(s.api_key),
            "engine": {
                "path": s.engine_path,
                "backend": s.engine_backend,
                "device": self.monitor._gpu_name,  # noqa: SLF001
            },
            "models_discovered": len(self.scanner.list()),
            "instances_loaded": len(loaded),
            "active_requests": self.tracker.active_count(),
            "system": get_system_memory(),
            "gpu": sample.get("gpu"),
            "last_scan": self.scanner.last_scan,
        }
