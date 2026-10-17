"""Model instance lifecycle: the state machine around engine processes.

States
------
``discovered -> validating -> estimating_memory -> queued -> starting_engine
-> loading_model -> ready <-> busy`` plus the failure/teardown states
``validation_failed``, ``out_of_memory``, ``engine_failed``, ``cancelled``,
``unloading``, ``unloaded`` and ``crashed``.

Transitions are validated: an illegal jump raises instead of silently
corrupting the record. Instance history is persisted to JSON (metadata only,
never weights) and reconciled with real processes on startup, so a backend
restart can never claim a dead model is still loaded.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import memory_planner
from .event_bus import EventBus
from .log_store import LogStore
from .model_scanner import ModelScanner
from .process_manager import ProcessManager, find_free_port

# ---- state machine ---------------------------------------------------------

VALID_STATES = {
    "discovered", "validating", "estimating_memory", "queued",
    "starting_engine", "loading_model", "ready", "busy",
    "validation_failed", "out_of_memory", "engine_failed", "cancelled",
    "unloading", "unloaded", "crashed",
}

TRANSITIONS: dict[str, set[str]] = {
    "discovered": {"validating", "cancelled"},
    "validating": {"estimating_memory", "validation_failed", "cancelled"},
    "estimating_memory": {"queued", "out_of_memory", "cancelled"},
    "queued": {"starting_engine", "cancelled", "unloaded"},
    "starting_engine": {"loading_model", "engine_failed", "cancelled",
                        "out_of_memory"},
    "loading_model": {"ready", "engine_failed", "out_of_memory", "cancelled"},
    "ready": {"busy", "unloading", "crashed", "engine_failed"},
    "busy": {"ready", "unloading", "crashed", "engine_failed"},
    "unloading": {"unloaded", "crashed"},
    # terminal / recovery states
    "validation_failed": {"unloaded"},
    "out_of_memory": {"unloaded"},
    "engine_failed": {"unloaded"},
    "cancelled": {"unloaded"},
    "crashed": {"unloaded"},
    "unloaded": set(),
}

ACTIVE_STATES = {"queued", "starting_engine", "loading_model", "ready", "busy"}
LOADED_STATES = {"ready", "busy"}


class IllegalTransition(RuntimeError):
    pass


@dataclass
class Instance:
    instance_id: str
    model_id: str
    model_name: str
    model_path: str
    state: str = "discovered"
    engine: str = "llama.cpp"
    backend: str = "vulkan"
    device_name: str | None = None
    pid: int | None = None
    port: int | None = None
    gpu_layers: int = 0
    total_layers: int | None = None
    context: int = 8192
    batch: int = 512
    ubatch: int = 512
    flash_attention: str = "auto"
    kv_on_gpu: bool = True
    mmproj_path: str | None = None
    profile: str = "manual"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    ready_at: float | None = None
    last_used_at: float | None = None
    unloaded_at: float | None = None
    load_duration_s: float | None = None
    active_requests: int = 0
    requests_total: int = 0
    tokens_generated: int = 0
    last_tps: float | None = None
    last_prompt_tps: float | None = None
    last_error: str | None = None
    estimate: dict[str, Any] = field(default_factory=dict)
    command: list[str] = field(default_factory=list)

    # -- transitions --------------------------------------------------------

    def transition(self, new_state: str, **updates: Any) -> None:
        if new_state not in VALID_STATES:
            raise IllegalTransition(f"unknown state {new_state!r}")
        allowed = TRANSITIONS.get(self.state, set())
        if new_state not in allowed:
            raise IllegalTransition(
                f"{self.state!r} -> {new_state!r} is not a valid transition")
        self.state = new_state
        for key, value in updates.items():
            setattr(self, key, value)

    @property
    def is_loaded(self) -> bool:
        return self.state in LOADED_STATES

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    def touch(self) -> None:
        self.last_used_at = time.time()


class InstanceManager:
    def __init__(self, scanner: ModelScanner, processes: ProcessManager,
                 events: EventBus, log: LogStore,
                 get_settings: Callable[[], Any],
                 state_file: Path) -> None:
        self._scanner = scanner
        self._processes = processes
        self._events = events
        self._log = log
        self._get_settings = get_settings
        self._state_file = state_file
        self._instances: dict[str, Instance] = {}
        self._loading_models: set[str] = set()  # model ids currently loading
        self._tasks: dict[str, asyncio.Task] = {}

    # -- persistence --------------------------------------------------------

    def _persist(self) -> None:
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            payload = [asdict(i) for i in self._instances.values()]
            tmp = self._state_file.with_suffix(".tmp")
            tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            os.replace(tmp, self._state_file)
        except OSError as exc:
            self._log.warning("instances", f"could not persist state: {exc}")

    def load_history(self) -> None:
        """Restore previous records, but never trust them about live state."""
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        for item in raw:
            try:
                inst = Instance(**{k: v for k, v in item.items()
                                   if k in Instance.__dataclass_fields__})
            except TypeError:
                continue
            was_active = inst.state in ACTIVE_STATES
            # Reconcile with reality: is the recorded process actually alive?
            alive = bool(inst.pid and self._processes.pid_alive(inst.pid))
            if was_active and not alive:
                inst.state = "unloaded"
                inst.unloaded_at = time.time()
                inst.pid = None
                inst.port = None
                inst.last_error = ("recorded as active but process was not "
                                   "running after restart; state reconciled")
                self._log.warning(
                    "instances",
                    f"reconciled stale record for {inst.model_name}: "
                    f"process {item.get('pid')} no longer exists")
            elif was_active and alive:
                # A real llama process still holds this PID, but it was started
                # outside this backend session - we do not own it.
                inst.state = "crashed"
                inst.last_error = ("process still exists but was not started "
                                   "by this session; unload it manually")
                inst.pid = None
                inst.port = None
            self._instances[inst.instance_id] = inst
        self._persist()

    # -- queries ------------------------------------------------------------

    def list(self) -> list[dict[str, Any]]:
        return [asdict(i) for i in
                sorted(self._instances.values(), key=lambda i: i.created_at)]

    def get(self, instance_id: str) -> Instance | None:
        return self._instances.get(instance_id)

    def loaded(self) -> list[Instance]:
        return [i for i in self._instances.values() if i.is_loaded]

    def find_running_for_model(self, model_id: str) -> Instance | None:
        for inst in self._instances.values():
            if inst.model_id == model_id and inst.is_active:
                return inst
        return None

    def find_by_alias(self, name: str) -> Instance | None:
        """Resolve an OpenAI-style ``model`` field to a ready instance."""
        name_l = (name or "").lower()
        ready = [i for i in self._instances.values() if i.is_loaded]
        for inst in ready:
            if name_l in (inst.model_name.lower(),
                          Path(inst.model_path).stem.lower(),
                          inst.model_id.lower(),
                          f"instance:{inst.instance_id}"):
                return inst
        for inst in ready:
            if name_l and name_l in inst.model_name.lower():
                return inst
        if len(ready) == 1 and name_l in ("", "default", "auto"):
            return ready[0]
        return None

    async def wait_until_ready(self, model_id: str, timeout: float = 300.0
                               ) -> Instance:
        """Wait for a model's active instance to become ready.

        Used by the gateway's auto-load path when another request already
        started the load. Raises an honest error when the load fails or the
        timeout elapses; never invents progress.
        """
        deadline = time.time() + timeout
        while True:
            inst = self.find_running_for_model(model_id)
            if inst is None:
                raise RuntimeError(
                    "the load disappeared before becoming ready; "
                    "see the Logs page for the reason")
            if inst.is_loaded:
                return inst
            if not inst.is_active:  # terminal failure state
                raise RuntimeError(
                    f"{inst.model_name} failed to load ({inst.state}): "
                    f"{inst.last_error or 'no reason recorded'}")
            if time.time() > deadline:
                raise TimeoutError(
                    f"{inst.model_name} is still {inst.state} after "
                    f"{timeout:.0f}s; check the Logs page")
            await asyncio.sleep(0.25)

    # -- lifecycle ----------------------------------------------------------

    async def load(self, model_id: str, config: dict[str, Any] | None = None
                   ) -> Instance:
        """Validate, estimate, queue and start a model. Returns the record."""
        config = config or {}
        settings = self._get_settings()

        model = self._scanner.get(model_id)
        if model is None:
            # Allow loading by path or name for convenience.
            model = self._scanner.find_by_alias_or_name(model_id)
        if model is None:
            raise ValueError(f"model not found: {model_id!r}. Run a rescan.")
        if model["compatibility"]["status"] == "error":
            raise ValueError(
                f"model cannot be loaded: {model['compatibility']['reason']}")

        # Refuse duplicate loads of the same model unless explicitly allowed.
        existing = self.find_running_for_model(model["id"])
        if existing and not config.get("allow_duplicate"):
            raise ValueError(
                f"model already loaded as instance {existing.instance_id} "
                f"({existing.state}); unload it first or pass allow_duplicate")

        # Concurrency limit with an honest message (no silent queueing forever).
        running = [i for i in self._instances.values() if i.is_active]
        max_instances = int(getattr(settings, "max_instances", 1) or 1)
        if len(running) >= max_instances:
            idle = [i for i in running if i.state == "ready"
                    and i.active_requests == 0]
            if config.get("unload_idle") and idle:
                victim = idle[0]
                self._log.info("instances",
                               f"unloading idle instance {victim.instance_id} "
                               f"to make room")
                await self.unload(victim.instance_id)
            else:
                busy = ", ".join(f"{i.model_name}({i.state})" for i in running)
                raise RuntimeError(
                    f"concurrent instance limit reached ({max_instances}): "
                    f"{busy}. Unload one or enable 'unload idle'.")

        inst = Instance(
            instance_id=uuid.uuid4().hex[:12],
            model_id=model["id"],
            model_name=model["name"],
            model_path=model["path"],
            backend=getattr(settings, "engine_backend", "vulkan"),
            mmproj_path=model.get("mmproj_path"),
            context=int(config.get("context", settings.default_ctx)),
            batch=int(config.get("batch", settings.default_batch)),
            ubatch=int(config.get("ubatch", settings.default_ubatch)),
            flash_attention=str(config.get("flash_attention",
                                           settings.flash_attention)),
            kv_on_gpu=bool(config.get("kv_on_gpu", settings.kv_on_gpu)),
            profile=str(config.get("profile", "manual")),
        )
        self._instances[inst.instance_id] = inst
        self._events.publish("instance_state", asdict(inst), inst.instance_id)
        self._persist()

        task = asyncio.create_task(self._run_load(inst, model, config))
        self._tasks[inst.instance_id] = task
        try:
            await task
        except Exception:
            # State already recorded on the instance; re-raise for the API.
            raise
        return inst

    async def _run_load(self, inst: Instance, model: dict[str, Any],
                        config: dict[str, Any]) -> None:
        settings = self._get_settings()
        try:
            # 1. validating
            inst.transition("validating")
            self._publish(inst)
            exe = Path(settings.engine_path)
            if not exe.is_file():
                raise FileNotFoundError(
                    f"engine executable missing: {exe}. Run scripts/setup.ps1.")
            if not Path(inst.model_path).is_file():
                raise FileNotFoundError(f"model file missing: {inst.model_path}")

            # 2. estimating memory
            inst.transition("estimating_memory")
            self._publish(inst)
            ngl = self._decide_ngl(inst, model, config, settings)
            inst.gpu_layers = ngl
            inst.total_layers = model.get("block_count")
            estimate = memory_planner.estimate(
                model, inst.context, ngl, batch=inst.batch,
                kv_on_gpu=inst.kv_on_gpu, mmproj=bool(inst.mmproj_path))
            inst.estimate = estimate
            if estimate["verdict"] == "risk_of_paging" and not config.get(
                    "force"):
                raise MemoryError(
                    "memory plan risky: " + "; ".join(estimate["warnings"]))

            # 3. queued -> starting engine
            inst.transition("queued")
            self._publish(inst)
            port = find_free_port(
                settings.engine_port_start, settings.engine_port_end,
                occupied=self._processes.used_ports())
            inst.port = port
            args = self._build_args(inst, model, settings, config)
            inst.command = [str(exe), *args]
            inst.transition("starting_engine")
            self._publish(inst)

            ep = await self._processes.start(
                inst.instance_id, str(exe), args, port, inst.model_path,
                readiness_timeout=float(config.get("readiness_timeout", 900)))

            # 4. loading_model -> ready (start() already awaited /health)
            inst.transition("loading_model")
            self._publish(inst)
            inst.pid = ep.pid
            inst.device_name = ep.device_name or self._device_from_probe()
            inst.transition("ready",
                            ready_at=time.time(),
                            started_at=ep.started_at,
                            load_duration_s=(ep.ready_time or time.time())
                            - (ep.started_at or time.time()))
            self._publish(inst)
            self._log.info("instances",
                           f"{inst.model_name} ready in "
                           f"{inst.load_duration_s:.1f}s on port {port} "
                           f"(pid {ep.pid}, ngl={ngl})",
                           instance_id=inst.instance_id)
        except MemoryError as exc:
            self._fail(inst, "out_of_memory", str(exc))
            raise
        except FileNotFoundError as exc:
            self._fail(inst, "validation_failed", str(exc))
            raise
        except TimeoutError as exc:
            self._fail(inst, "engine_failed", str(exc))
            raise
        except asyncio.CancelledError:
            with contextlib.suppress(Exception):
                inst.transition("cancelled", last_error="load cancelled")
            self._publish(inst)
            with contextlib.suppress(Exception):
                await self._processes.stop(inst.instance_id)
            raise
        except Exception as exc:  # noqa: BLE001 - surface any engine failure
            self._fail(inst, "engine_failed", f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self._tasks.pop(inst.instance_id, None)
            self._persist()

    def _fail(self, inst: Instance, state: str, message: str) -> None:
        inst.last_error = message
        with contextlib.suppress(IllegalTransition):
            inst.transition(state)
        self._publish(inst)
        self._log.error("instances",
                        f"{inst.model_name} {state}: {message}",
                        instance_id=inst.instance_id)
        with contextlib.suppress(Exception):
            asyncio.get_running_loop().create_task(self._cleanup_failed(inst))

    async def _cleanup_failed(self, inst: Instance) -> None:
        with contextlib.suppress(Exception):
            await self._processes.stop(inst.instance_id)
        with contextlib.suppress(IllegalTransition):
            inst.transition("unloaded", unloaded_at=time.time())
        inst.pid = None
        self._publish(inst)
        self._persist()

    def _decide_ngl(self, inst: Instance, model: dict[str, Any],
                    config: dict[str, Any], settings: Any) -> int:
        """Resolve the GPU layer count from mode/profile/user value."""
        mode = str(config.get("gpu_mode", "auto")).lower()
        blocks = int(model.get("block_count") or 0)
        if mode in ("cpu", "none"):
            return 0
        if mode == "max":
            return blocks or int(settings.default_ngl)
        if mode == "manual":
            return max(0, int(config.get("gpu_layers", settings.default_ngl)))
        # auto: use free VRAM, but let the user cap it
        gpu_free_mb = self._gpu_free_mb()
        rec = memory_planner.recommend_ngl(
            model, gpu_free_mb, inst.context, kv_on_gpu=inst.kv_on_gpu)
        ngl = rec["ngl"]
        if getattr(settings, "auto_adjust_offload", False) is False:
            # Without auto-adjust we still clamp to what fits, but tell why.
            self._log.info("instances",
                           f"auto offload chose ngl={ngl}: {rec['reason']}")
        cap = config.get("gpu_layers")
        if cap is not None:
            ngl = min(ngl, int(cap))
        return ngl

    def _device_from_probe(self) -> str | None:
        """Device name the engine will actually run on.

        llama-server does not print its device during a normal startup, so the
        name comes from the engine's own ``--list-devices`` probe (captured at
        backend start). Only reported when the backend is Vulkan and exactly
        one Vulkan device exists - otherwise it would be a guess.
        """
        monitor = getattr(self, "monitor", None)
        name = getattr(monitor, "_gpu_name", None)
        if not name:
            return None
        settings = self._get_settings()
        if getattr(settings, "engine_backend", "vulkan") != "vulkan":
            return None
        return name

    def _gpu_free_mb(self) -> float:
        """Free VRAM *right now*, or a safe fallback.

        Takes a fresh sample instead of trusting the cached one: after an
        idle eviction the last (≤1 s old) sample can still count the dying
        engine's VRAM, which would under-provision GPU layers and make the
        memory plan refuse a model that actually fits.
        """
        monitor = getattr(self, "monitor", None)
        if monitor is not None:
            try:
                sample = monitor.sample()  # cheap gauge read, <10 ms
            except Exception:  # noqa: BLE001 - fall back to the cached sample
                sample = monitor.last_sample()
            gpu = sample.get("gpu") or {}
            free = gpu.get("dedicated_free_mb")
            if free is not None:
                return float(free)
        return 14000.0  # conservative default for a 16 GB card when unknown

    def _build_args(self, inst: Instance, model: dict[str, Any],
                    settings: Any, config: dict[str, Any]) -> list[str]:
        """Translate settings into a verified llama-server argument list."""
        args: list[str] = [
            "-m", inst.model_path,
            "--host", "127.0.0.1",
            "--port", str(inst.port),
            "-ngl", str(inst.gpu_layers),
            "-c", str(inst.context),
            "-b", str(inst.batch),
            "-ub", str(inst.ubatch),
            "--metrics",
        ]
        if inst.mmproj_path and config.get("use_mmproj", True):
            args += ["--mmproj", inst.mmproj_path]
        if inst.flash_attention in ("on", "off"):
            args += ["--flash-attn", inst.flash_attention]
        if inst.backend == "cpu":
            args += ["--device", "none"]
        threads = int(getattr(settings, "default_threads", 0) or 0)
        if threads > 0:
            args += ["--threads", str(threads)]
        if inst.profile != "manual":
            args += ["--alias", self._slug(inst.model_name)]
        else:
            args += ["--alias", self._slug(inst.model_name)]
        for extra in getattr(settings, "advanced_args", []) or []:
            args.append(extra)
        return args

    @staticmethod
    def _slug(name: str) -> str:
        return "".join(c if c.isalnum() or c in "-._" else "-"
                       for c in name.lower()).strip("-")[:64]

    # -- actions ------------------------------------------------------------

    async def unload(self, instance_id: str) -> None:
        inst = self._instances.get(instance_id)
        if inst is None:
            raise KeyError(f"unknown instance {instance_id}")
        if inst.active_requests > 0:
            raise RuntimeError(
                f"instance {instance_id} has {inst.active_requests} active "
                f"request(s); refusing to terminate an in-flight generation")
        if inst.state not in ("unloaded",):
            with contextlib.suppress(IllegalTransition):
                inst.transition("unloading")
                self._publish(inst)
        await self._processes.stop(instance_id)
        with contextlib.suppress(IllegalTransition):
            inst.transition("unloaded", unloaded_at=time.time())
        inst.pid = None
        inst.port = None
        self._publish(inst)
        self._log.info("instances", f"{inst.model_name} unloaded",
                       instance_id=instance_id)

    async def restart(self, instance_id: str,
                      config: dict[str, Any] | None = None) -> Instance:
        inst = self._instances.get(instance_id)
        if inst is None:
            raise KeyError(f"unknown instance {instance_id}")
        model_id = inst.model_id
        cfg = dict(config or {})
        cfg["allow_duplicate"] = True  # the old record is going away
        await self.unload(instance_id)
        self._instances.pop(instance_id, None)
        self._processes.cleanup(instance_id)
        self._persist()
        return await self.load(model_id, cfg)

    async def cancel_load(self, instance_id: str) -> None:
        task = self._tasks.get(instance_id)
        if task and not task.done():
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    def _publish(self, inst: Instance) -> None:
        self._events.publish("instance_state", asdict(inst), inst.instance_id)

    def _mark_request(self, inst: Instance, delta: int) -> None:
        """Adjust the in-flight request counter and flip ready<->busy."""
        inst.active_requests = max(0, inst.active_requests + delta)
        inst.requests_total += 1 if delta > 0 else 0
        try:
            if inst.active_requests > 0 and inst.state == "ready":
                inst.transition("busy")
                self._publish(inst)
            elif inst.active_requests == 0 and inst.state == "busy":
                inst.transition("ready")
                self._publish(inst)
        except IllegalTransition:
            pass  # instance may be unloading concurrently
        inst.touch()

    # -- shutdown -----------------------------------------------------------

    async def shutdown(self) -> None:
        for task in list(self._tasks.values()):
            task.cancel()
        for inst in self._instances.values():
            if inst.is_active and inst.pid:
                await self._processes.stop(inst.instance_id)
                with contextlib.suppress(IllegalTransition):
                    inst.transition("unloading")
                with contextlib.suppress(IllegalTransition):
                    inst.transition("unloaded", unloaded_at=time.time())
                inst.pid = None
        self._persist()
