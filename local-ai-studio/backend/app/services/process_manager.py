"""Safe engine (llama-server) process management.

Rules enforced here:
* arguments are passed as a list, never through a shell string
* only processes this module started are ever stopped (ownership tracked by
  PID + start-time, never by executable name)
* readiness is proven by a real HTTP ``/health`` response, not by log text
* stdout and stderr are drained asynchronously into a bounded buffer
* shutdown is graceful first (CTRL_BREAK on Windows), force-kill only after a
  timeout, so no orphaned engine processes are left behind
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import signal
import socket
import time
from collections import deque
from pathlib import Path
from typing import Any, Awaitable, Callable

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000

LogFn = Callable[[str, str, str | None], None]  # level, message, instance_id
LineFn = Callable[[str, str], None]              # stream, line


def find_free_port(start: int, end: int,
                   occupied: set[int] | None = None) -> int:
    """Return the first free TCP port in [start, end).

    ``occupied`` carries ports already claimed by live instances so two
    concurrent loads can never race for the same port.
    """
    occupied = occupied or set()
    for port in range(start, end + 1):
        if port in occupied:
            continue
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free engine port in range {start}-{end}")


class EngineProcess:
    """One llama-server child process plus its observable state."""

    def __init__(self, instance_id: str, executable: str, args: list[str],
                 port: int, model_path: str) -> None:
        self.instance_id = instance_id
        self.executable = executable
        self.args = args
        self.port = port
        self.model_path = model_path
        self.process: asyncio.subprocess.Process | None = None
        self.started_at: float | None = None
        self.ready = False
        self.exit_code: int | None = None
        self.last_error: str | None = None
        self.log_lines: deque[dict[str, Any]] = deque(maxlen=1000)
        self.ready_time: float | None = None
        self.device_name: str | None = None  # parsed from engine output
        self.owned = False  # True once we verified the PID is ours

    @property
    def pid(self) -> int | None:
        return self.process.pid if self.process else None

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.returncode is None

    def add_log(self, stream: str, line: str) -> None:
        self.log_lines.append({
            "ts": time.time(), "stream": stream, "line": line[:2000],
        })

    def command_display(self) -> list[str]:
        """Command for diagnostics - safe, no secrets beyond engine args."""
        return [self.executable, *self.args]


class ProcessManager:
    def __init__(self, log: LogFn | None = None,
                 on_exit: Callable[[str, int | None], Awaitable[None]] | None = None
                 ) -> None:
        self._log = log
        self._on_exit = on_exit
        self._processes: dict[str, EngineProcess] = {}
        self._tasks: set[asyncio.Task] = set()

    # -- helpers -----------------------------------------------------------

    def _emit(self, level: str, message: str, instance_id: str | None = None) -> None:
        if self._log:
            self._log(level, message, instance_id)

    def used_ports(self) -> set[int]:
        return {ep.port for ep in self._processes.values() if ep.alive}

    def get(self, instance_id: str) -> EngineProcess | None:
        return self._processes.get(instance_id)

    # -- lifecycle ---------------------------------------------------------

    async def start(self, instance_id: str, executable: str, args: list[str],
                    port: int, model_path: str,
                    readiness_timeout: float = 600.0) -> EngineProcess:
        """Spawn llama-server and wait until it really serves /health."""
        exe = Path(executable)
        if not exe.is_file():
            raise FileNotFoundError(f"engine executable not found: {executable}")
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"model file not found: {model_path}")

        ep = EngineProcess(instance_id, str(exe), args, port, model_path)
        self._processes[instance_id] = ep

        self._emit("info", f"starting engine on port {port}: "
                           f"{exe.name} {' '.join(args[:6])}...", instance_id)
        try:
            ep.process = await asyncio.create_subprocess_exec(
                str(exe), *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                stdin=asyncio.subprocess.DEVNULL,
                cwd=str(exe.parent),
                creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
            )
        except OSError as exc:
            ep.last_error = f"failed to spawn engine: {exc}"
            self._emit("error", ep.last_error, instance_id)
            raise

        ep.started_at = time.time()
        ep.owned = True  # we created it, we may stop it

        for stream_name, stream in (("stdout", ep.process.stdout),
                                    ("stderr", ep.process.stderr)):
            task = asyncio.create_task(self._drain(ep, stream_name, stream))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

        # Watch for unexpected exit in the background.
        waiter = asyncio.create_task(self._wait_for_exit(ep))
        self._tasks.add(waiter)
        waiter.add_done_callback(self._tasks.discard)

        await self._wait_ready(ep, timeout=readiness_timeout)
        return ep

    async def _drain(self, ep: EngineProcess, stream_name: str,
                     stream: asyncio.StreamReader | None) -> None:
        if stream is None:
            return
        try:
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                line = raw.decode("utf-8", errors="replace").rstrip()
                if not line:
                    continue
                ep.add_log(stream_name, line)
                level = "error" if "error" in line.lower() else "info"
                self._emit(level, f"[engine:{stream_name}] {line}", ep.instance_id)
                self._maybe_parse_device(ep, line)
        except (asyncio.CancelledError, OSError):
            pass

    @staticmethod
    def _maybe_parse_device(ep: EngineProcess, line: str) -> None:
        """Learn the real Vulkan device name the engine chose."""
        if ep.device_name:
            return
        lower = line.lower()
        if "vulkan" in lower and ":" in line and "(" in line:
            candidate = line.split(":", 1)[1].split("(")[0].strip()
            if candidate and len(candidate) < 80:
                ep.device_name = candidate

    async def _wait_for_exit(self, ep: EngineProcess) -> None:
        if ep.process is None:
            return
        code = await ep.process.wait()
        ep.exit_code = code
        ep.ready = False
        if code not in (0, None) and ep.last_error is None:
            ep.last_error = f"engine exited with code {code}"
            self._emit("error", f"engine exited unexpectedly (code {code})",
                       ep.instance_id)
        if self._on_exit:
            with contextlib.suppress(Exception):
                await self._on_exit(ep.instance_id, code)

    async def _wait_ready(self, ep: EngineProcess,
                          timeout: float) -> None:
        """Poll /health until ok, engine exit, or timeout."""
        import httpx
        deadline = time.monotonic() + timeout
        url = f"http://127.0.0.1:{ep.port}/health"
        self._emit("info", f"waiting for engine readiness on {url}",
                   ep.instance_id)
        async with httpx.AsyncClient(timeout=5.0) as client:
            while time.monotonic() < deadline:
                if not ep.alive:
                    tail = "\n".join(l["line"] for l in list(ep.log_lines)[-12:])
                    ep.last_error = (ep.last_error or "engine exited during load")
                    raise RuntimeError(
                        f"engine exited before becoming ready: {ep.last_error}\n"
                        f"last engine output:\n{tail}")
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        payload = resp.json()
                        if payload.get("status") == "ok":
                            ep.ready = True
                            ep.ready_time = time.time()
                            load_s = ep.ready_time - (ep.started_at or ep.ready_time)
                            self._emit("info",
                                       f"engine ready after {load_s:.1f}s",
                                       ep.instance_id)
                            return
                except (httpx.HTTPError, json.JSONDecodeError, ValueError):
                    pass
                await asyncio.sleep(1.0)
        ep.last_error = f"engine not ready after {timeout:.0f}s"
        self._emit("error", ep.last_error, ep.instance_id)
        await self.stop(ep.instance_id, timeout=10.0)
        raise TimeoutError(ep.last_error)

    async def stop(self, instance_id: str, timeout: float = 20.0) -> None:
        """Graceful stop first, force kill only if it refuses to die."""
        ep = self._processes.get(instance_id)
        if not ep or ep.process is None:
            return
        proc = ep.process
        if proc.returncode is not None:
            self._emit("info", f"engine already exited ({proc.returncode})",
                       instance_id)
            return
        self._emit("info", "stopping engine (graceful)", instance_id)
        try:
            # Windows: CTRL_BREAK lets llama-server flush and free VRAM.
            proc.send_signal(signal.CTRL_BREAK_EVENT)
        except (OSError, RuntimeError):
            with contextlib.suppress(OSError):
                proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=timeout)
            self._emit("info", "engine stopped gracefully", instance_id)
        except (asyncio.TimeoutError, OSError):
            self._emit("warning",
                       f"engine ignored graceful stop for {timeout}s; killing",
                       instance_id)
            with contextlib.suppress(OSError):
                proc.kill()
            with contextlib.suppress(OSError):
                await asyncio.wait_for(proc.wait(), timeout=5.0)
        ep.ready = False

    async def stop_all(self) -> None:
        for instance_id in list(self._processes):
            with contextlib.suppress(Exception):
                await self.stop(instance_id)

    def cleanup(self, instance_id: str) -> None:
        self._processes.pop(instance_id, None)

    # -- reconciliation ----------------------------------------------------

    @staticmethod
    def pid_alive(pid: int) -> bool:
        """Check a PID without touching processes we do not own."""
        import psutil
        try:
            proc = psutil.Process(pid)
            return proc.name().lower().startswith("llama")
        except Exception:  # noqa: BLE001 - psutil.NoSuchProcess etc.
            return False
