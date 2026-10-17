"""Windows resource monitor: CPU, RAM, commit charge, pagefile, GPU.

GPU metrics come from the Windows PDH performance counters, which work on AMD
hardware (unlike NVML, which is NVIDIA-only):

* ``\\GPU Adapter Memory(*)\\Dedicated Usage`` - VRAM in use per adapter
* ``\\GPU Adapter Memory(*)\\Shared Usage`` - shared-system memory in use
* ``\\GPU Engine(*)\\Utilization Percentage`` - per-process GPU engine usage
  (3d / compute / copy queues), which lets us attribute GPU load to the exact
  llama-server PID we started

If a counter is unavailable the value is reported as ``None`` and surfaced as
"Unavailable" in the UI. Numbers are never invented or interpolated.
"""
from __future__ import annotations

import asyncio
import contextlib
import ctypes
import ctypes.wintypes as wt
import time
from collections import deque
from typing import Any, Callable

import psutil

PDH_FMT_DOUBLE = 0x00000200
PDH_CSTATUS_VALID_DATA = 0
PDH_CSTATUS_NEW_DATA = 1
ERROR_SUCCESS = 0


class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _fields_ = [("CStatus", wt.DWORD), ("doubleValue", ctypes.c_double)]


class PDH_FMT_COUNTERVALUE_ITEM(ctypes.Structure):
    _fields_ = [("szName", ctypes.c_wchar_p),
                ("FmtValue", PDH_FMT_COUNTERVALUE)]


GPU_MEMORY_DEDICATED = r"\GPU Adapter Memory(*)\Dedicated Usage"
GPU_MEMORY_SHARED = r"\GPU Adapter Memory(*)\Shared Usage"
GPU_ENGINE_UTIL = r"\GPU Engine(*)\Utilization Percentage"


class PdhGpuCounters:
    """Thin, failure-tolerant wrapper around the PDH API."""

    def __init__(self) -> None:
        self._pdh = ctypes.windll.pdh
        self._query = ctypes.c_void_p()
        self._counters: dict[str, ctypes.c_void_p] = {}
        self.available = False
        self.error: str | None = None
        self._open()

    def _open(self) -> None:
        status = self._pdh.PdhOpenQueryW(None, 0, ctypes.byref(self._query))
        if status != ERROR_SUCCESS:
            self.error = f"PdhOpenQuery failed: 0x{status:08x}"
            return
        for name, path in (("dedicated", GPU_MEMORY_DEDICATED),
                           ("shared", GPU_MEMORY_SHARED),
                           ("engine_util", GPU_ENGINE_UTIL)):
            counter = ctypes.c_void_p()
            status = self._pdh.PdhAddEnglishCounterW(
                self._query, path, 0, ctypes.byref(counter))
            if status != ERROR_SUCCESS:
                self.error = (f"PdhAddEnglishCounter({path}) failed: "
                              f"0x{status:08x}")
                return
            self._counters[name] = counter
        # Prime the counters; first sample is always baseline.
        self._pdh.PdhCollectQueryData(self._query)
        self.available = True

    def _array(self, key: str) -> dict[str, float]:
        if not self.available or key not in self._counters:
            return {}
        buf_size = wt.DWORD(0)
        count = wt.DWORD(0)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            self._counters[key], PDH_FMT_DOUBLE, ctypes.byref(buf_size),
            ctypes.byref(count), None)
        # PDH_MORE_DATA (0x800007D5) is the expected "need buffer" result.
        if buf_size.value == 0:
            return {}
        raw = ctypes.create_string_buffer(buf_size.value)
        status = self._pdh.PdhGetFormattedCounterArrayW(
            self._counters[key], PDH_FMT_DOUBLE, ctypes.byref(buf_size),
            ctypes.byref(count), raw)
        if status != ERROR_SUCCESS:
            return {}
        items = ctypes.cast(
            raw, ctypes.POINTER(PDH_FMT_COUNTERVALUE_ITEM))
        out: dict[str, float] = {}
        for i in range(count.value):
            item = items[i]
            if item.FmtValue.CStatus in (PDH_CSTATUS_VALID_DATA,
                                         PDH_CSTATUS_NEW_DATA):
                out[item.szName or ""] = float(item.FmtValue.doubleValue)
        return out

    def sample(self) -> dict[str, Any]:
        if not self.available:
            return {"available": False, "error": self.error}
        self._pdh.PdhCollectQueryData(self._query)
        dedicated = self._array("dedicated")
        shared = self._array("shared")
        engines = self._array("engine_util")

        adapters: dict[str, dict[str, Any]] = {}
        for name, mb in dedicated.items():
            luid = _luid_of(name)
            adapters.setdefault(luid, {"luid": luid})["dedicated_used_mb"] = \
                round(mb / 1024 / 1024, 1)
        for name, mb in shared.items():
            luid = _luid_of(name)
            adapters.setdefault(luid, {"luid": luid})["shared_used_mb"] = \
                round(mb / 1024 / 1024, 1)

        agg = _aggregate_engine_counters(engines)

        return {
            "available": True,
            "adapters": list(adapters.values()),
            "per_pid_utilization": agg["per_pid_utilization"],
            "per_pid_luid": agg["per_pid_luid"],
            "per_luid_utilization": agg["per_luid_utilization"],
        }

    def close(self) -> None:
        if self._query:
            with contextlib.suppress(Exception):
                self._pdh.PdhCloseQuery(self._query)
            self._query = ctypes.c_void_p()


def _luid_of(counter_name: str) -> str:
    """Extract the adapter LUID token from a PDH instance name."""
    if "luid_" in counter_name:
        start = counter_name.index("luid_")
        end = counter_name.find("_phys", start)
        return counter_name[start:end] if end > start else counter_name[start:]
    return counter_name


def _pid_of_engine(counter_name: str) -> int | None:
    """Extract the owning PID from a ``GPU Engine(pid_N_...)`` instance."""
    if not counter_name.startswith("pid_"):
        return None
    try:
        return int(counter_name.split("_", 2)[1])
    except (IndexError, ValueError):
        return None


def _aggregate_engine_counters(engines: dict[str, float]) -> dict[str, Any]:
    """Fold PDH GPU-engine counter instances into per-process views.

    Counter names look like
    ``pid_<pid>_luid_<luid>_phys_0_eng_2_engtype_compute 0``; several engine
    types share one process, so their percentages are summed. The returned
    ``per_pid_luid`` maps each process to the adapter it ran on (the LUID the
    engine instance actually rendered with).
    """
    per_pid: dict[int, float] = {}
    per_pid_luid: dict[int, str] = {}
    per_luid_util: dict[str, float] = {}
    for name, pct in engines.items():
        pid = _pid_of_engine(name)
        luid = _luid_of(name)
        if pid is not None:
            per_pid[pid] = per_pid.get(pid, 0.0) + pct
            if "luid_" in name and pid not in per_pid_luid:
                per_pid_luid[pid] = luid
        if "luid_" in name:
            per_luid_util[luid] = per_luid_util.get(luid, 0.0) + pct
    return {
        "per_pid_utilization": {str(k): round(v, 1)
                                for k, v in per_pid.items()},
        "per_pid_luid": {str(k): v for k, v in per_pid_luid.items()},
        "per_luid_utilization": {k: round(v, 1)
                                 for k, v in per_luid_util.items()},
    }


def get_commit_status() -> dict[str, Any]:
    """Windows commit charge + pagefile figures via GlobalMemoryStatusEx."""

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wt.DWORD), ("dwMemoryLoad", wt.DWORD),
            ("ullTotalPhys", ctypes.c_uint64),
            ("ullAvailPhys", ctypes.c_uint64),
            ("ullTotalPageFile", ctypes.c_uint64),
            ("ullAvailPageFile", ctypes.c_uint64),
            ("ullTotalVirtual", ctypes.c_uint64),
            ("ullAvailVirtual", ctypes.c_uint64),
            ("ullAvailExtendedVirtual", ctypes.c_uint64),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return {}
    total = status.ullTotalPageFile
    avail = status.ullAvailPageFile
    return {
        "commit_limit_bytes": total,
        "commit_used_bytes": total - avail,
        "commit_available_bytes": avail,
        "commit_used_percent": round((total - avail) / total * 100, 1)
        if total else None,
    }


class ResourceMonitor:
    """Samples the system on a fixed interval into a bounded history."""

    def __init__(self, get_settings, events, log,
                 history_max: int = 900) -> None:
        self._get_settings = get_settings
        self._events = events
        self._log = log
        self._history: deque[dict[str, Any]] = deque(maxlen=history_max)
        self._task: asyncio.Task | None = None
        self._pdh = PdhGpuCounters()
        self._gpu_total_mb: float | None = None
        self._gpu_name: str | None = None
        self._last: dict[str, Any] = {}
        self._instances_ref = None  # set by start(): a callable or a list
        psutil.cpu_percent(interval=None)  # prime
        if not self._pdh.available:
            self._log.warning("monitor",
                              f"GPU counters unavailable: {self._pdh.error}. "
                              f"GPU metrics will be reported as Unavailable.")
        else:
            self._log.info("monitor", "GPU perf counters active (PDH)")

    # -- static GPU facts ---------------------------------------------------

    def set_gpu_identity(self, name: str | None, total_mb: float | None
                         ) -> None:
        """Record the GPU the engine actually uses (from engine output)."""
        if name and not self._gpu_name:
            self._gpu_name = name
            self._log.info("monitor", f"GPU in use: {name} "
                           f"({total_mb:.0f} MB)" if total_mb else
                           f"GPU in use: {name}")
        if total_mb and not self._gpu_total_mb:
            self._gpu_total_mb = total_mb

    def probe_gpu_via_engine(self, engine_dir) -> None:
        """Ask the engine which devices exist (authoritative, read-only)."""
        import subprocess
        exe = engine_dir / "llama-server.exe"
        if not exe.is_file():
            return
        try:
            proc = subprocess.run([str(exe), "--list-devices"],
                                  capture_output=True, text=True, timeout=60,
                                  creationflags=0x08000000)
        except (OSError, subprocess.SubprocessError) as exc:
            self._log.warning("monitor", f"--list-devices failed: {exc}")
            return
        for line in (proc.stdout + "\n" + proc.stderr).splitlines():
            line = line.strip()
            if ":" in line and "(" in line:
                name = line.split(":", 1)[1].split("(")[0].strip()
                mem = line.split("(", 1)[1]
                total_mb = None
                if "MiB" in mem:
                    try:
                        total_mb = float(mem.split("MiB")[0].strip())
                    except ValueError:
                        total_mb = None
                self.set_gpu_identity(name, total_mb)
                break

    # -- sampling -----------------------------------------------------------

    def sample(self) -> dict[str, Any]:
        """One synchronous sample. Cheap enough (<10 ms) for a 1 s interval."""
        settings = self._get_settings()
        vm = psutil.virtual_memory()
        swap = psutil.swap_memory()
        commit = get_commit_status()
        gpu = self._pdh.sample()

        gpu_block: dict[str, Any] = {
            "available": bool(gpu.get("available")),
            "name": self._gpu_name,
            "total_mb": self._gpu_total_mb,
        }
        if gpu.get("available"):
            adapters = gpu.get("adapters", [])
            dedicated = sum(a.get("dedicated_used_mb", 0) for a in adapters)
            gpu_block.update({
                "dedicated_used_mb": round(dedicated, 1),
                "dedicated_free_mb": (round(self._gpu_total_mb - dedicated, 1)
                                      if self._gpu_total_mb else None),
                "adapters": adapters,
                "utilization_by_luid": gpu.get("per_luid_utilization", {}),
            })
        else:
            gpu_block["error"] = gpu.get("error") or "counters unavailable"

        # Per-instance attribution: engine PID -> GPU util + process memory.
        # `_instances_ref` is a callable returning the current instance dicts.
        instances: dict[str, Any] = {}
        refs = self._instances_ref
        ref_list: list = []
        try:
            ref_list = list(refs() if callable(refs) else (refs or []))
        except Exception:  # noqa: BLE001 - monitoring must never crash
            ref_list = []
        per_pid_util = gpu.get("per_pid_utilization", {}) if gpu.get(
            "available") else {}
        for inst in ref_list:
            pid = inst.get("pid")
            entry: dict[str, Any] = {"pid": pid, "state": inst["state"]}
            if pid:
                try:
                    p = psutil.Process(pid)
                    mem = p.memory_info()
                    entry.update({
                        "rss_mb": round(mem.rss / 1024 / 1024, 1),
                        "cpu_percent": p.cpu_percent(interval=None),
                        "gpu_util_percent": per_pid_util.get(str(pid)),
                        "luid": (gpu.get("per_pid_luid") or {}).get(str(pid)),
                    })
                except psutil.NoSuchProcess:
                    entry["rss_mb"] = None
                    entry["note"] = "process gone"
            instances[inst["instance_id"]] = entry

        disk = psutil.disk_io_counters() or None
        sample = {
            "ts": time.time(),
            "cpu_percent": psutil.cpu_percent(interval=None),
            "cpu_count": psutil.cpu_count(logical=True),
            "ram_total_bytes": vm.total,
            "ram_available_bytes": vm.available,
            "ram_used_percent": vm.percent,
            "pagefile_used_bytes": swap.used,
            "pagefile_total_bytes": swap.total,
            "pagefile_percent": swap.percent,
            **commit,
            "gpu": gpu_block,
            "instances": instances,
            "disk_read_mb_s": (round(disk.read_bytes / 1024 / 1024, 2)
                               if disk else None),
            "disk_write_mb_s": (round(disk.write_bytes / 1024 / 1024, 2)
                                if disk else None),
        }
        self._last = sample
        self._history.append(sample)
        self._events.publish("resources", sample)
        return sample

    # -- async loop ---------------------------------------------------------

    async def _loop(self) -> None:
        while True:
            try:
                interval = float(self._get_settings().sample_interval_seconds)
            except Exception:  # noqa: BLE001
                interval = 1.0
            try:
                await asyncio.to_thread(self.sample)
            except Exception as exc:  # noqa: BLE001 - monitoring must not die
                self._log.warning("monitor", f"sample failed: {exc}")
            await asyncio.sleep(max(0.5, interval))

    def start(self, instances_ref) -> None:
        """``instances_ref`` is a callable returning instance dicts."""
        self._instances_ref = instances_ref
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No loop yet (early startup / unit test): the reference is kept
            # and sampling still works when invoked manually.
            return
        if self._task is None or self._task.done():
            self._task = loop.create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        self._pdh.close()

    # -- queries ------------------------------------------------------------

    def last_sample(self) -> dict[str, Any]:
        return self._last

    def history(self, seconds: int | None = None) -> list[dict[str, Any]]:
        items = list(self._history)
        if seconds is None:
            return items
        cutoff = time.time() - seconds
        return [s for s in items if s["ts"] >= cutoff]
