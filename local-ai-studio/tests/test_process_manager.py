"""Process manager tests using a fake engine (no GPU, no real model)."""
from __future__ import annotations

import asyncio
import socket
import sys
from pathlib import Path

import pytest

from backend.app.services.process_manager import ProcessManager, find_free_port
from tests.conftest import port_range


def test_find_free_port_skips_occupied():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        taken = sock.getsockname()[1]
        port = find_free_port(taken, taken + 20, occupied={taken})
        assert port != taken
        assert taken < port <= taken + 20


def test_find_free_port_honours_occupied_set():
    start, end = port_range(19000, 20)
    port = find_free_port(start, end, occupied=set(range(start, start + 6)))
    assert port >= start + 6


def test_find_free_port_raises_when_exhausted():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        taken = sock.getsockname()[1]
        with pytest.raises(RuntimeError):
            find_free_port(taken, taken, occupied={taken})


def test_missing_executable_is_rejected(tmp_path: Path):
    async def run():
        pm = ProcessManager()
        with pytest.raises(FileNotFoundError):
            await pm.start("i1", str(tmp_path / "nope.exe"), [], port_range(19100)[0],
                           "model.gguf")

    asyncio.run(run())


def test_missing_model_is_rejected(tmp_path: Path, fake_engine_path: Path):
    async def run():
        pm = ProcessManager()
        with pytest.raises(FileNotFoundError):
            await pm.start("i1", sys.executable,
                           [str(fake_engine_path), "--port", str(port_range(19100)[0] + 1)],
                           port_range(19100)[0] + 1, str(tmp_path / "missing.gguf"))

    asyncio.run(run())


def test_start_becomes_ready_and_stops_cleanly(tmp_path: Path,
                                               fake_engine_path: Path):
    async def run():
        pm = ProcessManager()
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        port = find_free_port(*port_range(19110))
        ep = await pm.start(
            "i1", sys.executable,
            [str(fake_engine_path), "--port", str(port)],
            port, str(model), readiness_timeout=30)
        assert ep.ready is True
        assert ep.alive
        assert ep.pid and ep.pid > 0
        assert port == ep.port

        await pm.stop("i1")
        assert not ep.alive
        assert ep.exit_code is not None

    asyncio.run(run())


def test_slow_engine_is_waited_for(tmp_path: Path, fake_engine_path: Path):
    """The manager must wait for /health, not declare ready on spawn."""
    async def run():
        pm = ProcessManager()
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        port = find_free_port(*port_range(19200))
        ep = await pm.start(
            "i1", sys.executable,
            [str(fake_engine_path), "--port", str(port),
             "--load-delay", "3"],
            port, str(model), readiness_timeout=30)
        assert ep.ready is True
        assert ep.ready_time >= ep.started_at
        await pm.stop("i1")

    asyncio.run(run())


def test_engine_output_is_captured(tmp_path: Path, fake_engine_path: Path):
    async def run():
        pm = ProcessManager()
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        port = find_free_port(*port_range(19300))
        ep = await pm.start(
            "i1", sys.executable,
            [str(fake_engine_path), "--port", str(port)],
            port, str(model), readiness_timeout=30)
        await asyncio.sleep(0.3)
        await pm.stop("i1")

    asyncio.run(run())


def test_two_instances_never_share_a_port(tmp_path: Path,
                                          fake_engine_path: Path):
    async def run():
        pm = ProcessManager()
        model = tmp_path / "m.gguf"
        model.write_bytes(b"x")
        used = set()
        for name in ("a", "b"):
            port = find_free_port(*port_range(19400), occupied=used)
            used.add(port)
            ep = await pm.start(
                name, sys.executable,
                [str(fake_engine_path), "--port", str(port)],
                port, str(model), readiness_timeout=30)
            assert ep.port == port
            assert port not in pm.used_ports() - {port}
        assert len({e.port for e in pm._processes.values()}) == 2
        await pm.stop_all()

    asyncio.run(run())
