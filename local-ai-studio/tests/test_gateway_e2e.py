"""End-to-end gateway test: load -> chat -> stream -> unload.

Drives the real instance manager, process manager and OpenAI-compatible
gateway against the fake engine from conftest, so the full request path is
exercised without a GPU or a real model file.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings, SettingsStore
from backend.app.main import create_app
from tests.conftest import port_range, write_gguf


def _build_app(tmp_path: Path, fake_engine_path: Path, load_delay: float = 0):
    """Create an app whose engine is the fake one, plus its model id."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = write_gguf(model_dir / "Fake-Q4_K_M.gguf", {
        "general.architecture": "qwen35moe",
        "general.name": "Fake Model",
        "general.file_type": 15,
        "qwen35moe.block_count": 40,
        "qwen35moe.context_length": 8192,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.attention.value_length": 256,
        "qwen35moe.full_attention_interval": 4,
        "tokenizer.ggml.model": "gpt2",
    })

    store = SettingsStore(path=tmp_path / "settings.json")
    store.replace(Settings(
        model_dirs=[str(model_dir)],
        engine_path=sys.executable,          # fake engine runs on python
        engine_port_start=port_range(19500)[0],
        engine_port_end=port_range(19500)[1],
        converted_dir=str(tmp_path / "converted"),
        default_ctx=2048,
    ).model_dump())

    app = create_app(store)

    delay = ["--load-delay", str(load_delay)] if load_delay else []

    def fake_args(self, inst, model, settings, config):
        return [str(fake_engine_path), "--port", str(inst.port), *delay]

    from backend.app.services.instance_manager import InstanceManager
    InstanceManager._build_args = fake_args

    from backend.app.services.model_scanner import model_id
    return app, model_id(model_path)


@pytest.fixture
def e2e_app_factory(tmp_path: Path):
    """Factory fixture for the async tests (fresh temp dir per call)."""
    counter = {"n": 0}

    def factory(fake_engine_path: Path, load_delay: float = 0.0):
        counter["n"] += 1
        sub = tmp_path / f"app{counter['n']}"
        sub.mkdir()
        return _build_app(sub, fake_engine_path, load_delay)

    return factory


def _make_e2e(tmp_path: Path, fake_engine_path: Path,
              extra_engine_args: tuple[str, ...] = ()):
    """One app instance backed by the fake engine (shared by e2e fixtures)."""
    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = write_gguf(model_dir / "Fake-Q4_K_M.gguf", {
        "general.architecture": "qwen35moe",
        "general.name": "Fake Model",
        "general.file_type": 15,
        "qwen35moe.block_count": 40,
        "qwen35moe.context_length": 8192,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.attention.value_length": 256,
        "qwen35moe.full_attention_interval": 4,
        "tokenizer.ggml.model": "gpt2",
    })

    store = SettingsStore(path=tmp_path / "settings.json")
    store.replace(Settings(
        model_dirs=[str(model_dir)],
        engine_path=sys.executable,          # fake engine runs on python
        engine_port_start=port_range(19500)[0],
        engine_port_end=port_range(19500)[1],
        converted_dir=str(tmp_path / "converted"),
        default_ctx=2048,
    ).model_dump())

    app = create_app(store)
    ctx: object = app.state.ctx

    # Replace the llama-server argument builder with a fake-engine invocation.
    def fake_args(self, inst, model, settings, config):
        return [str(fake_engine_path), "--port", str(inst.port),
                *extra_engine_args]

    from backend.app.services.instance_manager import InstanceManager
    original = InstanceManager._build_args
    InstanceManager._build_args = fake_args
    try:
        with TestClient(app) as client:
            yield client, ctx, model_path
    finally:
        InstanceManager._build_args = original


@pytest.fixture
def e2e(tmp_path: Path, fake_engine_path: Path):
    gen = _make_e2e(tmp_path, fake_engine_path)
    client, ctx, model_path = next(gen)
    try:
        yield client, ctx, model_path
    finally:
        gen.close()


@pytest.fixture
def e2e_record(tmp_path: Path, fake_engine_path: Path):
    """Like ``e2e``, but the fake engine records the body it receives."""
    record = tmp_path / "engine_request.json"
    gen = _make_e2e(tmp_path, fake_engine_path, ("--record", str(record)))
    client, ctx, _model_path = next(gen)
    try:
        yield client, ctx, record
    finally:
        gen.close()


def _load(e2e, **extra):
    client, ctx, _ = e2e
    model = client.get("/api/models").json()["models"][0]
    res = client.post("/api/models/load",
                      json={"model_id": model["id"], **extra})
    return res


def test_load_chat_stream_and_unload(e2e):
    client, ctx, _ = e2e

    res = _load(e2e)
    assert res.status_code == 200, res.text
    inst = res.json()["instance"]
    assert inst["state"] == "ready"
    assert inst["pid"] and inst["port"]
    assert inst["estimate"]["weights_bytes"] > 0
    assert inst["load_duration_s"] is not None

    # /v1/models must now report the model as loaded.
    v1 = client.get("/v1/models").json()["data"][0]
    assert v1["metadata"]["loaded"] is True

    # Non-streaming chat: real content from the engine.
    res = client.post("/v1/chat/completions", json={
        "model": "Fake Model",
        "messages": [{"role": "user", "content": "hello"}],
    })
    assert res.status_code == 200
    body = res.json()
    assert body["choices"][0]["message"]["content"] == "Hello from fake engine"
    assert body["usage"]["completion_tokens"] == 4

    # Streaming chat: SSE frames terminated by [DONE].
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "Fake Model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }) as res:
        assert res.status_code == 200
        raw = "".join(res.iter_text())
    assert "data: [DONE]" in raw
    assert "Hello" in raw

    # Request tracking saw real usage.
    totals = client.get("/api/requests").json()["totals"]
    assert totals["requests"] >= 2
    assert totals["tokens_completion"] >= 4

    # Unload stops the owned process.
    pid = inst["pid"]
    res = client.post("/api/models/unload", json={"instance_id": inst["instance_id"]})
    assert res.status_code == 200

    import psutil
    with pytest.raises(psutil.NoSuchProcess):
        psutil.Process(pid)

    state = [i for i in client.get("/api/instances").json()["instances"]
             if i["instance_id"] == inst["instance_id"]][0]
    assert state["state"] == "unloaded"
    assert state["pid"] is None


def test_streaming_requests_usage_and_honours_client_choice(e2e_record):
    """The gateway asks the engine for a usage chunk so token counts are real."""
    client, ctx, record = e2e_record
    assert _load(e2e_record).status_code == 200

    with client.stream("POST", "/v1/chat/completions", json={
        "model": "Fake Model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
    }) as res:
        raw = "".join(res.iter_text())
    assert "data: [DONE]" in raw
    assert "completion_tokens" in raw            # usage frame forwarded
    seen = json.loads(record.read_text(encoding="utf-8"))
    assert seen["stream_options"] == {"include_usage": True}
    totals = client.get("/api/requests").json()["totals"]
    assert totals["tokens_completion"] >= 4      # usage really got recorded

    # A client that explicitly opts out must not be overridden.
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "Fake Model",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": True,
        "stream_options": {"include_usage": False},
    }) as res:
        "".join(res.iter_text())
    seen = json.loads(record.read_text(encoding="utf-8"))
    assert seen["stream_options"] == {"include_usage": False}


def test_duplicate_load_is_refused(e2e):
    client, ctx, _ = e2e
    assert _load(e2e).status_code == 200
    res = _load(e2e)
    assert res.status_code == 409
    assert "already loaded" in res.json()["error"]["message"]
    client.post("/api/models/unload", json={
        "model_id": client.get("/api/models").json()["models"][0]["id"]})


@pytest.mark.asyncio
async def test_busy_state_reported_while_loading(e2e_app_factory, tmp_path,
                                                 fake_engine_path):
    """A model that is still loading must answer model_busy, not hang."""
    import asyncio
    import httpx

    app, model_id = e2e_app_factory(fake_engine_path, load_delay=3)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            load_task = asyncio.create_task(client.post("/api/models/load", json={
                "model_id": model_id, "force": True}))
            await asyncio.sleep(0.5)  # let the load start

            res = await client.post("/v1/chat/completions", json={
                "model": "Fake Model",
                "messages": [{"role": "user", "content": "x"}]})
            assert res.status_code == 409, res.text
            err = res.json()["error"]
            assert err["type"] in ("model_busy", "model_loading")
            assert err["state"] in ("validating", "estimating_memory",
                                    "queued", "starting_engine",
                                    "loading_model", "queued")

            load_res = await load_task
            assert load_res.status_code == 200, load_res.text

            # Once ready, the same request must succeed.
            res = await client.post("/v1/chat/completions", json={
                "model": "Fake Model",
                "messages": [{"role": "user", "content": "x"}]})
            assert res.status_code == 200
            inst = load_res.json()["instance"]
            await client.post(f"/api/instances/{inst['instance_id']}/unload")


@pytest.mark.asyncio
async def test_autoload_waits_until_ready_then_serves(e2e_app_factory,
                                                      tmp_path,
                                                      fake_engine_path):
    """With auto-load on, naming an unloaded model must load-and-wait,
    not 409 - this is what makes OpenAI clients (e.g. coding agents) work."""
    import httpx

    app, model_id = e2e_app_factory(fake_engine_path, load_delay=1.0)
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport,
                                     base_url="http://test") as client:
            res = await client.put("/api/settings",
                                   json={"auto_load_on_request": True})
            assert res.status_code == 200

            res = await client.post("/v1/chat/completions", json={
                "model": "Fake Model",
                "messages": [{"role": "user", "content": "x"}]})
            assert res.status_code == 200, res.text
            body = res.json()
            assert body["choices"][0]["message"]["content"]

            # The next request must reuse the now-loaded instance.
            res = await client.post("/v1/chat/completions", json={
                "model": "Fake Model",
                "messages": [{"role": "user", "content": "again"}]})
            assert res.status_code == 200, res.text


def test_stale_instance_record_is_reconciled_on_restart(tmp_path: Path,
                                                       fake_engine_path: Path):
    """A backend restart must never claim a dead model is still loaded."""
    import json
    state_file = tmp_path / "instances.json"
    state_file.write_text(json.dumps([{
        "instance_id": "stale1", "model_id": "m", "model_name": "Old",
        "model_path": "C:/gone.gguf", "state": "ready", "pid": 999999,
        "port": 18400, "created_at": time.time(), "gpu_layers": 40,
        "context": 8192, "batch": 512, "ubatch": 512,
        "flash_attention": "auto", "kv_on_gpu": True, "profile": "manual",
    }]), encoding="utf-8")

    store = SettingsStore(path=tmp_path / "settings.json")
    store.replace(Settings(model_dirs=[str(tmp_path)],
                           engine_path=sys.executable,
                           converted_dir=str(tmp_path / "converted")).model_dump())
    app = create_app(store)
    with TestClient(app):
        instances = app.state.ctx.instances.list()
        stale = [i for i in instances if i["instance_id"] == "stale1"]
        assert stale, "history should be preserved"
        assert stale[0]["state"] == "unloaded"
        assert stale[0]["pid"] is None
        assert "not running" in (stale[0]["last_error"] or "")
