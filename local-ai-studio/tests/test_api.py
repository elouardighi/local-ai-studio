"""API tests: gateway routing, errors and management endpoints.

Uses FastAPI's TestClient with a mocked instance registry, so no engine and no
GPU are needed. The one exception is test_streaming_end_to_end, which drives
the real process manager against the fake engine.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings, SettingsStore
from backend.app.main import create_app
from backend.app.services.context import AppContext
from tests.conftest import write_gguf


@pytest.fixture
def client(tmp_path: Path, monkeypatch) -> TestClient:
    """App wired to a temp data dir with a synthetic model on disk."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    write_gguf(model_dir / "Test-Q4_K_M.gguf", {
        "general.architecture": "qwen35moe",
        "general.name": "Test Model",
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
    store.replace(Settings(model_dirs=[str(model_dir)],
                           engine_path=str(tmp_path / "no-engine.exe"),
                           converted_dir=str(tmp_path / "converted")).model_dump())

    app = create_app(store)
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_real_state(client: TestClient):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["service"] == "local-ai-studio"
    assert body["loaded_instances"] == 0


def test_v1_models_reflects_real_discovery(client: TestClient):
    res = client.get("/v1/models")
    assert res.status_code == 200
    data = res.json()["data"]
    assert len(data) == 1
    entry = data[0]
    assert entry["id"] == "Test Model"
    assert entry["metadata"]["architecture"] == "qwen35moe"
    assert entry["metadata"]["loaded"] is False
    assert entry["metadata"]["compatibility"] == "supported"


def test_api_models_endpoint_lists_records(client: TestClient):
    res = client.get("/api/models")
    body = res.json()
    assert body["count"] == 1
    assert body["models"][0]["format"] == "GGUF"
    assert body["models"][0]["instance_state"] == "unloaded"


def test_rescan_works(client: TestClient):
    res = client.post("/api/models/scan")
    assert res.status_code == 200
    assert res.json()["count"] == 1


def test_chat_completions_unknown_model_is_404(client: TestClient):
    res = client.post("/v1/chat/completions", json={
        "model": "does-not-exist",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert res.status_code == 404
    assert res.json()["error"]["type"] == "model_not_found"


def test_chat_completions_unloaded_model_explains_how_to_load(client: TestClient):
    res = client.post("/v1/chat/completions", json={
        "model": "Test Model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert res.status_code == 404
    err = res.json()["error"]
    assert err["type"] == "model_not_loaded"
    assert "/api/models/load" in err["message"]


def test_chat_completions_requires_messages(client: TestClient):
    res = client.post("/v1/chat/completions", json={"model": "Test Model"})
    assert res.status_code == 400
    assert res.json()["error"]["type"] == "invalid_request_error"


def test_invalid_json_body_is_rejected(client: TestClient):
    res = client.post("/v1/chat/completions",
                      content=b"{not json",
                      headers={"Content-Type": "application/json"})
    assert res.status_code == 400


def test_unsupported_endpoint_returns_clear_error(client: TestClient):
    res = client.get("/v1/does-not-exist")
    assert res.status_code == 404
    assert res.json()["error"]["type"] == "endpoint_not_found"


def test_unknown_top_level_path_is_404(client: TestClient):
    res = client.get("/whatever")
    assert res.status_code == 404


def test_metrics_without_instance_returns_app_counters(client: TestClient):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "local_ai_studio_requests_total" in res.text


def test_embeddings_refuses_chat_model(client: TestClient):
    """Honesty check: do not silently return meaningless vectors."""
    res = client.post("/v1/embeddings", json={
        "model": "Test Model", "input": "hello"})
    # Either 404 (not loaded) or 400 (not embedding capable) is honest.
    assert res.status_code in (400, 404)
    assert res.json()["error"]["type"] in (
        "model_not_loaded", "model_not_embedding_capable", "model_not_found")


def test_settings_roundtrip_and_key_masking(client: TestClient):
    res = client.put("/api/settings", json={"api_key": "secret123",
                                            "api_port": 1234})
    assert res.status_code == 200
    assert res.json()["settings"]["api_key"] == "***set***"

    # Sending the masked value back must not clobber the stored secret.
    res = client.put("/api/settings", json={"api_key": "***set***",
                                            "default_ctx": 4096})
    assert res.status_code == 200
    assert res.json()["settings"]["default_ctx"] == 4096

    store: SettingsStore = client.app.state.ctx.settings_store
    assert store.settings.api_key == "secret123"


def test_settings_reject_invalid_values(client: TestClient):
    res = client.put("/api/settings", json={"api_port": 99999})
    assert res.status_code == 422
    res = client.put("/api/settings", json={"engine_backend": "metal"})
    assert res.status_code == 422


def test_advanced_args_allowlist_blocks_injection(client: TestClient):
    res = client.put("/api/settings", json={
        "advanced_args": ["--api-key", "stolen"]})
    assert res.status_code == 422
    assert "not allowed" in res.json()["error"]["message"]


def test_api_key_enforced_when_configured(client: TestClient):
    client.put("/api/settings", json={"api_key": "sekrit"})
    assert client.get("/v1/models").status_code == 401
    ok = client.get("/v1/models", headers={"Authorization": "Bearer sekrit"})
    assert ok.status_code == 200
    # The management API stays open on loopback.
    assert client.get("/api/system").status_code == 200


def test_system_endpoint_reports_real_facts(client: TestClient):
    body = client.get("/api/system").json()
    assert body["status"] == "ok"
    assert body["models_discovered"] == 1
    assert body["system"]["ram_total_bytes"] > 0


def test_instances_endpoint_starts_empty(client: TestClient):
    body = client.get("/api/instances").json()
    assert body["count"] == 0


def test_load_missing_model_returns_409(client: TestClient):
    res = client.post("/api/models/load", json={"model_id": "nope"})
    assert res.status_code == 409
    assert "not found" in res.json()["error"]["message"]


def test_load_with_broken_engine_fails_honestly(client: TestClient):
    """A missing engine binary must surface as a clear error, not a hang."""
    model = client.get("/api/models").json()["models"][0]
    res = client.post("/api/models/load", json={"model_id": model["id"]})
    assert res.status_code == 409
    assert "engine executable missing" in res.json()["error"]["message"]


def test_logs_endpoint_filters(client: TestClient):
    body = client.get("/api/logs?limit=10").json()
    assert "lines" in body
    assert all("message" in line for line in body["lines"])


def test_requests_endpoint_reports_totals(client: TestClient):
    body = client.get("/api/requests").json()
    assert body["totals"]["requests"] >= 0


def test_settings_reset_restores_defaults(client: TestClient):
    client.put("/api/settings", json={"default_ctx": 4096})
    res = client.post("/api/settings/reset")
    assert res.status_code == 200
    assert res.json()["settings"]["default_ctx"] == 8192
