"""Context handling: never ask for more than the model can use.

llama.cpp clamps the context to the model's trained maximum but allocates
the KV cache for the size it was *asked* for. Asking a 2048-token model for
100 000 tokens therefore burns gigabytes of VRAM on a cache that can never
be used, and leaves the instance record claiming a context the engine does
not have.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings, SettingsStore
from backend.app.main import create_app
from backend.app.services.instance_manager import clamp_context
from tests.conftest import write_gguf


def test_the_context_is_capped_at_the_model_maximum():
    assert clamp_context(100_000, {"context_length": 2048}) == 2048
    assert clamp_context(2048, {"context_length": 2048}) == 2048
    assert clamp_context(1000, {"context_length": 2048}) == 1000
    # A model that does not say is left alone.
    assert clamp_context(100_000, {}) == 100_000
    # A silly request is still given a floor.
    assert clamp_context(10, {"context_length": 2048}) == 512


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    # A model with a small trained window, like TinyLlama.
    write_gguf(model_dir / "Tiny-Q4_K_M.gguf", {
        "general.architecture": "llama",
        "general.name": "Tiny",
        "general.file_type": 15,
        "llama.block_count": 22,
        "llama.context_length": 2048,
    })
    store = SettingsStore(path=tmp_path / "settings.json")
    store.replace(Settings(model_dirs=[str(model_dir)],
                            converted_dir=str(tmp_path / "converted"),
                            default_ctx=100_000).model_dump())
    app = create_app(store)
    with TestClient(app) as test_client:
        yield test_client


def test_the_recommended_context_never_exceeds_the_model_maximum(client: TestClient):
    models = client.get("/api/library/models").json()["models"]
    recommended = models[0]["recommended"]
    assert recommended["max_context"] == 2048
    assert recommended["context"] <= 2048
    # And the user is told why it is small.
    assert "warning" in recommended
    assert "larger context" in recommended["warning"]


def test_v1_models_reports_the_model_ceiling(client: TestClient):
    data = client.get("/v1/models").json()["data"][0]
    recommended = data["metadata"]["recommended"]
    assert recommended["max_context"] == 2048
    assert recommended["context"] <= 2048


def test_the_context_error_says_the_context_cannot_be_raised(client: TestClient):
    """The engine says "try increasing it"; the gateway says what is true."""
    from backend.app.api.openai_compat import _explain_context_error
    from backend.app.services.instance_manager import Instance

    instance = Instance(instance_id="i1", model_id="m", model_name="Tiny",
                        model_path="C:/tiny.gguf", context=2048,
                        context_max=2048)
    payload = {"error": {
        "code": 400,
        "message": "request (5889 tokens) exceeds the available context "
                   "size (2048 tokens), try increasing it",
        "type": "exceed_context_size_error",
        "n_prompt_tokens": 5889, "n_ctx": 2048,
    }}
    out = _explain_context_error(payload, instance)
    assert out["error"]["model_max_context"] == 2048
    assert "cannot be raised" in out["error"]["hint"]
    assert "larger context" in out["error"]["hint"]


def test_other_errors_are_left_alone(client: TestClient):
    from backend.app.api.openai_compat import _explain_context_error
    from backend.app.services.instance_manager import Instance

    instance = Instance(instance_id="i1", model_id="m", model_name="Tiny",
                        model_path="C:/tiny.gguf", context=2048,
                        context_max=2048)
    payload = {"error": {"message": "something else entirely"}}
    assert _explain_context_error(payload, instance) is payload
    assert "hint" not in payload["error"]


# ---------------------------------------------------------------------------
# The planner has to know about VRAM
# ---------------------------------------------------------------------------

def _model(**overrides):
    base = {
        "size_bytes": 12 * 1024 ** 3,
        "block_count": 40,
        "embedding_length": 2048,
        "head_count": 16,
        "head_count_kv": 8,
        "key_length": 256,
        "value_length": 256,
        "full_attention_interval": 4,
    }
    base.update(overrides)
    return base


def test_a_plan_that_does_not_fit_in_vram_is_refused():
    """The planner used to report the VRAM need but never judge it."""
    from backend.app.services import memory_planner as mp

    plan = mp.estimate(_model(), ctx=100_000, ngl=40, gpu_free_mb=8000)
    assert plan["verdict"] == "risk_of_paging"
    assert any("VRAM" in w for w in plan["warnings"])
    # And it says what to do about it.
    assert any("smaller context" in w for w in plan["warnings"])


def test_a_plan_that_fits_is_accepted():
    from backend.app.services import memory_planner as mp

    plan = mp.estimate(_model(), ctx=8192, ngl=40, gpu_free_mb=16_000)
    assert plan["verdict"] == "fits"


def test_the_recommendation_and_the_plan_agree():
    """What the recommendation offers must pass the plan's own check."""
    from backend.app.services import memory_planner as mp

    free = 16_000.0
    rec = mp.recommend_context(_model(), free)
    plan = mp.estimate(_model(), ctx=rec["context"], ngl=40, gpu_free_mb=free)
    assert plan["verdict"] != "risk_of_paging"


def test_the_recommendation_makes_room_for_the_projector(tmp_path: Path):
    """The projector is 800 MB of weights and has to be counted."""
    from backend.app.services import memory_planner as mp

    proj = tmp_path / "mmproj.gguf"
    proj.write_bytes(b"0" * (860 * 1024 * 1024))
    model = _model(mmproj_path=str(proj))

    without = mp.recommend_context(model, 16_000.0)
    with_proj = mp.recommend_context(model, 16_000.0, mmproj=True)
    assert with_proj["context"] < without["context"]

    # And the plan agrees with the recommendation when the projector counts.
    plan = mp.estimate(model, ctx=with_proj["context"], ngl=40,
                       mmproj=True, gpu_free_mb=16_000.0)
    assert plan["verdict"] != "risk_of_paging"
