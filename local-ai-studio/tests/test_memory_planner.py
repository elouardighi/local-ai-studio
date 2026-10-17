"""Memory planner tests: KV cache maths, offload fractions, verdicts."""
from __future__ import annotations

from pathlib import Path

from backend.app.services import memory_planner as mp

GB = 1024 ** 3
MB = 1024 ** 2


def _model(**overrides) -> dict:
    base = {
        "size_bytes": 12 * GB,
        "block_count": 40,
        "embedding_length": 2048,
        "head_count": 16,
        "head_count_kv": 2,
        "key_length": 256,
        "value_length": 256,
        "full_attention_interval": 4,
        "ssm_state_size": 128,
        "ssm_group_count": 16,
        "mmproj_path": None,
    }
    base.update(overrides)
    return base


def _system(available_gb: float = 32) -> dict:
    """Fixed memory snapshot.

    Without this, verdict assertions depend on whatever the machine happens
    to be doing (a loaded model can leave < 1 GB free and flip any plan to
    ``risk_of_paging``); the live path is covered separately by
    ``test_plan_reports_real_system_memory``.
    """
    return {
        "ram_total_bytes": 64 * GB,
        "ram_available_bytes": int(available_gb * GB),
        "commit_limit_bytes": 64 * GB,
        "commit_available_bytes": 60 * GB,
    }


def test_hybrid_kv_cache_only_counts_full_attention_layers():
    """qwen35moe-style hybrid: 40 blocks but only every 4th has attention."""
    kv = mp.estimate_kv_cache_bytes(_model(), ctx=8192)
    full_layers = 40 // 4  # = 10
    elems = 2 * (256 + 256)          # kv-heads x (key+value) dims
    expected = full_layers * 8192 * elems * 2  # f16
    assert kv["bytes"] == expected
    assert kv["confidence"] == "medium"
    assert "10 full-attention layers" in kv["detail"]


def test_kv_cache_scales_linearly_with_context():
    small = mp.estimate_kv_cache_bytes(_model(), ctx=4096)["bytes"]
    large = mp.estimate_kv_cache_bytes(_model(), ctx=8192)["bytes"]
    assert large == small * 2


def test_unknown_geometry_uses_conservative_heuristic():
    kv = mp.estimate_kv_cache_bytes({"block_count": 32}, ctx=2048)
    assert kv["confidence"] == "low"
    assert kv["bytes"] > 0


def test_full_offload_moves_all_weights_to_gpu():
    plan = mp.estimate(_model(), ctx=8192, ngl=40, system=_system())
    assert plan["gpu_weight_bytes"] == 12 * GB
    assert plan["cpu_weight_bytes"] == 0
    assert plan["verdict"] == "fits"


def test_partial_offload_reports_ram_use():
    plan = mp.estimate(_model(), ctx=8192, ngl=20, system=_system())
    assert plan["gpu_weight_bytes"] == 6 * GB
    assert plan["cpu_weight_bytes"] == 6 * GB
    assert plan["verdict"] == "partial_offload"
    assert any("live in RAM" in w for w in plan["warnings"])


def test_cpu_only_mode_warns():
    plan = mp.estimate(_model(), ctx=4096, ngl=0, system=_system())
    assert plan["gpu_weight_bytes"] == 0
    assert plan["verdict"] == "partial_offload"
    assert any("CPU-only" in w for w in plan["warnings"])


def test_verdict_reports_paging_risk_when_ram_is_short():
    """6.5 GB of CPU weights on a machine with 2 GB free must page."""
    plan = mp.estimate(_model(), ctx=8192, ngl=20,
                       system=_system(available_gb=2))
    assert plan["verdict"] == "risk_of_paging"
    assert any("page to disk" in w for w in plan["warnings"])


def test_ngl_is_clamped_to_layer_count():
    plan = mp.estimate(_model(), ctx=4096, ngl=999)
    assert plan["gpu_layers"] == 40


def test_mmproj_adds_to_weight_bytes(tmp_path: Path):
    proj = tmp_path / "mmproj.gguf"
    proj.write_bytes(b"0" * (512 * MB))
    model = _model(mmproj_path=str(proj))
    with_proj = mp.estimate(model, ctx=4096, ngl=40, mmproj=True)
    without = mp.estimate(model, ctx=4096, ngl=40, mmproj=False)
    assert with_proj["weights_bytes"] == without["weights_bytes"] + 512 * MB


def test_recommend_ngl_never_exceeds_layers_and_accounts_for_kv():
    rec = mp.recommend_ngl(_model(), gpu_free_mb=4096, ctx=8192)
    assert 0 <= rec["ngl"] <= 40
    # A tiny VRAM budget must yield zero, not a negative or a crash.
    assert mp.recommend_ngl(_model(), gpu_free_mb=100, ctx=8192)["ngl"] == 0


def test_recommend_ngl_full_vram_fits_everything():
    rec = mp.recommend_ngl(_model(), gpu_free_mb=24000, ctx=8192)
    assert rec["ngl"] == 40


def test_plan_reports_real_system_memory():
    plan = mp.estimate(_model(), ctx=4096, ngl=40)
    assert plan["system"]["ram_total_bytes"] > 0
    assert plan["note"].startswith("Estimates only")
