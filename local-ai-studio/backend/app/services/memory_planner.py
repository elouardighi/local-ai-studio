"""Memory planner: estimates what a model needs and where it should live.

Everything here is an **estimate** built from GGUF metadata and current system
measurements. Measured values (engine-reported RSS, Windows GPU perf counters)
are collected separately by the resource monitor and must never be confused
with these numbers - the API labels them ``estimated`` vs ``measured``.

What is modelled
----------------
* weight bytes (from the GGUF file size, which is authoritative)
* KV cache, which for hybrid architectures (e.g. ``qwen35moe``) only exists in
  full-attention layers - linear/SSM layers keep a small fixed state instead
* compute buffers (a rough function of batch size and hidden size)
* the Vulkan driver overhead on top of the model allocation

What is deliberately *not* claimed
----------------------------------
The planner never pretends the Windows pagefile makes a model runnable. If the
plan spills past physical RAM the result says so in plain words.
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import subprocess
from typing import Any

GB = 1024 ** 3
MB = 1024 ** 2

# Fixed overhead the Vulkan driver and runtime need on top of model memory.
DRIVER_OVERHEAD_MB = 384
# Rough compute-buffer estimate: fraction of weights + hidden-size term.
COMPUTE_BUFFER_FRACTION = 0.05
COMPUTE_BUFFER_FIXED_MB = 128


def get_system_memory() -> dict[str, Any]:
    """Physical RAM + Windows committed-memory figures (all measured)."""
    import psutil

    vm = psutil.virtual_memory()
    # GlobalMemoryStatusEx gives the commit limit, which psutil does not.
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", wt.DWORD),
            ("dwMemoryLoad", wt.DWORD),
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
    commit = {}
    if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        commit = {
            "commit_limit_bytes": status.ullTotalPageFile,
            "commit_available_bytes": status.ullAvailPageFile,
            "commit_used_bytes": status.ullTotalPageFile - status.ullAvailPageFile,
        }
    return {
        "ram_total_bytes": vm.total,
        "ram_available_bytes": vm.available,
        "ram_used_percent": vm.percent,
        **commit,
    }


def estimate_kv_cache_bytes(model: dict[str, Any], ctx: int) -> dict[str, Any]:
    """Estimate KV cache bytes, handling hybrid linear-attention models."""
    blocks = int(model.get("block_count") or 0)
    interval = model.get("full_attention_interval")
    head_count_kv = int(model.get("head_count_kv") or model.get("head_count") or 0)
    key_len = int(model.get("key_length") or 0)
    value_len = int(model.get("value_length") or 0)
    state_size = model.get("ssm_state_size")
    groups = int(model.get("ssm_group_count") or 0)

    if not blocks or not head_count_kv or not (key_len or value_len):
        # Unknown geometry: fall back to a conservative per-token heuristic.
        per_token = 2 * blocks * 1024 * 2  # 2 (K+V) * layers * 1k elems * f16
        return {
            "bytes": int(per_token * ctx),
            "confidence": "low",
            "detail": "geometry unknown; conservative heuristic used",
        }

    if interval and blocks:
        full_layers = max(1, -(-blocks // int(interval)))  # ceil division
        linear_layers = blocks - full_layers
    else:
        full_layers, linear_layers = blocks, 0

    elems_per_token = head_count_kv * (key_len + value_len)
    kv_bytes = full_layers * ctx * elems_per_token * 2  # f16

    detail = (f"{full_layers} full-attention layers x {ctx} tokens x "
              f"{head_count_kv} kv-heads x ({key_len}+{value_len}) dims, f16")
    if linear_layers and state_size:
        # Hybrid SSM/linear layers keep a tiny fixed state per sequence.
        state_bytes = linear_layers * int(state_size) * max(groups, 1) * 4
        detail += (f"; {linear_layers} linear layers keep a fixed "
                   f"{state_bytes / MB:.1f} MB state")
    return {"bytes": int(kv_bytes), "confidence": "medium", "detail": detail}


def estimate(model: dict[str, Any], ctx: int, ngl: int,
             batch: int = 512, kv_on_gpu: bool = True,
             mmproj: bool = False,
             system: dict[str, Any] | None = None,
             gpu_free_mb: float | None = None) -> dict[str, Any]:
    """Produce a full memory plan for loading ``model`` with ``ctx``/``ngl``.

    ``system`` overrides the live memory snapshot (tests pass a fixed one so
    verdicts don't depend on whatever else the machine is doing); production
    callers omit it and get measured values from :func:`get_system_memory`.

    ``gpu_free_mb`` is the VRAM that is actually free. Without it the plan
    reports what the GPU would need but cannot say whether it fits - and a
    context that does not fit is the most common way a load fails.
    """
    weights = int(model.get("size_bytes") or 0)
    if mmproj and model.get("mmproj_path"):
        import os
        try:
            weights += os.path.getsize(model["mmproj_path"])
        except OSError:
            pass

    blocks = int(model.get("block_count") or 0)
    hidden = int(model.get("embedding_length") or 0)
    ngl = max(0, min(ngl, blocks)) if blocks else max(0, ngl)

    kv = estimate_kv_cache_bytes(model, ctx)

    # Weights are assumed evenly spread across layers (best available model
    # without parsing the tensor index; MoE expert routing adds variance).
    layer_fraction = (ngl / blocks) if blocks else (1.0 if ngl else 0.0)
    gpu_weights = int(weights * layer_fraction)
    cpu_weights = weights - gpu_weights

    compute_mb = (COMPUTE_BUFFER_FIXED_MB
                  + COMPUTE_BUFFER_FRACTION * gpu_weights / MB
                  + batch * hidden * 4 / MB)

    gpu_total = gpu_weights / MB + kv["bytes"] / MB * (1 if kv_on_gpu else 0) \
        + compute_mb + DRIVER_OVERHEAD_MB
    ram_total = cpu_weights / MB + kv["bytes"] / MB * (0 if kv_on_gpu else 1) \
        + 512  # runtime, tokenizer, python-side buffers

    sysmem = system if system is not None else get_system_memory()
    ram_available_mb = float(sysmem.get("ram_available_bytes") or 0) / MB

    warnings: list[str] = []
    verdict = "fits"

    if cpu_weights > 0:
        verdict = "partial_offload"
        warnings.append(
            f"{cpu_weights / GB:.1f} GB of weights will live in RAM "
            f"(GPU layers {ngl}/{blocks})."
        )
    if ram_total > ram_available_mb:
        verdict = "risk_of_paging"
        warnings.append(
            f"Estimated RAM need ({ram_total / 1024:.1f} GB) exceeds currently "
            f"available RAM ({ram_available_mb / 1024:.1f} GB). Windows will "
            f"page to disk; inference may become extremely slow or fail."
        )
    if gpu_free_mb is not None and gpu_total > float(gpu_free_mb):
        verdict = "risk_of_paging"
        warnings.append(
            f"The plan needs about {gpu_total:,.0f} MB of VRAM but only "
            f"{float(gpu_free_mb):,.0f} MB is free. The KV cache grows with "
            f"the context, so a smaller context is the way to make this fit."
        )
    if ngl == 0 and weights > 0:
        warnings.append("CPU-only configuration selected.")

    return {
        "weights_bytes": weights,
        "gpu_weight_bytes": gpu_weights,
        "cpu_weight_bytes": cpu_weights,
        "kv_cache_bytes": kv["bytes"],
        "kv_cache_confidence": kv["confidence"],
        "kv_cache_detail": kv["detail"],
        "compute_buffer_mb": round(compute_mb, 1),
        "driver_overhead_mb": DRIVER_OVERHEAD_MB,
        "estimated_gpu_mb": round(gpu_total, 1),
        "estimated_ram_mb": round(ram_total, 1),
        "gpu_free_mb": gpu_free_mb,
        "gpu_layers": ngl,
        "total_layers": blocks,
        "context": ctx,
        "verdict": verdict,
        "warnings": warnings,
        "system": sysmem,
        "note": ("Estimates only. Measured values come from the engine and "
                 "Windows GPU counters after loading."),
    }


def recommend_ngl(model: dict[str, Any], gpu_free_mb: float,
                  ctx: int, kv_on_gpu: bool = True) -> dict[str, Any]:
    """Largest GPU layer count expected to fit in currently free VRAM."""
    blocks = int(model.get("block_count") or 0)
    weights = int(model.get("size_bytes") or 0)
    if not blocks or not weights:
        return {"ngl": 0, "reason": "model geometry unknown"}

    kv = estimate_kv_cache_bytes(model, ctx)
    budget = gpu_free_mb - DRIVER_OVERHEAD_MB
    if kv_on_gpu:
        budget -= kv["bytes"] / MB
    budget -= COMPUTE_BUFFER_FIXED_MB

    per_layer_mb = (weights / blocks) / MB
    if per_layer_mb <= 0:
        return {"ngl": 0, "reason": "model size unknown"}
    fit = int(budget / per_layer_mb)
    fit = max(0, min(fit, blocks))
    reason = (f"free VRAM {gpu_free_mb:.0f} MB - driver/KV/compute reserve; "
              f"~{per_layer_mb:.0f} MB per layer")
    return {"ngl": fit, "reason": reason, "total_layers": blocks}


def recommend_context(model: dict[str, Any], gpu_free_mb: float,
                       ngl: int | None = None,
                       mmproj: bool = False) -> dict[str, Any]:
    """The context length that actually fits in the free VRAM.

    A model advertises a context length - 32k, 128k, 150k - that says
    something about the *architecture*, not about the card it is going on.
    The KV cache grows with the context and has to live somewhere, so the
    real number is whatever the free VRAM can hold.

    This is the number a client should be told, otherwise it builds a
    prompt the engine will only reject.
    """
    weights = int(model.get("size_bytes") or 0)
    if mmproj and model.get("mmproj_path"):
        import os
        try:
            weights += os.path.getsize(model["mmproj_path"])
        except OSError:
            pass
    blocks = int(model.get("block_count") or 0)
    max_ctx = int(model.get("context_length") or 0)
    if not weights or not blocks:
        return {"context": max_ctx or 4096, "confidence": "low",
                "reason": "model geometry unknown; using the advertised "
                          "context"}

    # Everything the model needs besides the KV cache.
    ngl = blocks if ngl is None else max(0, min(int(ngl), blocks))
    layer_fraction = ngl / blocks
    gpu_weights_mb = weights * layer_fraction / MB
    hidden = int(model.get("embedding_length") or 0)
    compute_mb = (COMPUTE_BUFFER_FIXED_MB
                  + COMPUTE_BUFFER_FRACTION * gpu_weights_mb
                  + 512 * hidden * 4 / MB)
    fixed_mb = gpu_weights_mb + compute_mb + DRIVER_OVERHEAD_MB

    # The KV cache is linear in the context, so one sample gives the
    # per-token cost.
    sample = 4096
    kv = estimate_kv_cache_bytes(model, sample)
    per_token = kv["bytes"] / sample if sample else 0
    if per_token <= 0:
        return {"context": max_ctx or 4096, "confidence": "low",
                "reason": "KV cache size unknown; using the advertised "
                          "context"}

    available_mb = gpu_free_mb - fixed_mb
    if available_mb <= 0:
        return {"context": 2048, "confidence": "low",
                "reason": f"the weights alone need about {fixed_mb:,.0f} MB "
                          f"but only {gpu_free_mb:,.0f} MB of VRAM is free"}

    ctx = int(available_mb * MB / per_token)
    ctx = max(2048, ctx)
    ctx -= ctx % 1024                       # round down to a tidy number
    if max_ctx:
        ctx = min(ctx, max_ctx)
    return {
        "context": ctx,
        "confidence": kv.get("confidence", "low"),
        "reason": (f"about {fixed_mb:,.0f} MB for the weights and buffers, "
                   f"leaving {available_mb:,.0f} MB of VRAM for the KV cache"
                   + (f"; the model was trained for {max_ctx:,} tokens"
                      if max_ctx and ctx < max_ctx else "")),
    }
