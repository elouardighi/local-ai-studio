"""Management API: model discovery, load/unload, instance lifecycle."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.context import AppContext
from ..services.memory_planner import estimate, recommend_ngl

router = APIRouter(prefix="/api")


def _ctx(request: Request) -> AppContext:
    return request.app.state.ctx


class LoadConfig(BaseModel):
    model_id: str
    context: int | None = Field(default=None, ge=512, le=1_048_576)
    gpu_mode: str = "auto"  # auto | max | manual | cpu
    gpu_layers: int | None = Field(default=None, ge=0, le=1000)
    kv_on_gpu: bool = True
    batch: int | None = Field(default=None, ge=1, le=65536)
    ubatch: int | None = Field(default=None, ge=1, le=65536)
    flash_attention: str | None = None
    profile: str = "manual"
    use_mmproj: bool = True
    allow_duplicate: bool = False
    unload_idle: bool = False
    force: bool = False


class UnloadRequest(BaseModel):
    instance_id: str | None = None
    model_id: str | None = None


@router.get("/models")
async def list_models(request: Request) -> Any:
    ctx = _ctx(request)
    models = []
    for m in ctx.scanner.list():
        inst = ctx.instances.find_running_for_model(m["id"])
        entry = dict(m)
        entry["loaded"] = bool(inst and inst.is_loaded)
        entry["instance_id"] = inst.instance_id if inst else None
        entry["instance_state"] = inst.state if inst else "unloaded"
        entry["last_loaded_at"] = inst.ready_at if inst else None
        models.append(entry)
    return {
        "models": models,
        "count": len(models),
        "last_scan": ctx.scanner.last_scan,
        "scan_errors": ctx.scanner.errors,
    }


@router.post("/models/scan")
async def rescan(request: Request) -> Any:
    ctx = _ctx(request)
    models = ctx.scanner.scan()
    return {"count": len(models), "last_scan": ctx.scanner.last_scan,
            "errors": ctx.scanner.errors}


@router.get("/models/{model_id}")
async def model_detail(model_id: str, request: Request) -> Any:
    ctx = _ctx(request)
    model = ctx.scanner.get(model_id)
    if model is None:
        return JSONResponse(status_code=404,
                            content={"error": {"message": "model not found",
                                               "type": "not_found"}})
    s = ctx.settings_store.settings
    return {
        "model": model,
        "estimates": {
            "full_gpu": estimate(model, s.default_ctx, model.get("block_count")
                                 or 999),
            "recommended": recommend_ngl(
                model, ctx.monitor.last_sample().get("gpu", {})
                .get("dedicated_free_mb") or 0.0, s.default_ctx),
        },
        "instance": (lambda i: i.__dict__ if i else None)(
            ctx.instances.find_running_for_model(model_id)),
    }


@router.post("/models/estimate")
async def estimate_load(request: Request, config: LoadConfig) -> Any:
    """Dry-run the memory plan without touching the engine."""
    ctx = _ctx(request)
    model = ctx.scanner.get(config.model_id) or ctx.scanner.find_by_alias_or_name(
        config.model_id)
    if model is None:
        # The model may simply not have been scanned yet - a folder was
        # imported, or a download had just finished. Try once more after a
        # fresh scan before giving up.
        ctx.scanner.scan()
        model = ctx.scanner.get(config.model_id) or ctx.scanner.find_by_alias_or_name(
            config.model_id)
    if model is None:
        return JSONResponse(status_code=404, content={"error": {
            "message": "model not found. It is not in any scanned folder - "
                       "press Rescan, or import the folder it lives in.",
            "type": "not_found"}})
    s = ctx.settings_store.settings
    ctx_len = config.context or s.default_ctx
    if config.gpu_mode == "cpu":
        ngl = 0
    elif config.gpu_mode == "manual":
        ngl = config.gpu_layers if config.gpu_layers is not None else s.default_ngl
    elif config.gpu_mode == "max":
        ngl = model.get("block_count") or s.default_ngl
    else:
        gpu_free = (ctx.monitor.last_sample().get("gpu", {})
                    .get("dedicated_free_mb") or 12000.0)
        ngl = recommend_ngl(model, gpu_free, ctx_len,
                            kv_on_gpu=config.kv_on_gpu)["ngl"]
    gpu_free = None
    sample = ctx.monitor.last_sample() or {}
    free = ((sample.get("gpu") or {}).get("dedicated_free_mb"))
    if free:
        gpu_free = float(free)
    return {
        "plan": estimate(model, ctx_len, ngl, batch=config.batch or s.default_batch,
                         kv_on_gpu=config.kv_on_gpu,
                         mmproj=bool(model.get("mmproj_path")) and config.use_mmproj,
                         gpu_free_mb=gpu_free),
        "gpu_layers": ngl,
        "context": ctx_len,
    }


@router.post("/models/load")
async def load_model(request: Request, config: LoadConfig) -> Any:
    ctx = _ctx(request)
    try:
        inst = await ctx.instances.load(config.model_id, config.model_dump(
            exclude_none=True, exclude={"model_id"}))
    except (ValueError, RuntimeError, MemoryError, FileNotFoundError,
            TimeoutError) as exc:
        return JSONResponse(status_code=409, content={
            "error": {"message": str(exc), "type": type(exc).__name__}})
    return {"instance": inst.__dict__}


@router.post("/models/unload")
async def unload_model(request: Request, req: UnloadRequest) -> Any:
    ctx = _ctx(request)
    instance_id = req.instance_id
    if instance_id is None and req.model_id:
        inst = ctx.instances.find_running_for_model(req.model_id)
        if inst is None:
            return JSONResponse(status_code=404, content={"error": {
                "message": "no running instance for that model"}})
        instance_id = inst.instance_id
    if instance_id is None:
        return JSONResponse(status_code=400, content={"error": {
            "message": "instance_id or model_id is required"}})
    try:
        await ctx.instances.unload(instance_id)
    except (KeyError, RuntimeError) as exc:
        return JSONResponse(status_code=409, content={
            "error": {"message": str(exc)}})
    return {"unloaded": instance_id}


@router.get("/instances")
async def list_instances(request: Request) -> Any:
    ctx = _ctx(request)
    live = ctx.monitor.last_sample().get("instances", {})
    out = []
    for inst in ctx.instances.list():
        entry = dict(inst)
        entry["measurements"] = live.get(inst["instance_id"], {})
        out.append(entry)
    return {"instances": out, "count": len(out),
            "active_requests": ctx.tracker.active_count()}


@router.post("/instances/{instance_id}/unload")
async def unload_instance(instance_id: str, request: Request) -> Any:
    ctx = _ctx(request)
    try:
        await ctx.instances.unload(instance_id)
    except KeyError as exc:
        return JSONResponse(status_code=404,
                            content={"error": {"message": str(exc)}})
    except RuntimeError as exc:
        return JSONResponse(status_code=409,
                            content={"error": {"message": str(exc)}})
    return {"unloaded": instance_id}


@router.post("/instances/{instance_id}/restart")
async def restart_instance(instance_id: str, request: Request) -> Any:
    ctx = _ctx(request)
    try:
        inst = await ctx.instances.restart(instance_id)
    except (KeyError, RuntimeError, ValueError, FileNotFoundError,
            TimeoutError) as exc:
        return JSONResponse(status_code=409,
                            content={"error": {"message": str(exc)}})
    return {"instance": inst.__dict__}


@router.post("/instances/{instance_id}/cancel")
async def cancel_instance(instance_id: str, request: Request) -> Any:
    ctx = _ctx(request)
    inst = ctx.instances.get(instance_id)
    if inst is None:
        return JSONResponse(status_code=404,
                            content={"error": {"message": "unknown instance"}})
    await ctx.instances.cancel_load(instance_id)
    return {"cancelled": instance_id, "state": inst.state}


@router.get("/instances/{instance_id}/logs")
async def instance_logs(instance_id: str, request: Request,
                        limit: int = 200) -> Any:
    ctx = _ctx(request)
    ep = ctx.processes.get(instance_id)
    lines = list(ep.log_lines)[-limit:] if ep else []
    return {"instance_id": instance_id, "engine_lines": lines,
            "backend_logs": ctx.log.tail(limit=limit, instance_id=instance_id)}
