"""System status, live events (WebSocket + SSE), logs, requests, settings."""
from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

from fastapi import APIRouter, Query, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse

from ..services.context import AppContext

router = APIRouter(prefix="/api")


def _ctx(request: Request) -> AppContext:
    return request.app.state.ctx


@router.get("/system")
async def system(request: Request) -> Any:
    ctx = _ctx(request)
    return ctx.summary()


@router.get("/system/gpu-devices")
async def gpu_devices(request: Request) -> Any:
    """Vulkan devices as reported by the engine itself (authoritative)."""
    import subprocess
    from pathlib import Path
    ctx = _ctx(request)
    exe = Path(ctx.settings_store.settings.engine_path)
    if not exe.is_file():
        return JSONResponse(status_code=404, content={
            "error": {"message": f"engine not found: {exe}"}})
    try:
        proc = subprocess.run([str(exe), "--list-devices"],
                              capture_output=True, text=True, timeout=60,
                              creationflags=0x08000000)
    except (OSError, subprocess.SubprocessError) as exc:
        return JSONResponse(status_code=500,
                            content={"error": {"message": str(exc)}})
    return {"raw": (proc.stdout + proc.stderr).strip(),
            "detected": ctx.monitor._gpu_name,  # noqa: SLF001
            "total_mb": ctx.monitor._gpu_total_mb}  # noqa: SLF001


@router.get("/system/metrics")
async def metrics_history(request: Request, seconds: int | None = None
                          ) -> Any:
    ctx = _ctx(request)
    return {"samples": ctx.monitor.history(seconds),
            "interval": ctx.settings_store.settings.sample_interval_seconds}


@router.get("/logs")
async def logs(request: Request, limit: int = 200, level: str | None = None,
               component: str | None = None, search: str | None = None,
               instance_id: str | None = None) -> Any:
    ctx = _ctx(request)
    return {"lines": ctx.log.tail(limit=limit, level=level,
                                  component=component, search=search,
                                  instance_id=instance_id)}


@router.get("/requests")
async def requests(request: Request, limit: int = 100) -> Any:
    ctx = _ctx(request)
    return {"recent": ctx.tracker.recent(limit), "totals": ctx.tracker.totals()}


@router.get("/settings")
async def get_settings(request: Request) -> Any:
    ctx = _ctx(request)
    s = ctx.settings_store.settings
    return {"settings": s.to_public_dict(),
            "requires_restart": ["api_host", "api_port", "engine_path"]}


@router.put("/settings")
async def put_settings(request: Request) -> Any:
    ctx = _ctx(request)
    try:
        data = await request.json()
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={
            "error": {"message": "body is not JSON"}})
    try:
        updated = ctx.settings_store.update(data)
    except Exception as exc:  # pydantic ValidationError and ValueError
        return JSONResponse(status_code=422, content={
            "error": {"message": f"invalid settings: {exc}"}})
    ctx.log.info("settings", "settings updated: "
                 + ", ".join(sorted(data.keys())))
    return {"settings": updated.to_public_dict()}


@router.post("/settings/reset")
async def reset_settings(request: Request) -> Any:
    from ..config import Settings
    ctx = _ctx(request)
    updated = ctx.settings_store.replace(Settings().model_dump())
    ctx.log.warning("settings", "settings reset to defaults")
    return {"settings": updated.to_public_dict()}


# ---------------------------------------------------------------------------
# Live events
# ---------------------------------------------------------------------------

@router.websocket("/events")
async def events_ws(websocket: WebSocket) -> Any:
    await websocket.accept()
    ctx: AppContext = websocket.app.state.ctx
    # Send a snapshot so a fresh dashboard renders immediately.
    await websocket.send_json({
        "type": "snapshot",
        "data": {
            "recent": ctx.events.recent(50),
            "instances": ctx.instances.list(),
            "resources": ctx.monitor.last_sample(),
        },
    })

    async def _receiver() -> None:
        """Keep the receive side alive so disconnects are detected."""
        while True:
            await websocket.receive_text()

    rx = asyncio.create_task(_receiver())
    try:
        # Pump events out for as long as the browser stays connected.
        async for event in ctx.events.subscribe():
            await websocket.send_json(event)
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        rx.cancel()
        # The receiver may already have finished with WebSocketDisconnect
        # (browser closed the tab) rather than being cancelled; awaiting it
        # re-raises that, so suppress it too. Without this every dashboard
        # tab close dumped a traceback into the backend error log.
        with contextlib.suppress(asyncio.CancelledError,
                                 WebSocketDisconnect, RuntimeError):
            await rx


@router.get("/events")
async def events_sse(request: Request) -> Any:
    """Server-Sent Events fallback for clients that cannot use WebSocket."""
    ctx = _ctx(request)

    async def generator():
        yield f"data: {json.dumps({'type': 'hello'})}\n\n"
        try:
            async for event in ctx.events.subscribe():
                yield f"data: {json.dumps(event, default=str)}\n\n"
        except asyncio.CancelledError:
            return

    return StreamingResponse(generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache"})
