"""OpenAI-compatible API gateway on the configured host/port.

Routes requests to the *real* loaded engine instance that owns them. There is
no fake model list and no simulated generation: every response body originates
from llama-server.

Error policy (always JSON, always honest about state):
* model unknown to the registry      -> 404 ``model_not_found``
* known but not loaded               -> 404 ``model_not_loaded``
* currently loading / unloading      -> 409 ``model_busy`` with the state
* engine crashed / failed            -> 502 ``engine_failed``
* no API key configured but one sent -> ignored (local tool)
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse

from ..services.context import AppContext
from ..services.instance_manager import Instance

router = APIRouter()


def _ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def _error(status: int, code: str, message: str,
           extra: dict[str, Any] | None = None) -> JSONResponse:
    payload = {"error": {"message": message, "type": code,
                         "code": code}}
    if extra:
        payload["error"].update(extra)
    return JSONResponse(status_code=status, content=payload)


def _engine_base(instance: Instance) -> str:
    return f"http://127.0.0.1:{instance.port}"


def _explain_context_error(payload: Any, instance: Instance) -> Any:
    """Make an "exceeds the context size" error honest.

    The engine says "try increasing it". For a model with a small trained
    window - TinyLlama, for one - that advice cannot be followed: the
    context cannot be raised. The gateway knows the model's real ceiling,
    so it says what is actually true and what to do about it.
    """
    if not isinstance(payload, dict):
        return payload
    error = payload.get("error")
    if not isinstance(error, dict):
        return payload
    kind = str(error.get("type") or "")
    message = str(error.get("message") or "")
    if "exceeds the available context size" not in message \
            and kind != "exceed_context_size_error":
        return payload
    maximum = getattr(instance, "context_max", None)
    error["model_max_context"] = maximum
    error["instance_context"] = getattr(instance, "context", None)
    if maximum and getattr(instance, "context", 0) >= maximum:
        error["hint"] = (
            f"this model was trained with a {maximum:,} token window, so the "
            f"context cannot be raised. A request of "
            f"{error.get('n_prompt_tokens', '?')} tokens cannot be served by "
            f"it at all - and a long conversation cannot even be summarised "
            f"by it. Use a model with a larger context for agent work.")
    else:
        error["hint"] = (
            f"this instance runs with a {getattr(instance, 'context', '?')} "
            f"token context. Load it with a larger context, or use a model "
            f"with a bigger trained window.")
    return payload


async def _resolve(request: Request, model_name: str | None
                   ) -> tuple[Instance | None, JSONResponse | None]:
    """Resolve a model name to a ready instance or an honest error.

    When ``auto_load_on_request`` is enabled the gateway loads the model and
    *waits* for it (LM Studio-style), so OpenAI clients can name any model on
    disk and simply use it. Failures come back as honest 503s.
    """
    ctx = _ctx(request)
    inst = ctx.instances.find_by_alias(model_name or "")
    if inst is not None:
        return inst, None

    # Distinguish "unknown" from "known but not loaded" for a useful message.
    known = ctx.scanner.find_by_alias_or_name(model_name or "")
    if known is None:
        return None, _error(
            404, "model_not_found",
            f"no model named '{model_name}' is registered. "
            f"GET /v1/models lists available models.")

    running = ctx.instances.find_running_for_model(known["id"])
    if running is not None and not running.is_loaded:
        if not ctx.settings_store.settings.auto_load_on_request:
            return None, _error(
                409, "model_busy",
                f"model '{model_name}' is {running.state}; retry shortly",
                {"state": running.state, "instance_id": running.instance_id})
        # Another request already started the load: wait for it, don't 409.
        try:
            inst = await ctx.instances.wait_until_ready(known["id"])
            return inst, None
        except Exception as exc:  # noqa: BLE001 - turned into an honest 503
            return None, _error(
                503, "model_load_failed",
                f"auto-load of '{model_name}' failed: {exc}",
                {"model_id": known["id"]})

    if ctx.settings_store.settings.auto_load_on_request:
        # `load()` awaits the full lifecycle; `unload_idle` evicts an idle
        # model when the concurrency limit would otherwise refuse the switch.
        try:
            inst = await ctx.instances.load(known["id"],
                                            {"unload_idle": True})
            return inst, None
        except Exception as exc:  # noqa: BLE001 - turned into an honest 503
            return None, _error(
                503, "model_load_failed",
                f"auto-load of '{model_name}' failed: {exc}",
                {"model_id": known["id"]})

    return None, _error(
        404, "model_not_loaded",
        f"model '{model_name}' exists on disk but is not loaded. "
        f"Load it from the dashboard or POST /api/models/load.",
        {"model_id": known["id"]})


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health")
async def health(request: Request) -> Any:
    ctx = _ctx(request)
    loaded = ctx.instances.loaded()
    engines_ok = all(
        ctx.processes.get(i.instance_id) and
        ctx.processes.get(i.instance_id).alive
        for i in loaded) if loaded else True
    status = "ok" if engines_ok else "degraded"
    return JSONResponse(
        status_code=200 if engines_ok else 503,
        content={
            "status": status,
            "service": "local-ai-studio",
            "loaded_instances": len(loaded),
            "active_requests": ctx.tracker.active_count(),
        })


@router.get("/v1/health")
async def v1_health(request: Request) -> Any:
    return await health(request)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@router.get("/v1/models")
async def list_models(request: Request) -> Any:
    """Reflect real registered models (discovered + loaded state).

    The ``metadata`` block carries what a client needs to talk to the model
    properly: the real context length the model can use on this machine
    (which is what fits in the free VRAM, not what the model advertises),
    the output budget, and the load state.
    """
    ctx = _ctx(request)
    data = []
    for model in ctx.library.all_models():
        inst = ctx.instances.find_running_for_model(model["id"])
        recommended = model.get("recommended") or {}
        entry = {
            "id": model["name"],
            "object": "model",
            "created": int(model.get("created_at") or 0),
            "owned_by": "local",
            "metadata": {
                "model_id": model["id"],
                "path": model["path"],
                "format": model["format"],
                "architecture": model["architecture"],
                "quantization": model["quantization"],
                "size_human": model["size_human"],
                "display_name": model["name"],
                "loaded": bool(inst and inst.is_loaded),
                "state": inst.state if inst else "unloaded",
                "instance_id": inst.instance_id if inst else None,
                "compatibility": model["compatibility"]["status"],
                "recommended": {
                    "context": recommended.get("context"),
                    "output": recommended.get("output"),
                    "max_context": recommended.get("max_context")
                    or model.get("context_length"),
                    "reason": recommended.get("reason"),
                    "warning": recommended.get("warning"),
                },
            },
        }
        data.append(entry)
    return {"object": "list", "data": data}


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

@router.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Any:
    ctx = _ctx(request)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _error(400, "invalid_request_error", "request body is not JSON")

    model_name = body.get("model")
    # Validate the request shape before resolving the model, so malformed
    # requests always get a 400 regardless of what is loaded.
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        return _error(400, "invalid_request_error",
                      "'messages' must be a non-empty array")
    instance, err = await _resolve(request, model_name)
    if err is not None:
        return err
    assert instance is not None

    stream = bool(body.get("stream", False))
    request_id = ctx.tracker.begin(
        endpoint="chat.completions", model=model_name,
        instance_id=instance.instance_id, streaming=stream)
    ctx.instances._mark_request(instance, +1)  # noqa: SLF001

    body.setdefault("stream", False)
    if stream and "stream_options" not in body:
        # Ask the engine for a final usage chunk so the request log and the
        # instance token counters stay accurate; clients get it forwarded
        # (OpenAI streams it the same way, with empty choices).
        body["stream_options"] = {"include_usage": True}
    url = f"{_engine_base(instance)}/v1/chat/completions"
    timeout = httpx.Timeout(30.0, read=ctx.settings_store.settings
                            .request_timeout_seconds)
    try:
        if stream:
            return StreamingResponse(
                _stream_upstream(request, url, body, instance, request_id),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache",
                         "X-Request-Id": request_id})
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body)
        return _finalize_json(request, resp, instance, request_id)
    except httpx.TimeoutException:
        ctx.tracker.finish(request_id, 504, error="engine timed out")
        return _error(504, "engine_timeout",
                      "the engine did not respond in time")
    except httpx.HTTPError as exc:
        ctx.tracker.finish(request_id, 502, error=str(exc))
        return _error(502, "engine_failed", f"engine request failed: {exc}")
    finally:
        ctx.instances._mark_request(instance, -1)  # noqa: SLF001


def _finalize_json(request: Request, resp: httpx.Response,
                   instance: Instance, request_id: str) -> Any:
    ctx = _ctx(request)
    if resp.status_code >= 400:
        try:
            payload = resp.json()
        except ValueError:
            payload = {"error": {"message": resp.text[:500]}}
        payload = _explain_context_error(payload, instance)
        ctx.tracker.finish(request_id, resp.status_code,
                           error=str((payload.get("error") or {})
                                     .get("message", ""))[:500])
        return JSONResponse(status_code=resp.status_code, content=payload)
    payload = resp.json()
    usage = payload.get("usage") or {}
    ctx.tracker.finish(request_id, 200,
                       prompt_tokens=usage.get("prompt_tokens"),
                       completion_tokens=usage.get("completion_tokens"))
    inst = ctx.instances.get(instance.instance_id)
    if inst and usage.get("completion_tokens"):
        inst.tokens_generated += usage["completion_tokens"]
        inst.touch()
    return JSONResponse(status_code=200, content=payload)


async def _stream_upstream(request: Request, url: str, body: dict[str, Any],
                           instance: Instance,
                           request_id: str) -> AsyncIterator[bytes]:
    """Proxy SSE from the engine, honouring client disconnects."""
    ctx = _ctx(request)
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    try:
        timeout = httpx.Timeout(30.0, read=ctx.settings_store.settings
                                .request_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            async with client.stream("POST", url, json=body) as upstream:
                if upstream.status_code >= 400:
                    raw = await upstream.aread()
                    try:
                        payload = json.loads(raw)
                    except ValueError:
                        payload = {"error": {"message": raw.decode(
                            errors="replace")[:500]}}
                    payload = _explain_context_error(payload, instance)
                    ctx.tracker.finish(request_id, upstream.status_code,
                                       error=str((payload.get("error") or {})
                                                 .get("message", ""))[:500])
                    yield (b"event: error\ndata: "
                           + json.dumps(payload).encode() + b"\n\n")
                    return
                async for line in upstream.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        yield b"data: [DONE]\n\n"
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        yield (data.encode() + b"\n")
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    choices = chunk.get("choices") or []
                    if choices and choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                    yield (b"data: " + json.dumps(chunk).encode() + b"\n\n")
    except asyncio.CancelledError:
        # Browser closed the tab / Stop pressed: cancel the upstream work.
        ctx.tracker.finish(request_id, 499, error="client disconnected")
        ctx.log.info("gateway", "stream cancelled by client",
                     instance_id=instance.instance_id)
        raise
    except httpx.HTTPError as exc:
        ctx.tracker.finish(request_id, 502, error=str(exc))
        yield (b"event: error\ndata: " +
               json.dumps({"error": {"message": str(exc)}}).encode() + b"\n\n")
    else:
        ctx.tracker.finish(
            request_id, 200,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"))
        inst = ctx.instances.get(instance.instance_id)
        if inst and usage.get("completion_tokens"):
            inst.tokens_generated += usage["completion_tokens"]
            inst.touch()


# ---------------------------------------------------------------------------
# Completions / embeddings
# ---------------------------------------------------------------------------

@router.post("/v1/completions")
async def completions(request: Request) -> Any:
    ctx = _ctx(request)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _error(400, "invalid_request_error", "request body is not JSON")
    instance, err = await _resolve(request, body.get("model"))
    if err is not None:
        return err
    assert instance is not None
    stream = bool(body.get("stream", False))
    request_id = ctx.tracker.begin("completions", body.get("model"),
                                   instance.instance_id, stream)
    ctx.instances._mark_request(instance, +1)  # noqa: SLF001
    url = f"{_engine_base(instance)}/v1/completions"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(
                30.0, read=ctx.settings_store.settings.request_timeout_seconds)
        ) as client:
            if stream:
                return StreamingResponse(
                    _stream_upstream(request, url, body, instance, request_id),
                    media_type="text/event-stream")
            resp = await client.post(url, json=body)
        return _finalize_json(request, resp, instance, request_id)
    except httpx.HTTPError as exc:
        ctx.tracker.finish(request_id, 502, error=str(exc))
        return _error(502, "engine_failed", str(exc))
    finally:
        ctx.instances._mark_request(instance, -1)  # noqa: SLF001


@router.post("/v1/embeddings")
async def embeddings(request: Request) -> Any:
    """Forwarded only to instances whose model actually supports embeddings."""
    ctx = _ctx(request)
    try:
        body = await request.json()
    except json.JSONDecodeError:
        return _error(400, "invalid_request_error", "request body is not JSON")
    instance, err = await _resolve(request, body.get("model"))
    if err is not None:
        return err
    assert instance is not None
    model = ctx.scanner.get(instance.model_id)
    if model and not model["capabilities"].get("embedding"):
        return _error(400, "model_not_embedding_capable",
                      f"'{instance.model_name}' is a chat model, not an "
                      f"embedding model. /v1/embeddings would return "
                      f"meaningless vectors.")
    url = f"{_engine_base(instance)}/v1/embeddings"
    request_id = ctx.tracker.begin("embeddings", body.get("model"),
                                   instance.instance_id, False)
    ctx.instances._mark_request(instance, +1)  # noqa: SLF001
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(url, json=body)
        return _finalize_json(request, resp, instance, request_id)
    except httpx.HTTPError as exc:
        ctx.tracker.finish(request_id, 502, error=str(exc))
        return _error(502, "engine_failed", str(exc))
    finally:
        ctx.instances._mark_request(instance, -1)  # noqa: SLF001


@router.get("/metrics")
async def metrics(request: Request) -> Any:
    """Prometheus text from the loaded engine, or app counters."""
    ctx = _ctx(request)
    loaded = ctx.instances.loaded()
    if loaded:
        inst = loaded[0]
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{_engine_base(inst)}/metrics")
            return StreamingResponse(iter([resp.content]),
                                     media_type="text/plain; version=0.0.4")
        except httpx.HTTPError as exc:
            return _error(502, "engine_failed", f"engine metrics: {exc}")
    totals = ctx.tracker.totals()
    lines = [
        "# HELP local_ai_studio_requests_total Total gateway requests",
        "# TYPE local_ai_studio_requests_total counter",
        f"local_ai_studio_requests_total {totals['requests']}",
        "# HELP local_ai_studio_tokens_completion_total Generated tokens",
        "# TYPE local_ai_studio_tokens_completion_total counter",
        f"local_ai_studio_tokens_completion_total {totals['tokens_completion']}",
    ]
    return StreamingResponse(iter(["\n".join(lines) + "\n"]),
                             media_type="text/plain; version=0.0.4")


@router.api_route("/{unsupported:path}", methods=["GET", "POST", "PUT",
                                                   "DELETE"])
async def unsupported(request: Request, unsupported: str) -> Any:
    """Catch-all: never pretend an endpoint exists."""
    known_prefixes = ("v1/", "health", "metrics")
    if unsupported.startswith(known_prefixes):
        return _error(404, "endpoint_not_found",
                      f"'/{unsupported}' is not implemented by Local AI "
                      f"Studio. See README for supported endpoints.")
    return _error(404, "not_found",
                  f"unknown API path '/{unsupported}'. The OpenAI-compatible "
                  f"API lives under /v1; the management API under /api.")
