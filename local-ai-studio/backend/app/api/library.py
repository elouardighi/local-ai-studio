"""Library API: search, import folders, preflight and optional downloads.

The download endpoints are the only place in the whole project that reaches
the internet, and they are guarded three times over: the feature is off
until it is switched on in Settings, a *plan* must be created and shown to
the user first, and starting the download needs an explicit approval flag.
Nothing that arrives this way is ever loaded or executed automatically.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from ..services.context import AppContext
from ..services.download_manager import ApprovalRequired, DownloadError
from ..services import file_removal
from ..services.hf_client import HuggingFaceError
from ..services.model_library import parse_size

router = APIRouter(prefix="/api/library")


def _ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def _error(status: int, message: str, kind: str = "error",
           extra: dict[str, Any] | None = None) -> JSONResponse:
    body: dict[str, Any] = {"error": {"message": message, "type": kind}}
    if extra:
        body["error"].update(extra)
    return JSONResponse(status_code=status, content=body)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

@router.get("/models")
async def search_models(request: Request,
                        q: str | None = None,
                        architecture: str | None = None,
                        quantization: str | None = None,
                        format: str | None = None,
                        min_size: str | None = None,
                        max_size: str | None = None,
                        integrity: str | None = None,
                        compatibility: str | None = None,
                        duplicates_only: bool = False,
                        sort: str = "size",
                        order: str = "desc") -> Any:
    """Search the model library.

    Sizes accept ``10gb``, ``500 MB``, ``1.5g`` or a plain byte count.
    """
    ctx = _ctx(request)
    library = ctx.library
    # A filter that was given but cannot be read is a mistake worth
    # reporting; a filter that was left out simply means "no bound".
    lo = hi = None
    if min_size is not None and str(min_size).strip():
        lo = parse_size(min_size)
        if lo is None:
            return _error(400, f"cannot read the minimum size {min_size!r} "
                               f"(try '10gb' or a byte count)", "invalid_request")
    if max_size is not None and str(max_size).strip():
        hi = parse_size(max_size)
        if hi is None:
            return _error(400, f"cannot read the maximum size {max_size!r} "
                               f"(try '10gb' or a byte count)", "invalid_request")
    if lo is not None and hi is not None and lo > hi:
        return _error(400, "the minimum size is larger than the maximum size",
                      "invalid_request")
    return library.models(
        query=q, architecture=architecture, quantization=quantization,
        format=format, min_size=lo, max_size=hi, integrity_status=integrity,
        compatibility=compatibility, duplicates_only=duplicates_only,
        sort=sort, order=order)


@router.get("/folders")
async def list_folders(request: Request) -> Any:
    ctx = _ctx(request)
    return {"folders": ctx.library.folders()}


@router.post("/folders")
async def import_folder(request: Request) -> Any:
    """Import a folder into the library and rescan it."""
    ctx = _ctx(request)
    body = await _json(request)
    path = body.get("path")
    if not path:
        return _error(400, "a folder path is required", "invalid_request")
    try:
        result = ctx.library.add_folder(str(path))
    except ValueError as exc:
        return _error(400, str(exc), "invalid_request")
    return result


@router.delete("/folders")
async def remove_folder(request: Request) -> Any:
    ctx = _ctx(request)
    body = await _json(request)
    path = body.get("path")
    if not path:
        return _error(400, "a folder path is required", "invalid_request")
    try:
        result = ctx.library.remove_folder(str(path))
    except ValueError as exc:
        return _error(400, str(exc), "invalid_request")
    return result


@router.get("/browse")
async def browse(request: Request, path: str | None = None) -> Any:
    """List the sub-folders of a path, for the folder picker."""
    ctx = _ctx(request)
    try:
        return ctx.library.browse(path)
    except ValueError as exc:
        return _error(400, str(exc), "invalid_request")


@router.post("/rescan")
async def rescan(request: Request) -> Any:
    ctx = _ctx(request)
    models = ctx.library.rescan()
    return {"count": len(models), "last_scan": ctx.scanner.last_scan,
            "scan_errors": ctx.scanner.errors}


@router.get("/models/{model_id}/delete-plan")
async def delete_plan(model_id: str, request: Request) -> Any:
    """What deleting this model would remove. Nothing is deleted here."""
    ctx = _ctx(request)
    try:
        return ctx.library.delete_plan(model_id)
    except ValueError as exc:
        return _error(404, str(exc), "not_found")


class DeleteRequest(BaseModel):
    confirm: bool = False
    delete_mmproj: bool = False
    permanent: bool = False


@router.delete("/models/{model_id}")
async def delete_model(model_id: str, request: Request,
                       confirm: bool = False) -> Any:
    """Delete a model from disk.

    The file goes to the Windows Recycle Bin unless ``permanent`` is set,
    and nothing happens at all unless the delete is confirmed. The path
    comes from the library's own record, never from the request.

    The confirmation may be sent in the body or as ``?confirm=true`` -
    some HTTP stacks drop the body of a DELETE.
    """
    ctx = _ctx(request)
    body = await _json(request)
    req = DeleteRequest(
        confirm=bool(body.get("confirm")) or confirm,
        delete_mmproj=bool(body.get("delete_mmproj")),
        permanent=bool(body.get("permanent")),
    )
    if not req.confirm:
        return _error(400, "deleting a model has to be confirmed. Send "
                           "confirm: true after looking at the delete plan.",
                      "confirmation_required")
    try:
        return ctx.library.delete_model(model_id,
                                        delete_mmproj=req.delete_mmproj,
                                        permanent=req.permanent)
    except ValueError as exc:
        return _error(404, str(exc), "not_found")
    except RuntimeError as exc:
        return _error(409, str(exc), "busy")
    except file_removal.RemovalError as exc:
        return _error(500, str(exc), "delete_failed")


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

class PreflightRequest(BaseModel):
    model_id: str
    context: int | None = Field(default=None, ge=512, le=1_048_576)
    gpu_layers: int | None = Field(default=None, ge=0, le=1000)
    gpu_mode: str = "auto"


@router.post("/preflight")
async def preflight(request: Request, req: PreflightRequest) -> Any:
    """Run the checks that must pass before a model is loaded."""
    ctx = _ctx(request)
    report = ctx.library.preflight(req.model_id, context=req.context,
                                    gpu_layers=req.gpu_layers,
                                    gpu_mode=req.gpu_mode)
    if not report.get("ok") and report.get("error"):
        return _error(404, report["error"], "not_found")
    return report


@router.get("/opencode")
async def opencode_status(request: Request) -> Any:
    """What the OpenCode config looks like, and whether it is in step."""
    ctx = _ctx(request)
    return ctx.library.opencode_status()


@router.post("/opencode/sync")
async def opencode_sync(request: Request) -> Any:
    """Add the library's models to the OpenCode config."""
    ctx = _ctx(request)
    return ctx.library.sync_opencode()


# ---------------------------------------------------------------------------
# Hugging Face (optional, off by default)
# ---------------------------------------------------------------------------

@router.get("/hf/search")
async def hf_search(request: Request, q: str = "", limit: int = 30,
                    filter: str = "gguf") -> Any:
    """Search the Hugging Face Hub.

    Browsing is a read-only metadata request, so it works whether or not
    downloading is enabled. Only the transfer itself is gated.

    ``filter`` is passed to the Hub. It defaults to ``gguf``, which restricts
    the results to repositories that actually ship GGUF files - the only
    format this project can load.
    """
    ctx = _ctx(request)
    try:
        results = await ctx.hf.search_models(q, limit=limit,
                                             model_filter=filter or None)
    except HuggingFaceError as exc:
        return _error(502, str(exc), "hub_unreachable")
    return {"results": results, "count": len(results), "query": q,
            "filter": filter or None,
            "downloads_enabled": ctx.settings_store.settings.allow_hf_download}


@router.get("/hf/files")
async def hf_files(request: Request, repo: str) -> Any:
    """List the files in a Hub repository, with size and hash."""
    ctx = _ctx(request)
    try:
        files = await ctx.hf.list_files(repo)
    except HuggingFaceError as exc:
        return _error(502, str(exc), "hub_unreachable")
    return {"repo": repo, "files": files, "count": len(files),
            "downloads_enabled": ctx.settings_store.settings.allow_hf_download}


class PlanRequest(BaseModel):
    repo_id: str
    filename: str
    target_dir: str | None = None


@router.post("/downloads/plan")
async def plan_download(request: Request, req: PlanRequest) -> Any:
    """Describe a download. Nothing is fetched at this point.

    Planning only reads metadata, so it works whether or not downloading is
    switched on - you can always see what a download would do before
    deciding to allow it.
    """
    ctx = _ctx(request)
    try:
        plan = await ctx.downloads.plan(req.repo_id, req.filename,
                                        req.target_dir)
    except ApprovalRequired as exc:
        return _error(403, str(exc), "approval_required")
    except DownloadError as exc:
        return _error(400, str(exc), "download_refused")
    except HuggingFaceError as exc:
        return _error(502, str(exc), "hub_unreachable")
    return {"plan": plan,
            "downloads_enabled": ctx.settings_store.settings.allow_hf_download}


class StartDownloadRequest(BaseModel):
    plan_id: str
    approve: bool = False


@router.post("/downloads")
async def start_download(request: Request, req: StartDownloadRequest) -> Any:
    """Start a planned download. Requires ``approve: true``."""
    ctx = _ctx(request)
    try:
        record = await ctx.downloads.start(req.plan_id, approve=req.approve)
    except ApprovalRequired as exc:
        return _error(403, str(exc), "approval_required")
    except DownloadError as exc:
        return _error(400, str(exc), "download_refused")
    return {"download": record}


@router.get("/downloads")
async def list_downloads(request: Request) -> Any:
    ctx = _ctx(request)
    downloads = ctx.downloads.list_downloads()
    return {"downloads": downloads, "count": len(downloads),
            "enabled": ctx.settings_store.settings.allow_hf_download}


@router.post("/downloads/{download_id}/cancel")
async def cancel_download(download_id: str, request: Request) -> Any:
    ctx = _ctx(request)
    try:
        record = ctx.downloads.cancel(download_id)
    except DownloadError as exc:
        return _error(404, str(exc), "not_found")
    return {"download": record}


# ---------------------------------------------------------------------------
# helper
# ---------------------------------------------------------------------------

async def _json(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001 - a broken body is a bad request
        return {}
    return body if isinstance(body, dict) else {}
