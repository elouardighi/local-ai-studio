"""Local AI Studio backend entry point.

Run with::

    .venv\\Scripts\\python.exe -m uvicorn backend.app.main:app \\
        --host 127.0.0.1 --port 1234

The app serves both the management API (``/api``), the OpenAI-compatible
gateway (``/v1``) and the web dashboard (``/``) from one port, so there is no
CORS or second-server juggling in normal use.
"""
from __future__ import annotations

import contextlib
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import models as models_api
from .api import openai_compat, system as system_api
from .config import FRONTEND_DIR, SettingsStore, ensure_dirs
from .services.context import AppContext


def create_app(settings_store: SettingsStore | None = None) -> FastAPI:
    ensure_dirs()
    settings_store = settings_store or SettingsStore()
    settings = settings_store.settings

    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await app.state.ctx.startup()
        try:
            yield
        finally:
            with contextlib.suppress(Exception):
                await app.state.ctx.shutdown()

    app = FastAPI(
        title="Local AI Studio",
        version="0.1.0",
        description="Local model server with an OpenAI-compatible API",
        lifespan=lifespan,
    )
    app.state.ctx = AppContext(settings_store)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- optional API-key auth for the gateway only ------------------------
    @app.middleware("http")
    async def gateway_auth(request: Request, call_next):
        key = app.state.ctx.settings_store.settings.api_key
        path = request.url.path
        if key and path.startswith("/v1/"):
            supplied = (request.headers.get("authorization", "")
                        .removeprefix("Bearer ").strip()
                        or request.headers.get("x-api-key", ""))
            if supplied != key:
                return JSONResponse(status_code=401, content={
                    "error": {"message": "invalid or missing API key",
                              "type": "authentication_error"}})
        # Reject oversized bodies before they reach a handler.
        max_bytes = app.state.ctx.settings_store.settings.max_request_bytes
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_bytes:
            return JSONResponse(status_code=413, content={
                "error": {"message": f"request too large (>{max_bytes} bytes)",
                          "type": "payload_too_large"}})
        return await call_next(request)

    # -- routers ------------------------------------------------------------
    # Management API first, then the frontend, and the OpenAI-compatible
    # gateway last: its catch-all must never shadow /api or the dashboard.
    app.include_router(models_api.router)
    app.include_router(system_api.router)

    # -- frontend -----------------------------------------------------------
    if FRONTEND_DIR.is_dir():
        assets = FRONTEND_DIR / "src"
        if assets.is_dir():
            app.mount("/src", StaticFiles(directory=str(assets)), name="src")

        @app.middleware("http")
        async def dev_friendly_caching(request: Request, call_next):
            """Revalidate dashboard assets instead of serving stale copies.

            The HTML and its modules are versioned together; a browser that
            caches ``models.js`` but reloads ``index.html`` (or vice versa)
            runs mismatched code. ``no-cache`` still allows 304 revalidation,
            so this costs one conditional request per asset, not a re-download.
            """
            response = await call_next(request)
            path = request.url.path
            if path.startswith("/src/") or path in ("/", "/index.html"):
                response.headers["Cache-Control"] = "no-cache"
            return response

        @app.get("/", include_in_schema=False)
        async def index() -> FileResponse:
            return FileResponse(str(FRONTEND_DIR / "index.html"))

    # Gateway registered last so its 404 catch-all is a true fallback.
    app.include_router(openai_compat.router)

    return app


app = create_app()


def main() -> None:
    """Console entry point honouring the configured host/port."""
    import uvicorn
    store = SettingsStore()
    s = store.settings
    host = s.bound_host()
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"[warn] binding to {host}: remote access is enabled; make "
              f"sure you trust every device on this network.")
    try:
        uvicorn.run(create_app(store), host=host, port=s.api_port,
                    log_level="info")
    except OSError as exc:
        if getattr(exc, "winerror", None) == 10048 or "10048" in str(exc):
            print(f"\n[error] port {s.api_port} is already in use. "
                  f"Either close the application holding it, or change "
                  f"'api_port' in data/settings.json.")
        raise


if __name__ == "__main__":
    main()
