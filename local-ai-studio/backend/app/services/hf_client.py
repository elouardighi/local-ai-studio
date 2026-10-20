"""A read-only client for the Hugging Face Hub.

Used by the optional model downloader. It only ever performs GET requests
and never writes anything itself - the download manager decides what to do
with the answers.

Two endpoints are used:

* ``/api/models?search=...`` - find repositories by name;
* ``/api/models/{repo}/tree/main?recursive=true`` - list the files in a
  repository, with their size and their content hash.

The tree API reports LFS files with a SHA-256 in ``lfs.oid`` and ordinary git
blobs with a SHA-1 in ``oid``. Both are usable for verification, so a
download can be checked whichever kind of file it is.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import quote

import httpx

# Files we are willing to consider a model.
MODEL_SUFFIXES = (".gguf",)


class HuggingFaceError(RuntimeError):
    """A request to the Hub failed. The message is safe to show a user."""


def _split_repo(repo_id: str) -> tuple[str, str]:
    parts = repo_id.split("/")
    if len(parts) != 2 or not all(parts):
        raise HuggingFaceError(
            f"'{repo_id}' is not a repository id (expected 'owner/name')")
    return parts[0], parts[1]


class HuggingFaceClient:
    """Minimal, read-only Hub client."""

    def __init__(self, base_url: str = "https://huggingface.co",
                 timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    # -- helpers ----------------------------------------------------------

    async def _aget(self, url: str, params: dict[str, Any] | None = None
                    ) -> httpx.Response:
        async with httpx.AsyncClient(
                timeout=self.timeout, follow_redirects=True,
                headers={"User-Agent": "local-ai-studio/0.1"}) as c:
            resp = await c.get(url, params=params)
        return resp

    # -- search -----------------------------------------------------------

    async def search_models(self, query: str, limit: int = 30,
                            model_filter: str | None = None
                            ) -> list[dict[str, Any]]:
        """Find repositories whose name matches ``query``.

        ``model_filter`` is passed to the Hub as its ``filter`` parameter.
        ``"gguf"`` restricts the results to repositories that actually ship
        GGUF files, which is what this project can load.
        """
        query = (query or "").strip()
        if not query:
            return []
        url = f"{self.base_url}/api/models"
        params: dict[str, Any] = {
            "search": query, "limit": max(1, min(100, limit)),
            "sort": "downloads", "direction": -1, "full": "false",
        }
        if model_filter:
            params["filter"] = model_filter
        try:
            resp = await self._aget(url, params)
        except httpx.HTTPError as exc:
            raise HuggingFaceError(f"could not reach huggingface.co: {exc}") from exc
        if resp.status_code >= 400:
            raise HuggingFaceError(
                f"huggingface.co returned {resp.status_code} for '{query}'")
        try:
            data = resp.json()
        except ValueError as exc:
            raise HuggingFaceError("huggingface.co sent an unreadable answer") from exc
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            repo = item.get("modelId") or item.get("id")
            if not repo:
                continue
            out.append({
                "repo_id": repo,
                "author": item.get("author") or repo.split("/")[0],
                "name": item.get("id") or repo.split("/")[-1],
                "downloads": item.get("downloads") or 0,
                "likes": item.get("likes") or 0,
                "pipeline_tag": item.get("pipeline_tag"),
                "library_name": item.get("library_name"),
                "tags": [t for t in (item.get("tags") or [])][:20],
                "gated": bool(item.get("gated")),
            })
        return out

    # -- files ------------------------------------------------------------

    async def list_files(self, repo_id: str) -> list[dict[str, Any]]:
        """List every file in a repository, with size and content hash."""
        owner, name = _split_repo(repo_id)
        url = f"{self.base_url}/api/models/{quote(owner)}/{quote(name)}/tree/main"
        try:
            resp = await self._aget(url, {"recursive": "true"})
        except httpx.HTTPError as exc:
            raise HuggingFaceError(f"could not reach huggingface.co: {exc}") from exc
        if resp.status_code == 404:
            raise HuggingFaceError(f"repository '{repo_id}' was not found")
        if resp.status_code >= 400:
            raise HuggingFaceError(
                f"huggingface.co returned {resp.status_code} for '{repo_id}'")
        try:
            data = resp.json()
        except ValueError as exc:
            raise HuggingFaceError("huggingface.co sent an unreadable answer") from exc
        if not isinstance(data, list):
            return []

        out: list[dict[str, Any]] = []
        for item in data:
            if not isinstance(item, dict) or item.get("type") != "file":
                continue
            path = item.get("path")
            if not path:
                continue
            lfs = item.get("lfs") or {}
            algorithm = None
            digest = None
            if isinstance(lfs, dict) and lfs.get("oid"):
                # LFS object id: "sha256:<hex>"
                raw = str(lfs["oid"])
                if ":" in raw:
                    algorithm, digest = raw.split(":", 1)
                else:
                    algorithm, digest = "sha256", raw
                size = int(lfs.get("size") or item.get("size") or 0)
            elif item.get("oid"):
                algorithm, digest = "sha1", str(item["oid"])
                size = int(item.get("size") or 0)
            else:
                algorithm, digest = None, None
                size = int(item.get("size") or 0)
            out.append({
                "path": path,
                "filename": path.rsplit("/", 1)[-1],
                "size_bytes": size,
                "size_human": _human(size),
                "hash_algorithm": algorithm,
                "hash": digest,
                "is_model": path.lower().endswith(MODEL_SUFFIXES),
                "is_mmproj": "mmproj" in path.lower(),
            })
        out.sort(key=lambda f: (not f["is_model"], f["path"].lower()))
        return out

    def resolve_url(self, repo_id: str, filename: str) -> str:
        """The URL a file is downloaded from."""
        owner, name = _split_repo(repo_id)
        path = filename.lstrip("/")
        return (f"{self.base_url}/{quote(owner)}/{quote(name)}/resolve/main/"
                f"{quote(path, safe='/')}")


def _human(n: int) -> str:
    v = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if v < 1024 or unit == "TB":
            return f"{v:.1f} {unit}" if unit != "B" else f"{int(v)} B"
        v /= 1024.0
    return f"{v:.1f} TB"
