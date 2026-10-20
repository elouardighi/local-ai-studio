"""Keep an OpenCode provider's model list in step with Local AI Studio.

The work is done by ``backend.app.services.opencode_sync`` - the same code
the backend uses when a model is downloaded or deleted, so there is one
implementation of the config surgery and not two.

OpenCode does not ask an OpenAI-compatible server what models it has. For a
custom provider it reads the model list straight out of its own config
file, so a model that is downloaded after the config was written does not
appear in the ``/models`` menu until the config is updated.

Usage
-----
    python scripts/sync_opencode_models.py                    # add what is missing
    python scripts/sync_opencode_models.py --dry-run          # show the changes
    python scripts/sync_opencode_models.py --force            # rewrite all limits
    python scripts/sync_opencode_models.py --provider bionic  # a different provider
    python scripts/sync_opencode_models.py --config path.jsonc
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.app.services.opencode_sync import (  # noqa: E402
    DEFAULT_PROVIDER, OpenCodeConfig, find_config,
)

DEFAULT_BASE_URL = "http://127.0.0.1:1234/v1"


def fetch_models(base_url: str, timeout: float = 20.0) -> list[dict]:
    """The models the server reports, with their recommended limits."""
    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read() or b"null")
    except urllib.error.URLError as exc:
        raise SystemExit(f"cannot reach {url}: {exc}")
    except ValueError as exc:
        raise SystemExit(f"{url} sent an unreadable answer: {exc}")
    models = payload.get("data") if isinstance(payload, dict) else None
    return models or []


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL,
                    help="the Local AI Studio /v1 address")
    ap.add_argument("--config", type=Path, default=None,
                    help="the OpenCode config file to update")
    ap.add_argument("--provider", default=DEFAULT_PROVIDER,
                    help="the provider whose models should be synced")
    ap.add_argument("--force", action="store_true",
                    help="also rewrite the limits of models that are "
                         "already listed")
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would change, write nothing")
    args = ap.parse_args()

    path = find_config(str(args.config) if args.config else None)
    if path is None:
        raise SystemExit("no OpenCode config found. Pass --config to say "
                         "which file to update.")
    config = OpenCodeConfig(path, provider=args.provider)

    models = fetch_models(args.base_url)
    if not models:
        raise SystemExit(f"{args.base_url} reported no models")

    wanted: dict[str, dict] = {}
    for m in models:
        name = m.get("id")
        if not name:
            continue
        meta = m.get("metadata") or {}
        recommended = meta.get("recommended") or {}
        wanted[name] = {
            "name": meta.get("display_name") or name,
            "limit": {
                "context": int(recommended.get("context") or 8192),
                "output": int(recommended.get("output") or 4096),
            },
        }

    print(f"{len(wanted)} model(s) reported by {args.base_url}")
    for name, spec in wanted.items():
        print(f"  {name}  (context {spec['limit']['context']:,}, "
              f"output {spec['limit']['output']:,})")

    existing = config.provider_models()
    added = [n for n in wanted if n not in existing]
    updated = [n for n in wanted if n in existing and args.force]
    if not added and not updated:
        print(f"\n{path} is already in step with the server "
              f"(use --force to rewrite the limits).")
        return 0
    if args.dry_run:
        print(f"\n[dry run] {path} would be updated. Nothing written.")
        return 0

    for name in added:
        config.add_model(name, wanted[name])
    if updated:
        for name in updated:
            # Only the limits are the server's business. The display name is
            # the user's choice and is left alone.
            config.set_limit(name, wanted[name]["limit"]["context"],
                             wanted[name]["limit"]["output"])
    print(f"\nupdated {path}")
    print("Restart OpenCode, or open the /models menu again, to see them.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
