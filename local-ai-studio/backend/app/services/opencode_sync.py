"""Keeping an OpenCode config in step with the model library.

OpenCode does not ask an OpenAI-compatible server what models it has. For a
custom provider it reads the model list straight out of its own config file,
so a model that is downloaded after the config was written does not appear
in the ``/models`` menu until the config is updated.

This module does that update, and it writes **honest limits**: the context
length a model can really use is what fits in the free VRAM, not the number
the model advertises. A client that believes it has 150k tokens builds a
prompt the engine will only reject.

The config is edited as text, so comments, formatting and every other
provider in it are left exactly as they were. A backup is written before
anything changes, and a config that would no longer parse is never
written.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

# Where an OpenCode config may live.
GLOBAL_PATHS = (
    Path.home() / ".config" / "opencode" / "opencode.jsonc",
    Path(os.environ.get("APPDATA", "")) / "opencode" / "opencode.jsonc",
    Path(os.environ.get("LOCALAPPDATA", "")) / "opencode" / "opencode.jsonc",
)
PROJECT_NAMES = ("opencode.jsonc", "opencode.json")

# The provider this project manages.
DEFAULT_PROVIDER = "localai"


class ConfigError(RuntimeError):
    """The config could not be read or written."""


def find_config(explicit: str | None = None) -> Path | None:
    """Locate the OpenCode config to work on."""
    if explicit:
        path = Path(os.path.expandvars(explicit)).expanduser()
        return path if path.is_file() else None
    for candidate in GLOBAL_PATHS:
        if candidate.is_file():
            return candidate
    return None


# ---------------------------------------------------------------------------
# Editing the config as text
# ---------------------------------------------------------------------------

def _skip_string(text: str, i: int) -> int:
    """Index just past the string literal that starts at ``i``."""
    quote = text[i]
    i += 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote:
            return i + 1
        i += 1
    return i


def _skip_trivia(text: str, i: int) -> int:
    """Index just past whitespace and comments."""
    while i < len(text):
        ch = text[i]
        if ch.isspace():
            i += 1
        elif text.startswith("//", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end + 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
        else:
            break
    return i


def _matching_brace(text: str, open_index: int) -> int:
    """Index of the ``}`` that closes the ``{`` at ``open_index``."""
    depth = 0
    i = open_index
    while i < len(text):
        ch = text[i]
        if ch in "\"'":
            i = _skip_string(text, i)
            continue
        if text.startswith("//", i) or text.startswith("/*", i):
            i = _skip_trivia(text, i)
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ConfigError("the config has an unbalanced brace")


def find_block(text: str, key_path: list[str]) -> tuple[int, int] | None:
    """Locate the ``{ ... }`` block at ``key_path``.

    Returns the span of the braces, or None when the path is not there.
    """
    pos = 0
    for key in key_path:
        pattern = re.compile(r'"' + re.escape(key) + r'"\s*:')
        match = pattern.search(text, pos)
        if not match:
            return None
        i = _skip_trivia(text, match.end())
        if i >= len(text) or text[i] != "{":
            return None
        end = _matching_brace(text, i)
        if not key_path[:-1]:
            return i, end
        inner = find_block(text[i:end + 1], key_path[1:])
        if inner is None:
            return None
        return i + inner[0], i + inner[1]
    return None


def _indent_of(text: str, index: int) -> str:
    """The leading whitespace of the line the index sits on."""
    line_start = text.rfind("\n", 0, index) + 1
    prefix = text[line_start:index]
    return prefix[:len(prefix) - len(prefix.lstrip())]


def strip_jsonc(text: str) -> str:
    """Remove comments without touching anything inside a string."""
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in "\"'":
            end = _skip_string(text, i)
            out.append(text[i:end])
            i = end
            continue
        if text.startswith("//", i):
            end = text.find("\n", i)
            i = len(text) if end < 0 else end
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = len(text) if end < 0 else end + 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _render_value(value: Any) -> str:
    """A value as it is written in the config.

    Objects are written on one line with spaces inside the braces, which is
    how OpenCode configs usually look - so a rewrite does not churn the
    whole file.
    """
    if isinstance(value, dict):
        inner = ", ".join(f'"{k}": {json.dumps(v)}' for k, v in value.items())
        return "{ " + inner + " }"
    return json.dumps(value)


def _render_entry(name: str, spec: dict, indent: str) -> list[str]:
    """One entry of a models block, in the file's own style."""
    inner = indent + "  "
    lines = [f'{inner}"{name}": {{']
    items = list(spec.items())
    for position, (key, value) in enumerate(items):
        comma = "," if position < len(items) - 1 else ""
        lines.append(f'{inner}  "{key}": {_render_value(value)}{comma}')
    lines.append(f"{inner}}}")
    return lines


def render_models(models: dict[str, dict], indent: str) -> str:
    """The text of a ``models`` block."""
    lines = ["{"]
    names = list(models)
    for position, (name, spec) in enumerate(models.items()):
        entry = _render_entry(name, spec, indent + "  ")
        if position < len(names) - 1:
            entry[-1] = entry[-1] + ","
        lines.extend(entry)
    lines.append(indent + "}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The config
# ---------------------------------------------------------------------------

class OpenCodeConfig:
    """An OpenCode config file, read and edited as text."""

    def __init__(self, path: Path, provider: str = DEFAULT_PROVIDER) -> None:
        self.path = Path(path)
        self.provider = provider

    # -- reading ----------------------------------------------------------

    def read(self) -> str:
        try:
            return self.path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(f"cannot read {self.path}: {exc}") from exc

    def provider_models(self) -> dict[str, dict]:
        """The models the provider currently lists."""
        text = self.read()
        span = find_block(text, ["provider", self.provider, "models"])
        if span is None:
            return {}
        start, end = span
        try:
            return json.loads(text[start:end + 1])
        except ValueError:
            return {}

    def status(self) -> dict[str, Any]:
        """A summary of the config, for the dashboard."""
        exists = self.path.is_file()
        models: dict[str, dict] = {}
        error: str | None = None
        if exists:
            try:
                models = self.provider_models()
            except ConfigError as exc:
                error = str(exc)
        return {
            "path": str(self.path),
            "exists": exists,
            "provider": self.provider,
            "models": list(models),
            "count": len(models),
            "error": error,
        }

    # -- writing ----------------------------------------------------------

    def _write(self, text: str) -> None:
        # Never write a config that no longer parses.
        try:
            json.loads(strip_jsonc(text))
        except ValueError as exc:
            raise ConfigError(f"the edited config would not be valid JSON "
                              f"({exc}); nothing was written") from exc
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            backup = self.path.with_suffix(self.path.suffix + f".bak-{int(time.time())}")
            if self.path.is_file():
                shutil.copy2(self.path, backup)
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            raise ConfigError(f"cannot write {self.path}: {exc}") from exc

    def _edit(self, mutate) -> bool:
        """Apply ``mutate`` to the provider's models. Returns True if changed."""
        text = self.read()
        span = find_block(text, ["provider", self.provider, "models"])
        if span is None:
            return False
        start, end = span
        indent = _indent_of(text, start)
        try:
            existing = json.loads(text[start:end + 1])
        except ValueError as exc:
            raise ConfigError(f"the models block in {self.path} cannot be "
                              f"read ({exc})") from exc
        models = mutate(dict(existing))
        if models == existing:
            return False
        new = render_models(models, indent)
        self._write(text[:start] + new + text[end + 1:])
        return True

    def add_model(self, name: str, spec: dict, replace: bool = False) -> bool:
        """Add one model. An existing entry is left alone unless replaced."""
        def mutate(models: dict) -> dict:
            if name in models and not replace:
                return models
            models[name] = spec
            return models

        return self._edit(mutate)

    def remove_model(self, name: str) -> bool:
        """Remove one model. A model that is not listed is not an error."""
        def mutate(models: dict) -> dict:
            models.pop(name, None)
            return models

        return self._edit(mutate)

    def set_limit(self, name: str, context: int, output: int) -> bool:
        """Update only the limits of a model that is already listed."""
        def mutate(models: dict) -> dict:
            spec = models.get(name)
            if spec is None:
                return models
            updated = dict(spec)
            updated["limit"] = {"context": int(context), "output": int(output)}
            models[name] = updated
            return models

        return self._edit(mutate)
