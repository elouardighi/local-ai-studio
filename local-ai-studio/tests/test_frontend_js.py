"""Runs the frontend JS unit tests (Node's built-in runner).

Keeps the Logs-page normalizer honest: the `Invalid Date` bug came from the
page treating a CustomEvent as a log record, so the unwrap/normalize logic
lives in a pure module (frontend/src/logutil.js) that is tested here without
a browser. Skips cleanly when Node is not installed.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

FRONTEND_TEST_DIR = Path(__file__).resolve().parent.parent / "frontend" / "test"
NODE = shutil.which("node")


@pytest.mark.skipif(NODE is None, reason="node not installed")
def test_frontend_js_unit_tests():
    """node --test frontend/test/*.mjs must pass (logutil normalizer, etc.)."""
    test_files = sorted(FRONTEND_TEST_DIR.glob("*.test.mjs"))
    assert test_files, f"no JS tests found in {FRONTEND_TEST_DIR}"
    proc = subprocess.run(
        [NODE, "--test", *[str(f) for f in test_files]],
        capture_output=True, text=True, timeout=120,
        cwd=FRONTEND_TEST_DIR.parent,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, f"JS unit tests failed:\n{output}"
    assert "fail 0" in output, f"expected zero JS failures:\n{output}"
