"""Shared test helpers: synthetic GGUF files and a fake engine server.

Tests must run on any machine - no GPU, no network, no 11 GB model files.
"""
from __future__ import annotations

import struct
import sys
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def write_gguf(path: Path, metadata: dict, tensors: list | None = None,
               data_bytes: int = 0) -> Path:
    """Write a minimal but structurally valid GGUF v3 file.

    ``tensors`` is a list of ``(name, dims, type)`` tuples; the file then
    carries a real tensor directory and ``data_bytes`` of tensor data, which
    is what makes a truncated file detectable.
    """
    def pack_string(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    tensors = tensors or []
    out = bytearray()
    out += struct.pack("<I", 0x46554747)  # magic
    out += struct.pack("<I", 3)           # version
    out += struct.pack("<Q", len(tensors))  # tensor count
    out += struct.pack("<Q", len(metadata))
    for key, value in metadata.items():
        out += pack_string(key)
        if isinstance(value, str):
            out += struct.pack("<I", 8) + pack_string(value)
        elif isinstance(value, bool):
            out += struct.pack("<I", 7) + struct.pack("<?", value)
        elif isinstance(value, int):
            out += struct.pack("<I", 4) + struct.pack("<I", value)
        elif isinstance(value, float):
            out += struct.pack("<I", 6) + struct.pack("<f", value)
        elif isinstance(value, list) and value and isinstance(value[0], str):
            out += struct.pack("<I", 9)      # array
            out += struct.pack("<I", 8)      # of strings
            out += struct.pack("<Q", len(value))
            for item in value:
                out += pack_string(item)
        else:
            raise TypeError(f"unsupported test value: {value!r}")

    # Tensor directory: name, n_dims, dims (u64), type, offset.
    offset = 0
    for name, dims, ttype in tensors:
        out += pack_string(name)
        out += struct.pack("<I", len(dims))
        for dim in dims:
            out += struct.pack("<Q", dim)
        out += struct.pack("<I", ttype)
        out += struct.pack("<Q", offset)
        offset += 4 * (dims[0] if dims else 1)   # assume f32 elements
    # The tensor data starts on a 32-byte boundary, as the format requires.
    out += b"\0" * ((-len(out)) % 32)
    out += b"\0" * data_bytes
    path.write_bytes(bytes(out))
    return path


FAKE_ENGINE = r'''
"""Minimal llama-server stand-in used by process-manager tests.

Speaks just enough HTTP: /health and /v1/chat/completions (JSON + SSE).
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 18999
LOAD_DELAY = float(sys.argv[sys.argv.index("--load-delay") + 1]) if "--load-delay" in sys.argv else 0.0
RECORD = sys.argv[sys.argv.index("--record") + 1] if "--record" in sys.argv else None
READY = time.time() + LOAD_DELAY


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, payload, ctype="application/json"):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            if time.time() < READY:
                self._send(503, {"status": "loading model"})
            else:
                self._send(200, {"status": "ok"})
        elif self.path == "/metrics":
            body = b"llamacpp:tokens_predicted_total 42\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        if RECORD:
            with open(RECORD, "w", encoding="utf-8") as fh:
                json.dump(req, fh)
        if self.path != "/v1/chat/completions":
            self._send(404, {"error": {"message": "unknown endpoint"}})
            return
        if req.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for word in ("Hello", " from", " fake", " engine"):
                chunk = {"choices": [{"delta": {"content": word}}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.01)
            # Like the real engine: usage only arrives when asked for it.
            final = {"choices": [{"delta": {}, "finish_reason": "stop"}]}
            if req.get("stream_options", {}).get("include_usage"):
                final["usage"] = {"prompt_tokens": 5, "completion_tokens": 4}
            self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
        else:
            self._send(200, {
                "id": "fake-1", "model": req.get("model", "fake"),
                "choices": [{"message": {"role": "assistant", "content": "Hello from fake engine"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 4},
            })


HTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
'''


@pytest.fixture
def fake_engine_path(tmp_path: Path) -> Path:
    path = tmp_path / "fake_engine.py"
    path.write_text(FAKE_ENGINE, encoding="utf-8")
    return path


def port_range(base: int, span: int = 100) -> tuple[int, int]:
    """A port range that is unlikely to clash with another test run.

    The engine port is chosen by binding a socket and letting go of it
    again, so two runs of the suite at the same time can pick the same
    port. Offsetting the range by the process id makes that unlikely
    without needing a fixed range.
    """
    start = base + (os.getpid() % 500) * 2
    return start, start + span


@pytest.fixture
def gguf_model(tmp_path: Path) -> Path:
    """A tiny synthetic model resembling the user's Qwen3.6 MoE model."""
    path = tmp_path / "Test-Model-Q4_K_M.gguf"
    write_gguf(path, {
        "general.architecture": "qwen35moe",
        "general.name": "Test Model",
        "general.file_type": 15,
        "qwen35moe.block_count": 40,
        "qwen35moe.context_length": 262144,
        "qwen35moe.embedding_length": 2048,
        "qwen35moe.attention.head_count": 16,
        "qwen35moe.attention.head_count_kv": 2,
        "qwen35moe.attention.key_length": 256,
        "qwen35moe.attention.value_length": 256,
        "qwen35moe.expert_count": 256,
        "qwen35moe.expert_used_count": 8,
        "qwen35moe.full_attention_interval": 4,
        "qwen35moe.ssm.state_size": 128,
        "qwen35moe.ssm.group_count": 16,
        "tokenizer.ggml.model": "gpt2",
    })
    return path
