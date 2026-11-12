"""Offline: prefill_vs_decode.py against a fake server that reports timings."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).parent


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = len(body["messages"][0]["content"].split())
        timings = {"prompt_n": n, "prompt_ms": 10 + n / 10, "predicted_n": 200,
                   "predicted_ms": 1000.0, "predicted_per_second": 200.0,
                   "predicted_per_token_ms": 5.0}
        data = json.dumps({"timings": timings}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


def test_prints_four_prompt_lengths_then_the_decode_speed():
    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = subprocess.run(
            [sys.executable, "prefill_vs_decode.py"], cwd=HERE,
            capture_output=True, text=True, timeout=60,
            env={**os.environ, "PORT": str(server.server_port)})
    finally:
        server.shutdown()
    assert out.returncode == 0, out.stderr[-300:]
    rows = [l for l in out.stdout.splitlines() if l[:13].strip().isdigit()]
    sizes = [int(l.split()[0]) for l in rows]
    assert len(sizes) == 4 and sizes == sorted(sizes)
    assert "decode: 200 tokens in 1000 ms = 200 tok/s" in out.stdout
