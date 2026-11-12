"""Offline: stream_demo.py against a fake llama-server that replays the
recorded reply, with an error chunk (no "choices") in the middle."""

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

HERE = Path(__file__).parent
RECORDING = next(p for p in Path(__file__).parents
                 if (p / "data").is_dir()) / "data/recorded/llama-server-reply.json"


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        events = [e["data"] for e in json.loads(RECORDING.read_text())["events"]]
        events.insert(5, json.dumps({"error": {"message": "slot busy"}}))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for data in events:
            self.wfile.write(f"data: {data}\n\n".encode())

    def log_message(self, *args):
        pass


def test_prints_a_time_beside_every_token_and_survives_an_error_chunk():
    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        out = subprocess.run(
            [sys.executable, "stream_demo.py"], cwd=HERE, capture_output=True,
            text=True, timeout=30,
            env={**os.environ, "PORT": str(server.server_port)})
    finally:
        server.shutdown()
    assert out.returncode == 0, out.stderr[-300:]
    lines = out.stdout.splitlines()
    assert len(lines) == 33 and all(" ms  '" in line for line in lines)
