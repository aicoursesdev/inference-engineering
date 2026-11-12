import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def output(script, *args, check=True):
    done = subprocess.run([sys.executable, str(HERE / script), *args],
                          capture_output=True, text=True, check=check)
    return done


class Fake(BaseHTTPRequestHandler):
    """An OpenAI-style streaming reply, as trtllm-serve, vLLM or llama-server send it."""
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in ("Ganga, ", "Yamuna, ", "Godavari"):
            chunk = {"choices": [{"delta": {"content": word}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *args):
        pass


def test_ask_times_a_streamed_reply_from_any_compatible_server():
    server = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        done = output("ask.py", f"http://127.0.0.1:{server.server_port}", "any-model")
    finally:
        server.shutdown()
    assert "Ganga, Yamuna, Godavari" in done.stdout
    assert "3 chunks" in done.stdout and "TTFT" in done.stdout


def test_check_prereqs_reports_each_line_of_the_install_page():
    done = output("check_prereqs.py", check=False)
    lines = done.stdout.strip().splitlines()
    assert len(lines) == 6 and done.returncode in (0, 1)
    assert all(l[:4] in ("ok  ", "MISS") for l in lines)
