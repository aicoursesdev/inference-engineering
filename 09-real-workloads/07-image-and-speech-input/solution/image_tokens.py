"""How many prompt tokens is a picture? Send plain-colour images of several
sizes to a vision model and read the token count the server reports.

    llama-server -m Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf \
        --mmproj mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf -c 4096 -np 1 --port 8080
    uv run --with pillow python image_tokens.py [http://localhost:8080]
"""

import base64
import io
import sys
import time

import httpx
from PIL import Image

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"


def ask(width, height):
    buf = io.BytesIO()
    Image.new("RGB", (width, height), (200, 80, 60)).save(buf, "PNG")
    url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    body = {"max_tokens": 1, "temperature": 0, "cache_prompt": False,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": "What colour is this?"}]}]}
    t = time.perf_counter()
    r = httpx.post(URL + "/v1/chat/completions", json=body, timeout=300).json()
    return r["usage"]["prompt_tokens"], time.perf_counter() - t


ask(224, 224)  # warm up
print(f"{'image':>11} {'prompt tokens':>14} {'seconds':>8}")
for w, h in ((224, 224), (448, 448), (672, 672), (896, 896), (1344, 896)):
    n, secs = ask(w, h)
    print(f"{w:5d}x{h:<5d} {n:14d} {secs:8.1f}")
