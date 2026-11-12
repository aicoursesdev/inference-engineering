"""Send images of known sizes to a vision server; record prompt tokens
and time to first token. Works with any OpenAI-compatible endpoint.

python measure_images.py http://localhost:8091 img_*.png
"""
import base64
import json
import sys
import time

import httpx


def best(base, png=None, text="Describe this picture in one sentence.", n=3):
    """Best of n: other programs on a shared laptop only ever add time."""
    runs = [ask(base, png, text) for _ in range(n)]
    return runs[0][0], min(t for _, t in runs)


def ask(base, png=None, text="Describe this picture in one sentence."):
    content = [{"type": "text", "text": text}]
    if png:
        b64 = base64.b64encode(open(png, "rb").read()).decode()
        content.insert(0, {"type": "image_url",
                           "image_url": {"url": f"data:image/png;base64,{b64}"}})
    body = {"messages": [{"role": "user", "content": content}],
            "max_tokens": 16, "temperature": 0, "stream": True,
            "stream_options": {"include_usage": True}, "cache_prompt": False}
    t0, first, usage = time.perf_counter(), None, None
    with httpx.stream("POST", base + "/v1/chat/completions", json=body,
                      timeout=120) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[6:])
            if "error" in chunk:
                raise RuntimeError(chunk["error"])
            if chunk.get("usage"):
                usage = chunk["usage"]
            delta = chunk["choices"][0]["delta"] if chunk.get("choices") else {}
            if first is None and delta.get("content"):
                first = time.perf_counter() - t0
    return usage["prompt_tokens"], first


if __name__ == "__main__":
    base, files = sys.argv[1], sys.argv[2:]
    ask(base)  # warm-up
    print(f"{'input':16s} {'prompt tokens':>13s} {'TTFT s':>7s}")
    n, t = best(base)
    print(f"{'text only':16s} {n:13d} {t:7.2f}")
    long_text = "The monsoon reaches Kochi in early June. " * 100
    n, t = best(base, text=long_text)
    print(f"{'text, long':16s} {n:13d} {t:7.2f}")
    for f in files:
        n, t = best(base, f)
        print(f"{f[4:-4]:16s} {n:13d} {t:7.2f}")
