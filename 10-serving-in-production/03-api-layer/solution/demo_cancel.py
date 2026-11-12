"""Start a long streamed reply through the gateway, hang up after 5 chunks, and
watch llama-server's own gauge fall back to 0 (needs the gateway on :8090,
llama-server on :8081 with --metrics)."""
import re
import time

import httpx

BODY = {"model": "m", "stream": True, "max_tokens": 2000, "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
        "messages": [{"role": "user", "content": "Write a long story about monsoon rain."}]}


def processing():
    text = httpx.get("http://127.0.0.1:8081/metrics").text
    return int(re.search(r"^llamacpp:requests_processing (\d+)", text, re.M)[1])


def tokens():
    text = httpx.get("http://127.0.0.1:8081/metrics").text
    return int(float(re.search(r"^llamacpp:tokens_predicted_total (\S+)", text, re.M)[1]))


before = tokens()
with httpx.stream("POST", "http://127.0.0.1:8090/v1/chat/completions", json=BODY, timeout=30) as r:
    for i, line in enumerate(r.iter_lines()):
        if i == 5:
            print("client hangs up; server busy:", processing())
            break
for t in (0.2, 0.5, 1.0):
    time.sleep(t)
    print(f"+{t:.1f} s busy={processing()}")
time.sleep(1)
print("tokens generated for a 2000-token request:", tokens() - before)
