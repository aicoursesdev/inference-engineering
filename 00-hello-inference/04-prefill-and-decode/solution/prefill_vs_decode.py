"""Time the two phases on the local llama-server.

Sends prompts of growing length with max_tokens=1 (almost all prefill),
then one short prompt asking for a long reply (almost all decode). The
server reports its own timings for each phase in the reply.

    llama-server -hf Qwen/Qwen3-0.6B-GGUF:Q8_0 --port 8081
    uv run python prefill_vs_decode.py
"""

import os
import statistics

import httpx

PORT = os.environ.get("PORT", "8081")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
POLICY = ("Returns are accepted within 30 days of delivery if the item is "
          "unused and the tags are on. A courier collects it free of charge. "
          "Refunds reach the original payment method in 5 to 7 days. ")


def ask(text, max_tokens):
    body = {"model": "m", "temperature": 0, "max_tokens": max_tokens,
            "cache_prompt": False,  # no reuse: measure honest prefill
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text}]}
    return httpx.post(URL, json=body, timeout=120).json()["timings"]


ask("hello", 1)  # warm up
print("prompt_tokens  prefill_ms  prefill_tok_per_s   (median of 5)")
for copies in (1, 4, 16, 32):
    runs = [ask(POLICY * copies + "Summarise.", 1) for _ in range(5)]
    ms = statistics.median(t["prompt_ms"] for t in runs)
    n = runs[0]["prompt_n"]
    print(f"{n:13d}  {ms:10.1f}  {n / ms * 1000:17.0f}")

t = ask("Write a long story about a monsoon.", 200)
print(f"\ndecode: {t['predicted_n']} tokens in {t['predicted_ms']:.0f} ms"
      f" = {t['predicted_per_second']:.0f} tok/s,"
      f" {t['predicted_per_token_ms']:.2f} ms per token")
