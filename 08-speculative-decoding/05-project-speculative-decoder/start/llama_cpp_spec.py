"""Decode speed from llama-server, with and without a draft model.

    # plain
    llama-server -m Qwen3-1.7B-Q8_0.gguf -c 2048 -np 1 --port 8080
    # with a draft model (the same tokenizer): add -md, the type and a draft length
    # (without --spec-type draft-simple the draft model has no effect)
    llama-server -m Qwen3-1.7B-Q8_0.gguf -md Qwen3-0.6B-Q8_0.gguf \
        --spec-type draft-simple --spec-draft-n-max 4 -c 2048 -np 1 --port 8080
    uv run python llama_cpp_spec.py [http://localhost:8080]

Prints the server's own timings for each prompt, and its draft counters when
the server reports them (draft_n guessed, draft_n_accepted kept).
"""

import sys

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
POLICY = ("Returns: you may return rain gear within ten days of delivery if "
          "it is unworn and the tags are on. Refunds go back to the original "
          "payment method within five working days of the parcel reaching "
          "our Kochi warehouse.")
PROMPTS = {
    "open": "Explain in three sentences why the monsoon matters to farmers.",
    "copy": "Repeat this policy word for word, changing ten days to "
            f"fourteen days.\n\n{POLICY}",
    "code": "Write a Python function that returns the n-th Fibonacci number.",
}


def ask(text):
    body = {"temperature": 0, "max_tokens": 128, "cache_prompt": False,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": text}]}
    r = httpx.post(URL + "/v1/chat/completions", json=body, timeout=300).json()
    return r["timings"], r["choices"][0]["message"]["content"]


ask("Hi")  # warm up
print(f"{'prompt':6} {'tok/s':>7} {'tokens':>7} {'drafted':>8} {'kept':>6}")
for name, text in PROMPTS.items():
    best = max((ask(text)[0] for _ in range(3)),
               key=lambda t: t["predicted_per_second"])
    drafted, kept = best.get("draft_n"), best.get("draft_n_accepted")
    print(f"{name:6} {best['predicted_per_second']:7.1f} "
          f"{best['predicted_n']:7d} {drafted if drafted is not None else '-':>8} "
          f"{kept if kept is not None else '-':>6}")
