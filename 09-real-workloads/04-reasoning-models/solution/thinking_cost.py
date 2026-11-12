"""How many tokens does a reasoning model spend thinking, and what does that
do to the length of the request?

Qwen3 can think before it answers (inside <think> ... </think>) or be told
not to. llama-server splits the two for us: the thinking arrives as
`reasoning_content`, the answer as `content`. We count streamed chunks (one
token each, near enough) of each kind.

    llama-server -m Qwen3-0.6B-Q8_0.gguf -c 4096 -np 1 --port 8080
    uv run python thinking_cost.py [http://localhost:8080]
"""

import json
import sys
import time

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
QUESTIONS = [
    "A Tidewater jacket costs 1,800 rupees and is 15% off in the sale. "
    "What is the sale price?",
    "A parcel leaves Kochi at 9:40 and takes 3 hours 35 minutes. When does it "
    "arrive?",
    "Three orders weigh 1.2 kg, 0.8 kg and 2.5 kg. Shipping is 40 rupees per "
    "kg. What is the total shipping cost?",
    "Is 2,047 a prime number? Answer yes or no and say why.",
    "A refund of 2,340 rupees is split equally between 4 orders. How much "
    "does each get?",
]


def run(question, thinking):
    body = {"max_tokens": 1500, "stream": True, "seed": 7,
            # the settings Qwen's model card recommends for each mode
            "temperature": 0.6 if thinking else 0.7,
            "top_p": 0.95 if thinking else 0.8, "top_k": 20,
            "chat_template_kwargs": {"enable_thinking": thinking},
            "messages": [{"role": "user", "content": question}]}
    think = answer = 0
    t0, first_answer = time.perf_counter(), None
    with httpx.stream("POST", URL + "/v1/chat/completions", json=body,
                      timeout=300) as r:
        for line in r.iter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            c = json.loads(line[6:])["choices"]
            d = c[0]["delta"] if c else {}
            think += bool(d.get("reasoning_content"))
            if d.get("content"):
                answer += 1
                first_answer = first_answer or time.perf_counter() - t0
    return think, answer, first_answer or 0.0, time.perf_counter() - t0


run(QUESTIONS[0], False)  # warm up
print(f"{'thinking':9} {'think tok':>10} {'answer tok':>11} "
      f"{'wait for answer':>16} {'total s':>8}")
for thinking in (False, True):
    for q in QUESTIONS:
        th, an, first, total = run(q, thinking)
        print(f"{'on' if thinking else 'off':9} {th:10d} {an:11d} "
              f"{first:15.2f}s {total:8.1f}")
