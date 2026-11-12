"""A canary eval: 30 questions with one checkable answer each, sent to a running server.

    llama-server -m models/Qwen3-0.6B-Q8_0.gguf --port 8083 --jinja
    uv run python canary.py 8083 canary-q8.json

Temperature 0, thinking off, so the same build gives the same score. A change of
score on the same questions means the model or its numbers changed.
"""
import json, re, sys
import httpx

QUESTIONS = [  # (question, check) -- check is a regex the reply must match
    ("What is 17 + 25? Answer with the number only.", r"\b42\b"),
    ("What is 9 * 8? Answer with the number only.", r"\b72\b"),
    ("What is 100 - 37? Answer with the number only.", r"\b63\b"),
    ("What is 144 / 12? Answer with the number only.", r"\b12\b"),
    ("What is 15 + 28? Answer with the number only.", r"\b43\b"),
    ("What is 7 * 6? Answer with the number only.", r"\b42\b"),
    ("What is 250 + 175? Answer with the number only.", r"\b425\b"),
    ("What is 81 / 9? Answer with the number only.", r"\b9\b"),
    ("What is the capital of France? One word.", r"Paris"),
    ("What is the capital of Japan? One word.", r"Tokyo"),
    ("What is the capital of India? One word.", r"(New )?Delhi"),
    ("What is the capital of Kerala, India? One word.", r"Thiruvananthapuram|Trivandrum"),
    ("Which planet is closest to the Sun? One word.", r"Mercury"),
    ("What gas do plants take in from the air? One word.", r"carbon|CO2"),
    ("How many days are in a week? Number only.", r"\b7\b"),
    ("How many minutes are in an hour? Number only.", r"\b60\b"),
    ("What colour is the sky on a clear day? One word.", r"[Bb]lue"),
    ("What is the opposite of 'hot'? One word.", r"[Cc]old"),
    ("Spell 'cat' backwards. Answer with the word only.", r"tac"),
    ("Which is bigger, 9 or 12? Answer with the number only.", r"\b12\b"),
    ('Reply with only this JSON: {"ok": true}', r'\{\s*"ok"\s*:\s*true\s*\}'),
    ('Reply with a JSON object with key "city" and value "Pune". JSON only.', r'"city"\s*:\s*"Pune"'),
    ("Translate to French: 'good morning'. Reply with the phrase only.", r"[Bb]onjour"),
    ("Translate to Spanish: 'thank you'. Reply with the phrase only.", r"[Gg]racias"),
    ("What is the chemical symbol for water? Answer with the formula only.", r"H2O|H₂O"),
    ("What comes next: 2, 4, 6, 8, ? Number only.", r"\b10\b"),
    ("What is the first month of the year? One word.", r"January"),
    ("How many legs does a spider have? Number only.", r"\b8\b|eight"),
    ("Is 17 a prime number? Answer yes or no.", r"[Yy]es"),
    ("Write the word 'rain' in capital letters. Word only.", r"RAIN"),
]


def main(port, out):
    results = []
    for q, check in QUESTIONS:
        body = {"model": "m", "temperature": 0, "seed": 0, "max_tokens": 40, "messages": [{"role": "user", "content": q}],
                "chat_template_kwargs": {"enable_thinking": False}}
        reply = httpx.post(f"http://127.0.0.1:{port}/v1/chat/completions", json=body, timeout=60).json()["choices"][0]["message"]["content"].strip()
        results.append({"q": q, "reply": reply, "pass": bool(re.search(check, reply))})
    score = sum(r["pass"] for r in results)
    json.dump({"score": score, "total": len(results), "results": results}, open(out, "w"), indent=1, ensure_ascii=False)
    print(f"{out}: {score}/{len(results)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
