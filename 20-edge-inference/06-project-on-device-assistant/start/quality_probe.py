"""A tiny quality probe: 12 questions with one checkable answer each.

Temperature 0, thinking off, so the same file gives the same score. It is
not a benchmark: it catches a level that has broken, not a small loss.
"""
import re

import httpx

QUESTIONS = [  # (question, regex the reply must match)
    ("What is 17 + 25? Number only.", r"\b42\b"),
    ("What is 9 * 8? Number only.", r"\b72\b"),
    ("What is 100 - 37? Number only.", r"\b63\b"),
    ("What is 144 / 12? Number only.", r"\b12\b"),
    ("What is the capital of France? One word.", r"Paris"),
    ("What is the capital of Japan? One word.", r"Tokyo"),
    ("What is the capital of Kerala, India? One word.",
     r"Thiruvananthapuram|Trivandrum"),
    ("Which planet is closest to the Sun? One word.", r"Mercury"),
    ("How many days are in a week? Number only.", r"\b7\b"),
    ("What is the opposite of 'hot'? One word.", r"[Cc]old"),
    ("Translate to French: 'good morning'. Phrase only.", r"[Bb]onjour"),
    ("Reply with only this JSON: {\"ok\": true}", r"\{\s*\"ok\"\s*:\s*true\s*\}"),
]


def score(base):
    right = 0
    for q, check in QUESTIONS:
        body = {"model": "m", "temperature": 0, "seed": 0, "max_tokens": 40,
                "messages": [{"role": "user", "content": q}],
                "chat_template_kwargs": {"enable_thinking": False}}
        r = httpx.post(base + "/v1/chat/completions", json=body, timeout=120)
        reply = r.json()["choices"][0]["message"]["content"]
        right += bool(re.search(check, reply))
    return right, len(QUESTIONS)
