"""Ask a small model for a tool call as JSON, three ways, and measure it:
asked nicely (just words in the prompt), a JSON schema, and a hand-written
grammar. Counts how often the reply is valid, and what it costs in speed.

    llama-server -m Qwen3-0.6B-Q8_0.gguf -c 2048 -np 1 --port 8080
    uv run python schema_overhead.py [http://localhost:8080]
"""

import json
import statistics
import sys

import httpx

URL = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080")
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["tool", "path", "line"],
          "properties": {"tool": {"enum": ["read_file", "edit_file", "run_tests"]},
                         "path": {"type": "string"},
                         "line": {"type": "integer"}}}
# The same shape, written by hand in llama.cpp's grammar language (GBNF).
GBNF = r'''
root   ::= "{" ws "\"tool\":" ws tool "," ws "\"path\":" ws string "," ws "\"line\":" ws num ws "}"
tool   ::= "\"read_file\"" | "\"edit_file\"" | "\"run_tests\""
string ::= "\"" [a-zA-Z0-9_./-]* "\""
num    ::= [0-9]+
ws     ::= [ \n]*
'''
TASKS = ["open the checkout handler in cart.py near line 88",
         "run the tests for the payments module",
         "change line 12 of settings.py",
         "show me line 140 of orders/views.py",
         "edit the README at line 3",
         "run the test suite", "read utils/dates.py from line 20",
         "fix the typo on line 7 of config.yaml",
         "look at line 55 of api/routes.py",
         "edit line 301 of models/invoice.py", "run the linter tests",
         "read line 9 of main.py"]


def valid(text):
    try:
        d = json.loads(text)
    except ValueError:
        return False
    return (isinstance(d, dict) and set(d) == {"tool", "path", "line"}
            and d["tool"] in SCHEMA["properties"]["tool"]["enum"]
            and isinstance(d["path"], str) and isinstance(d["line"], int))


def ask(task, mode):
    body = {"temperature": 0.7, "max_tokens": 80, "seed": 1,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content":
                          "You control a code editor. Reply with ONLY a JSON "
                          "object with keys tool (read_file, edit_file or "
                          f"run_tests), path and line. Request: {task}"}]}
    if mode == "schema":
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "call", "schema": SCHEMA}}
    if mode == "grammar":
        body["grammar"] = GBNF
    r = httpx.post(URL + "/v1/chat/completions", json=body, timeout=120).json()
    return r["choices"][0]["message"]["content"], r["timings"]


# one throwaway request first, so the first timing is not a cold start
ask(TASKS[0], "prompt")
print(f"{'mode':9} {'valid':>6} {'tok/s':>7} {'ms/request':>11} {'tokens/reply':>13}")
for mode in ("prompt", "schema", "grammar"):
    ok, speeds, times, toks = 0, [], [], []
    for task in TASKS:
        text, t = ask(task, mode)
        ok += valid(text)
        speeds.append(t["predicted_per_second"])
        times.append(t["prompt_ms"] + t["predicted_ms"])
        toks.append(t["predicted_n"])
    print(f"{mode:9} {ok:3d}/{len(TASKS)} {statistics.median(speeds):7.1f} "
          f"{statistics.median(times):11.0f} {statistics.mean(toks):13.1f}")
