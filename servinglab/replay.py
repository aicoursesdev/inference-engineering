"""Module 09: replay one of Monsoon's traces against any OpenAI-compatible endpoint.

    from servinglab.replay import build_requests, replay, report, Prices

    reqs = build_requests("kodekart", scale=0.1, tasks=4)
    rows = asyncio.run(replay(reqs, "http://localhost:8080/v1", "qwen3"))
    print(report(rows, Prices(input=0.32, cached_input=0.032, output=0.89)))

A trace holds token counts, not text, so `build_requests` writes the text:
a deterministic filler of common one-token words, with the same start wherever
the trace says two requests share a prefix. A Kodekart step's prompt is the
previous step's prompt plus a new tool result, exactly the shape of a real
agent loop. (The replies are NOT fed back: each prompt is fixed in advance,
so every run, sampled or not, sees the same prefixes. A real agent would
append the model's own words.)

Sessions run in order, each step waiting for the last reply (an agent cannot
ask for step 5 before it has step 4's answer); different sessions run at the
same time, `concurrency` at once. Tidewater, Nyaya and the batch have no
sessions: every request is its own, sent at its scaled arrival time.

Prefix reuse is read from the server's own report: `usage.prompt_tokens_details
.cached_tokens` (vLLM, SGLang, OpenAI) or `timings.cache_n` (llama.cpp).
`cache=False` asks llama.cpp not to reuse (`cache_prompt: false`); on vLLM,
start the server with and without `--disable-prefix-caching` instead (and with
`--enable-prompt-tokens-details`, or it sends no cached count).

Shortcuts, each with its limit: filler words are one token on most tokenizers
but not all, so `build_requests` aims at the trace's token counts and the
report counts what the server says it received; and a step's gap
(`think_s`) is the trace's, scaled by `speed`, not measured.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field

import httpx

from .data import traces

WORDS = ("the of and to in is that for it as with was on be by at this have "
         "from or one had not but what all were when we there can an your "
         "which their said if do will each about how up out them then she "
         "many some so these would other into has more her two like him see "
         "time could no make than first been its who now people my made over "
         "did down only way find use may water long little very after words "
         "called just where most know get through back much before go good "
         "new write our me man too any day same right look think also around "
         "another came come work three word must because does part even place "
         "well such here take why things help put years different away again "
         "off went old number great tell men say small every found still "
         "between name should home big give air line set own under read last "
         "never us left end along while might next sound below saw something "
         "thought both few those always looked show large often together "
         "asked house world going want school important until form food keep "
         "children feet land side without boy once animals life enough took "
         "sometimes four head above kind began almost live page got earth "
         "need far hand high year mother light country father let night "
         "picture being study second soon story since white ever paper "
         "hard near sentence better best across during today however sure "
         "knew try told young sun thing whole hear example heard several "
         "change answer room sea against top turned learn point city play "
         "toward five using himself usually money seen didn't car morning "
         "body upon family later turn move face door cut done group true "
         "leave color red friend pull dog").split()


def filler(seed, n_words):
    """n_words words, the same every time for the same seed."""
    out, h = [], hashlib.sha256(str(seed).encode()).digest()
    while len(out) < n_words:
        h = hashlib.sha256(h).digest()
        out += [WORDS[b % len(WORDS)] for b in h]
    return " ".join(out[:n_words])


@dataclass
class Req:
    id: str
    session: str       # steps of one session run in order
    arrival_s: float   # trace time, seconds since midnight
    prompt: str
    max_tokens: int
    input_tokens: int  # what the trace says (scaled), for reference
    prefix_tokens: int  # how much of it the trace says is reused (scaled)
    prefix_key: str = ""  # the trace's prefix_id


def build_requests(name, scale=1.0, tasks=None, limit=None, max_tokens=32):
    """Turn a trace into requests with text. `scale` shrinks every length (a
    laptop cannot prefill 15,000 tokens a step); `tasks` keeps only the
    first N Kodekart tasks; `limit` keeps the first N requests."""
    rows = traces(name)
    first_of = {}  # Kodekart: the n-th step-1 request starts task n
    chain, out, steps = {}, [], 0
    for r in rows:
        n_in = max(1, round(r["input_tokens"] * scale))
        n_pre = min(n_in, round(r["prefix_tokens"] * scale))
        pid = r["prefix_id"]
        if r["customer"] == "kodekart":
            if pid == "kk-system":
                task = f"kk-task-{len(first_of):02d}"
                first_of[task] = r["id"]
            else:
                task = pid
            if tasks is not None and int(task[-2:]) >= tasks:
                continue
            head = chain.get(task) or filler(pid, n_pre)
            session = task
        else:
            head, session = filler(pid, n_pre), r["id"]
        text = (head + " " + filler(r["id"], max(0, n_in - n_pre))).strip()
        if r["customer"] == "kodekart":
            chain[task] = text
        out.append(Req(r["id"], session, r["arrival_s"], text,
                       min(max_tokens, max(1, r["output_tokens"])), n_in, n_pre, pid))
    return out[:limit] if limit else out


@dataclass
class Row:
    id: str
    session: str
    prompt_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    ttft_s: float = 0.0
    total_s: float = 0.0
    valid_json: bool | None = None  # None when no schema was asked for
    error: str = ""


def cached_of(body):
    """Cached prompt tokens from a reply, in whichever dialect the server speaks."""
    u = body.get("usage", {})
    d = (u.get("prompt_tokens_details") or {}).get("cached_tokens")
    if d is not None:
        return d
    return (body.get("timings") or {}).get("cache_n", 0)


async def _send(client, url, model, req, cache, schema):
    payload = {"model": model, "temperature": 0, "max_tokens": req.max_tokens,
               "messages": [{"role": "user", "content": req.prompt}],
               "stream": True, "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False},
               "timings_per_token": False}
    if not cache:
        payload["cache_prompt"] = False
    if schema:
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "reply", "schema": schema}}
    row, text, last = Row(req.id, req.session), [], {}
    t0 = time.perf_counter()
    async with client.stream("POST", url + "/chat/completions",
                             json=payload) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: ") or line.endswith("[DONE]"):
                continue
            chunk = json.loads(line[6:])
            for c in chunk.get("choices", []):
                piece = c.get("delta", {}).get("content")
                if piece:
                    if not row.ttft_s:
                        row.ttft_s = time.perf_counter() - t0
                    text.append(piece)
            if "usage" in chunk or "timings" in chunk:
                last = {**last, **chunk}
    row.total_s = time.perf_counter() - t0
    u = last.get("usage", {})
    row.prompt_tokens = u.get("prompt_tokens", 0)
    row.output_tokens = u.get("completion_tokens", 0)
    row.cached_tokens = cached_of(last)
    if schema:
        try:
            json.loads("".join(text))
            row.valid_json = True
        except ValueError:
            row.valid_json = False
    return row


async def replay(requests, base_url, model, concurrency=4, speed=1.0,
                 think_s=0.0, cache=True, schema=None, timeout=600, transport=None):
    """Send every request; return one Row each, in trace order.

    `concurrency` sessions at once. `think_s` seconds (divided by `speed`)
    pass between a reply and that session's next step: the time a real agent
    spends running its tool. `model` may be a list of names (adapters): sessions take them in turn. `transport` lets a test stand in for the server. Errors become a Row with `error` set.
    """
    sessions = {}
    for r in requests:
        sessions.setdefault(r.session, []).append(r)
    gate = asyncio.Semaphore(concurrency)
    rows = {}

    async def run(steps, model):
        async with gate:
            for i, req in enumerate(steps):
                if i and think_s:
                    await asyncio.sleep(think_s / speed)
                try:
                    rows[req.id] = await _send(client, base_url, model, req,
                                               cache, schema)
                except (httpx.HTTPError, ValueError) as e:
                    rows[req.id] = Row(req.id, req.session,
                                       error=f"{type(e).__name__}: {e}")

    # a list of model names (say, LoRA adapters) is dealt out one per session
    names = [model] if isinstance(model, str) else list(model)
    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        await asyncio.gather(*(run(s, names[i % len(names)])
                               for i, s in enumerate(sessions.values())))
    return [rows[r.id] for r in requests]


@dataclass
class Prices:
    """Dollars per million tokens. `cached_input` is what a cache hit costs
    (many APIs charge a tenth of `input`); set it equal to `input` to see the
    bill with no caching discount."""
    input: float
    output: float
    cached_input: float = None

    def __post_init__(self):
        if self.cached_input is None:
            self.cached_input = self.input


def report(rows, prices, per="session"):
    """Hit rate, tokens and cost per task (`per="session"`) or per request."""
    ok = [r for r in rows if not r.error]
    prompt = sum(r.prompt_tokens for r in ok)
    cached = sum(r.cached_tokens for r in ok)
    out = sum(r.output_tokens for r in ok)
    cost = ((prompt - cached) * prices.input + cached * prices.cached_input
            + out * prices.output) / 1e6
    n = len({r.session for r in ok}) if per == "session" else len(ok)
    ttfts = sorted(r.ttft_s for r in ok if r.ttft_s)
    p50 = ttfts[len(ttfts) // 2] if ttfts else 0.0
    return {"requests": len(ok), "errors": len(rows) - len(ok),
            "prompt_tokens": prompt, "cached_tokens": cached,
            "hit_rate": cached / prompt if prompt else 0.0,
            "output_tokens": out, "cost_usd": cost,
            "cost_per_task_usd": cost / n if n else 0.0,
            "ttft_p50_s": p50}


def trace_hit_rate(requests):
    """The most a perfect cache could save: reusable tokens / all prompt tokens,
    from the trace alone. A shared prefix counts only from its second use (the
    first request has to compute it); a Kodekart step's prefix is the previous
    step's whole prompt, which the server has just computed."""
    total = sum(r.input_tokens for r in requests)
    seen, hits = set(), 0
    for r in requests:
        if r.prefix_key in seen or r.prefix_key.startswith("kk-task"):
            hits += r.prefix_tokens
        seen.add(r.prefix_key)
    return hits / total if total else 0.0
