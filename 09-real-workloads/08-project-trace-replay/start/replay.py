"""Project 09.8: your trace-replay tool.

Everything but the TODOs is finished. The finished file is servinglab/replay.py,
which Modules 10 and 11 reuse; `make diff 09.8` shows what it has that this
file lacks.
"""

import asyncio
import hashlib
import json
import time
from dataclasses import dataclass, field

import httpx

from servinglab.data import traces

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
    raise NotImplementedError(
        "TODO 09.8: turn the trace into requests with filler text: a shared prefix is the same text, and each Kodekart step is the previous step's prompt plus a new tail, see the lesson's 'The steps' section")


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
    raise NotImplementedError(
        "TODO 09.8: read the cached prompt tokens from usage.prompt_tokens_details.cached_tokens, else timings.cache_n, see the lesson's 'The steps' section")


async def _send(client, url, model, req, cache, schema):
    """POST one streaming chat request and return its Row: time to first
    content chunk, token counts from the usage block, cached tokens, and
    whether the reply parses as JSON when a schema was asked for."""
    raise NotImplementedError(
        "TODO 09.8: send one streaming request, time the first chunk and read the usage and cache counts, see the lesson's 'The steps' section")


async def replay(requests, base_url, model, concurrency=4, speed=1.0,
                 think_s=0.0, cache=True, schema=None, timeout=600, transport=None):
    """Send every request; return one Row each, in trace order.

    `concurrency` sessions at once. `think_s` seconds (divided by `speed`)
    pass between a reply and that session's next step: the time a real agent
    spends running its tool. `model` may be a list of names (adapters): sessions take them in turn. `transport` lets a test stand in for the server. Errors become a Row with `error` set.
    """
    raise NotImplementedError(
        "TODO 09.8: run each session's steps in order, `concurrency` sessions at once, one Row per request in trace order, an error becoming a Row with `error` set, see the lesson's 'The steps' section")


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
    raise NotImplementedError(
        "TODO 09.8: hit rate, tokens, median time to first token and cost per task from uncached input, cached input and output tokens, see the lesson's 'The steps' section")


def trace_hit_rate(requests):
    """The most a perfect cache could save: reusable tokens / all prompt tokens,
    from the trace alone. A shared prefix counts only from its second use (the
    first request has to compute it); a Kodekart step's prefix is the previous
    step's whole prompt, which the server has just computed."""
    raise NotImplementedError(
        "TODO 09.8: reusable tokens over all prompt tokens, a shared prefix counting only from its second use and a Kodekart step's prefix always counting, see the lesson's 'The steps' section")
