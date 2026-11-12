"""Latency and throughput numbers from timed requests (built in lesson 00.5).

    from servinglab.metrics import Timed, summary, SLO

    req = Timed(sent=0.0, token_times=[0.21, 0.25, 0.30, 0.34])
    req.ttft, req.tpot          # 0.21 s to the first token, 0.043 s per token
    summary([req, ...], SLO["interactive"])

All times are seconds on one clock (time.perf_counter()). A request is
`sent` when you wrote it to the socket, and `token_times` holds the moment
each output token arrived.
"""

from dataclasses import dataclass, field

import numpy as np

# Monsoon Cloud's targets (docs: its own, not an industry standard):
# (time to first token, time per output token), both p95, in seconds.
# "batch" has none: it just has to finish.
SLO = {
    "interactive": (1.0, 0.050),   # Tidewater Goods
    "agent": (2.0, 0.050),         # Kodekart
    "long": (8.0, 0.080),          # Nyaya Docs
    "batch": (None, None),         # nightly batch
}


@dataclass
class Timed:
    sent: float
    token_times: list = field(default_factory=list)

    @property
    def n_tokens(self):
        return len(self.token_times)

    @property
    def ttft(self):
        """Time to first token: from sending the request to its first token."""
        return self.token_times[0] - self.sent

    @property
    def tpot(self):
        """Time per output token after the first: the average gap, or None
        for a one-token reply."""
        if self.n_tokens < 2:
            return None
        return (self.token_times[-1] - self.token_times[0]) / (self.n_tokens - 1)

    @property
    def itls(self):
        """Inter-token latencies: the gap before each token after the first."""
        return list(np.diff(self.token_times))

    @property
    def e2e(self):
        """End-to-end latency: sent until the last token."""
        return self.token_times[-1] - self.sent


def from_events(events, sent=0.0):
    """A Timed from a recorded reply: `events` as in data/recorded/*.json,
    a list of {t_ms, data}. Counts the chunks that carry text."""
    import json
    times = []
    for e in events:
        if e["data"] == "[DONE]":
            continue
        choice = json.loads(e["data"])["choices"][0]
        if choice["delta"].get("content"):
            times.append(sent + e["t_ms"] / 1000)
    return Timed(sent, times)


def timed_stream(url, body, timeout=120):
    """POST a streaming chat request to an OpenAI-style server and time it.

    Returns a Timed with one time per streamed chunk that carries text.
    (A chunk is normally one token; servers may pack more into one.)
    """
    import json
    import time

    import httpx

    sent = time.perf_counter()
    times = []
    with httpx.stream("POST", url, json={**body, "stream": True},
                      timeout=timeout) as reply:
        reply.raise_for_status()
        for line in reply.iter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                chunk = json.loads(line[6:])
                if chunk.get("choices") and chunk["choices"][0]["delta"].get("content"):
                    times.append(time.perf_counter())
    return Timed(sent, times)


def percentile(values, p):
    """The p-th percentile (0 to 100), interpolating between values."""
    return float(np.percentile(values, p))


def window(reqs):
    """Seconds from the first request sent to the last token received."""
    return max(r.token_times[-1] for r in reqs) - min(r.sent for r in reqs)


def throughput(reqs):
    """Output tokens per second, over the whole window (all requests)."""
    return sum(r.n_tokens for r in reqs) / window(reqs)


def meets(req, ttft_slo, tpot_slo):
    """Did this one request meet both targets? None means no target."""
    if ttft_slo is not None and req.ttft > ttft_slo:
        return False
    if tpot_slo is not None and req.tpot is not None and req.tpot > tpot_slo:
        return False
    return True


def goodput(reqs, ttft_slo, tpot_slo):
    """Requests per second that met both targets, over the same window."""
    return sum(meets(r, ttft_slo, tpot_slo) for r in reqs) / window(reqs)


def summary(reqs, slo=(None, None)):
    """Every headline number as one dict (seconds; tokens/s; requests/s)."""
    ttfts = [r.ttft for r in reqs]
    tpots = [r.tpot for r in reqs if r.tpot is not None]
    good = sum(meets(r, *slo) for r in reqs)
    return {
        "requests": len(reqs),
        "ttft_p50": percentile(ttfts, 50), "ttft_p95": percentile(ttfts, 95),
        "tpot_p50": percentile(tpots, 50) if tpots else None,
        "tpot_p95": percentile(tpots, 95) if tpots else None,
        "tokens_per_s": throughput(reqs),
        "requests_per_s": len(reqs) / window(reqs),
        "goodput_rps": goodput(reqs, *slo),
        "good_fraction": good / len(reqs),
    }
