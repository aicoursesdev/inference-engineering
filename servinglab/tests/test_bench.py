import asyncio

import httpx
import numpy as np
import pytest

from servinglab import bench
from servinglab.data import traces


class FakeReply(httpx.AsyncByteStream):
    """A server that waits `first` seconds, then streams n tokens `gap` seconds apart."""

    def __init__(self, n, first, gap):
        self.n, self.first, self.gap = n, first, gap

    async def __aiter__(self):
        await asyncio.sleep(self.first)
        for _ in range(self.n):
            yield b'data: {"choices": [{"text": " w"}]}\n\n'
            await asyncio.sleep(self.gap)
        yield b'data: {"choices": [], "usage": {"completion_tokens": %d, "prompt_tokens": 7}}\n\n' % self.n
        yield b"data: [DONE]\n\n"


def fake_server(first=0.05, gap=0.01):
    return httpx.MockTransport(lambda request: httpx.Response(200, stream=FakeReply(5, first, gap)))


def test_arrivals_have_the_asked_mean_and_burstiness_changes_the_spread():
    poisson = np.diff(bench.arrival_times(10, 5000, burstiness=1.0))
    bursty = np.diff(bench.arrival_times(10, 5000, burstiness=0.25))
    assert poisson.mean() == pytest.approx(0.1, rel=0.05) and bursty.mean() == pytest.approx(0.1, rel=0.1)
    assert bursty.std() > 1.5 * poisson.std()  # same average rate, much clumpier


def test_prefix_sharing_makes_prompts_start_alike_and_off_makes_them_differ():
    rows = [r for r in traces("kodekart") if r["prefix_id"] == "kk-system"][:3]
    shared = bench.make_requests(rows, max_input=3000)
    alone = bench.make_requests(rows, max_input=3000, share_prefix=False)
    start = lambda reqs: {r["prompt"][:200] for r in reqs}
    assert len(start(shared)) == 1 and len(start(alone)) == 3


def test_a_run_reads_ttft_and_tpot_from_the_stream():
    reqs = bench.make_requests(traces("tidewater")[:6], max_output=5)
    results, seconds = asyncio.run(bench.run("http://fake", "m", reqs, rate=50, transport=fake_server()))
    assert all(r["ok"] for r in results)
    assert all(0.05 <= r["ttft"] < 0.2 for r in results)
    assert all(0.009 <= r["tpot"] < 0.05 for r in results) and results[0]["output_tokens"] == 5


def test_summary_goodput_counts_only_requests_that_meet_both_targets():
    slo = {"ttft": 1.0, "tpot": 0.05}
    mk = lambda ttft, tpot: {"ok": True, "ttft": ttft, "tpot": tpot, "e2e": ttft + 1, "output_tokens": 10, "lag": 0}
    results = [mk(0.5, 0.04), mk(0.9, 0.05), mk(1.5, 0.04), mk(0.5, 0.09)]  # two good, one slow start, one slow stream
    s = bench.summarize(results, seconds=2.0, slo=slo, offered_rate=2)
    assert s["slo_attainment"] == 0.5 and s["goodput"] == 1.0 and s["request_throughput"] == 2.0
    assert s["ttft"]["p50"] == pytest.approx(0.7)  # between 0.5 and 0.9, linear interpolation


def test_capacity_is_the_highest_rate_that_met_the_target():
    runs = [{"offered_rate": r, "slo_attainment": a} for r, a in [(1, 1.0), (2, 0.95), (4, 0.9), (8, 0.4)]]
    assert bench.capacity(runs) == 4 and bench.capacity(runs[3:]) is None


def test_sweep_and_report_on_a_fake_server(monkeypatch):
    real_run = bench.run
    monkeypatch.setattr(bench, "run", lambda *a, **k: real_run(*a, **k, transport=fake_server()))
    reqs = bench.make_requests(traces("tidewater")[:8], max_output=5)
    data = bench.sweep("http://fake", "m", reqs, [20, 40], bench.SLOS["interactive"])
    text = bench.report(data)
    assert data["capacity_rate"] == 40 and "| 20 |" in text and "Capacity: 40 requests per second" in text


def test_a_concurrency_cap_runs_requests_in_waves():
    reqs = bench.make_requests(traces("tidewater")[:6], max_output=5)
    start = __import__("time").perf_counter()
    results, seconds = asyncio.run(bench.run("http://fake", "m", reqs, rate=float("inf"), max_concurrency=2,
                                             transport=fake_server(first=0.1, gap=0.0)))
    assert seconds >= 0.3 and max(r["lag"] for r in results) >= 0.2  # three waves of 0.1 s; the last wave left late
