# incidents/: the three failures of lesson 12.7's drill

Made on 12 November 2026 on an Apple M5 Max laptop (macOS 26.6.2, 36 GB), which was also running other jobs, so treat the milliseconds as one recording, not a benchmark. Files 1 and 3 are real runs; file 2 is a simulation and says so in its first line.

## 1. A latency regression (real)

`llama-server` build 11960 (commit 8642541b2), `Qwen3-0.6B-Q8_0.gguf` from `Qwen/Qwen3-0.6B-GGUF`, started with `-np 4 -c 24576 --jinja --metrics --slots` (`incident1-server.log` is its log). Four Kodekart-style agents run 8 steps each at once with `12-operations/07-project-incident-drill/solution/record_latency_incident.py`: the same 2,200-token system prompt and a history that grows by 280 tokens a step.

- `incident1-good.*`: the build before the deploy (`cache_prompt` true, llama.cpp's default).
- `incident1-bad.*`: the build after it (`cache_prompt` false, as if a gateway change had switched prefix reuse off).
- `*.jsonl`: one line per request with the client's clock (`sent_s`, `ttft_ms`, `total_ms`) and the server's own `timings` (`prompt_n`, `cache_n`, `prompt_ms`, `predicted_n`, `predicted_ms`).
- `*.metrics.txt`: the server's `/metrics` after each run. Counters are cumulative: the bad file also holds the good run, so subtract.

## 2. An out-of-memory crash (SIMULATED)

`12-operations/07-project-incident-drill/solution/simulate_oom_incident.py` writes `incident2-metrics.jsonl` and `incident2-server.log`. Sizes are computed from Qwen3-4B's real `config.json` and the L4's 24 GB; the arrival times are `traces/batch.jsonl`; the service times and the neighbour process's growth are assumptions, listed at the top of the script. No GPU or engine produced this file: the log's lines follow PyTorch's out-of-memory message but were written by the script.

## 3. A silent quality regression (real)

`12-operations/05-silent-quality-bugs/solution/canary.py` sends 30 questions with one checkable answer each, at temperature 0, thinking off, to `llama-server` with `Qwen3-0.6B-Q8_0.gguf` (`incident3-good-Q8_0.json`, 25 of 30) and with `Qwen3-0.6B-Q2_K.gguf` (`incident3-bad-Q2_K.json`, 18 of 30): the "deploy" that swapped in a smaller file. Latency and error rates would not have moved.

## The 12.4 exercise (real)

`gap-trace.jsonl` and `gap-server.log`: three clients send a short chat request 50 ms apart to a `llama-server` with `-np 1 -c 4096 --metrics` (one slot), `Qwen3-0.6B-Q8_0.gguf`, same machine and build as above, recorded with `12-operations/04-find-the-gap/solution/record_gap.py`. Client 1's first token took 380.5 ms although the server's own prefill took 34.0 ms.
