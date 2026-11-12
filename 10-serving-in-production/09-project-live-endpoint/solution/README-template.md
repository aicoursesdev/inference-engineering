# Monsoon endpoint: README

Host: <provider, GPU, region>   Model: <name, quantization>   Engine: <llama.cpp or vLLM, version>

## What it costs
- GPU rate: <$/hour> (source: <page>, read <date>)
- If left running for a day: <$> ; for a month (730 h): <$>
- Spend cap set: <how>   Alert set: <how>

## Load test (Module 06 harness, <server URL>, <date>)
<paste the report table for three offered rates>
Capacity at Tidewater's target (TTFT p95 <= 1 s, TPOT p95 <= 50 ms): <req/s>

## Dashboard
<screenshot or description of the four lights: waiting, TTFT p95, KV-cache use, cache hits>

## How to turn it off
<the exact command, tested>
