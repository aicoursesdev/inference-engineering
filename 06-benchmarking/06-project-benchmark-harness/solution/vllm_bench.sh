#!/bin/sh
# Monsoon's Tidewater-sized load against a vLLM server, on a rented GPU. Flags checked against the vLLM
# docs (docs.vllm.ai/en/latest/cli/bench/serve/) on 2026-11-12; run `vllm bench serve --help` for your version.
#
#   vllm serve Qwen/Qwen3-0.6B --dtype float16 --max-model-len 4096 --port 8000     # a T4 has no bfloat16
#   sh vllm_bench.sh
set -e
mkdir -p results
for rate in 1 2 4 8 16; do
  vllm bench serve \
    --backend vllm --base-url http://localhost:8000 --endpoint /v1/completions \
    --model Qwen/Qwen3-0.6B \
    --dataset-name random --random-input-len 341 --random-output-len 120 \
    --num-prompts 200 --request-rate "$rate" --burstiness 1.0 --ignore-eos --seed 0 \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,95,99 \
    --goodput ttft:1000 tpot:50 \
    --save-result --result-dir results --result-filename "tidewater-rate-$rate.json"
done
