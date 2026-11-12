#!/bin/sh
# Offline throughput on a free Colab T4: no server, the engine runs inside the command.
# Command as in the vLLM docs (docs.vllm.ai/en/latest/benchmarking/cli/), with --dtype float16 for the T4.
vllm bench throughput \
  --model Qwen/Qwen3-0.6B --dtype float16 \
  --dataset-name random --random-input-len 500 --random-output-len 150 \
  --num-prompts 10
