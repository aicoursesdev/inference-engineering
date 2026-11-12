#!/bin/bash
# Not run here (needs an NVIDIA GPU). vLLM v0.33.0, checked against its docs.
vllm serve Qwen/Qwen3-0.6B --dtype float16 --max-model-len 4096 --port 8000
