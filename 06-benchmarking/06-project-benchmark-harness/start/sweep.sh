#!/bin/sh
# One request-rate sweep of Monsoon's Tidewater traffic against a local llama.cpp server.
#   llama-server -m models/Qwen3-0.6B-Q8_0.gguf --port 8080 -c 16384 -np 4
#   sh sweep.sh
uv run python -m servinglab.bench --url http://localhost:8080 --model local --trace tidewater \
    --rates 1,2,4,6,8,12,16 --requests 100 --slo interactive --out tidewater-sweep.json
