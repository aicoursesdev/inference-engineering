#!/bin/bash
# The laptop part of the project: three engines, one at a time, in three
# rounds (llama.cpp, Ollama, MLX-LM, then again) so a busy minute on the
# laptop hits every engine alike. Each row records the machine's load.
cd "$(dirname "$0")/../../.."
OUT=05-engines/09-project-three-engines/solution/results.csv
M=05-engines/09-project-three-engines/solution/measure_engine.py
rm -f $OUT
for round in 1 2 3; do
  llama-server -m models/Qwen3-0.6B-Q8_0.gguf --port 8080 -c 4096 >/dev/null 2>&1 &
  pid=$!; until curl -sf localhost:8080/health >/dev/null; do sleep 1; done
  uv run python $M --engine llama.cpp --version "0.8.0 build 11960" \
    --file Qwen3-0.6B-Q8_0.gguf --quant "Q8_0" \
    --base http://127.0.0.1:8080/v1 --model qwen3 --out $OUT
  kill $pid; wait $pid 2>/dev/null; sleep 2

  OLLAMA_KEEP_ALIVE=1m ollama serve >/dev/null 2>&1 &
  pid=$!; until curl -sf localhost:11434/api/tags >/dev/null; do sleep 1; done
  uv run python $M --engine Ollama --version "0.40.15" \
    --file "Qwen3-0.6B-Q8_0.gguf (imported)" --quant "Q8_0" \
    --base http://127.0.0.1:11434/v1 --model qwen3-06b-q8 --out $OUT
  kill $pid; wait $pid 2>/dev/null; sleep 2

  uv run --with mlx-lm mlx_lm.server --model mlx-community/Qwen3-0.6B-8bit --port 8081 >/dev/null 2>&1 &
  pid=$!; until curl -sf localhost:8081/v1/models >/dev/null; do sleep 1; done
  uv run python $M --engine MLX-LM --version "0.32.2" \
    --file mlx-community/Qwen3-0.6B-8bit --quant "8-bit, group 64" \
    --base http://127.0.0.1:8081/v1 --model mlx-community/Qwen3-0.6B-8bit --out $OUT
  pkill -f mlx_lm.server; sleep 2
done
