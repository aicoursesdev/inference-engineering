#!/bin/bash
# Run the task eval on several builds, one server at a time.
cd "$(dirname "$0")/../../.."
for m in models/Qwen3-0.6B-F16.gguf models/mine/Qwen3-0.6B-Q8_0.gguf models/mine/Qwen3-0.6B-Q4_K_M.gguf models/mine/Qwen3-0.6B-Q3_K_M.gguf models/mine/Qwen3-0.6B-Q2_K.gguf; do
  n=$(basename $m .gguf)
  llama-server -m $m --port 8080 -c 2048 >/dev/null 2>&1 &
  pid=$!
  until curl -sf localhost:8080/health >/dev/null; do sleep 1; done
  uv run python 07-quantization/08-quality-gates/solution/task_eval.py http://127.0.0.1:8080/v1 > 07-quantization/08-quality-gates/solution/answers-$n.json
  kill $pid; wait $pid 2>/dev/null
  echo $n $(python3 -c "import json;print(json.load(open('07-quantization/08-quality-gates/solution/answers-$n.json'))['accuracy'])")
done
