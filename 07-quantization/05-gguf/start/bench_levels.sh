#!/bin/bash
# Speed of each level with llama-bench (pp512 and tg128, 5 repeats), one at a
# time, with the machine's load average noted before and after.
cd "$(dirname "$0")/../../.."
OUT=07-quantization/05-gguf/solution
: > $OUT/bench-load.txt
for q in F16 Q8_0 Q6_K Q5_K_M Q4_K_M Q4_0 Q3_K_M Q2_K; do
  if [ $q = F16 ]; then m=models/Qwen3-0.6B-F16.gguf; else m=models/mine/Qwen3-0.6B-$q.gguf; fi
  echo "$q before $(uptime | sed 's/.*averages: //')" >> $OUT/bench-load.txt
  llama-bench -m $m -p 512 -n 128 -r 5 -o json 2>/dev/null > $OUT/bench-$q.json
  echo "$q after  $(uptime | sed 's/.*averages: //')" >> $OUT/bench-load.txt
done
