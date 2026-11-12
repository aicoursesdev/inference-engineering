"""Step 3: write down predictions from the roof alone, before looking at any measurement.

    uv run python predict_first.py --bandwidth 406 --tflops 48.4 \\
        Qwen3-0.6B=models/Qwen3-0.6B-Q8_0.gguf Qwen3-1.7B=models/Qwen3-1.7B-Q8_0.gguf

Each argument is a config name from data/models/ and the model file that run will load. The file's
size over the config's parameter count gives its bytes per weight. Writes predictions.json.
"""

import argparse
import json
import os

from servinglab.roofline import predict, shape
from servinglab.data import model_config

PROMPT = 512  # llama-bench's prompt length in record_measurements.py


def predictions(models, bandwidth_gbs, tflops):
    raise NotImplementedError(
        "TODO 02.7: for each (name, file bytes) use predict with bytes per weight = file bytes over the config's parameters, context 64 and the 512-token prompt, see the lesson's 'The steps' section")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bandwidth", type=float, required=True, help="GB/s, from machine.py")
    ap.add_argument("--tflops", type=float, required=True, help="float16 TFLOPS, from machine.py")
    ap.add_argument("models", nargs="+", help="config-name=path/to/file.gguf")
    ap.add_argument("--out", default="predictions.json")
    a = ap.parse_args()
    pairs = [(m.split("=")[0], os.path.getsize(m.split("=", 1)[1])) for m in a.models]
    result = predictions(pairs, a.bandwidth, a.tflops)
    json.dump(result, open(a.out, "w"), indent=1)
    for r in result["predictions"]:
        print(f"{r['model']:<12} decode {r['decode_tokens_per_s']:>6.0f} tokens/s, prefill {r['prefill_tokens_per_s']:>7.0f} tokens/s")
