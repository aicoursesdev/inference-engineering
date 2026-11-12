"""Step 5: put the predictions beside the measurements and look at the gap.

    uv run python compare.py predictions.json measurements.json

With no arguments it uses the files recorded on the M5 Max laptop for this course.
"""

import json
import sys
from pathlib import Path

from servinglab.data import DATA
from predict_first import predictions

RECORDED = DATA / "recorded" / "llama-bench-m5max.json"


def table(preds, measured):
    """One row per model file: predicted and measured speed in each phase, and the share reached."""
    by_bytes = {r["file_bytes"]: r for r in preds["predictions"]}
    rows = []
    for file, results in measured["per_model"].items():
        decode = next(r for r in results if r["n_gen"] > 0)
        prefill = next(r for r in results if r["n_prompt"] > 0)
        p = by_bytes[decode["model_size"]]
        rows.append({"file": Path(file).name, "bytes": decode["model_size"],
                     "decode_pred": p["decode_tokens_per_s"], "decode": decode["avg_ts"],
                     "prefill_pred": p["prefill_tokens_per_s"], "prefill": prefill["avg_ts"]})
    return rows


def fit_overhead(row):
    """Seconds per decode step the pure roofline leaves unexplained for this one row."""
    return 1 / row["decode"] - 1 / row["decode_pred"]


def render(rows):
    lines = [f"{'file':<24}{'decode pred':>12}{'measured':>9}{'share':>7}   {'prefill pred':>12}{'measured':>9}{'share':>7}"]
    for r in rows:
        lines.append(f"{r['file']:<24}{r['decode_pred']:>12.0f}{r['decode']:>9.0f}{r['decode'] / r['decode_pred']:>7.0%}"
                     f"   {r['prefill_pred']:>12.0f}{r['prefill']:>9.0f}{r['prefill'] / r['prefill_pred']:>7.0%}")
    return "\n".join(lines)


if __name__ == "__main__":
    measured = json.loads(Path(*sys.argv[2:3] or [RECORDED]).read_text())
    if len(sys.argv) > 1:
        preds = json.loads(Path(sys.argv[1]).read_text())
    else:  # the course's own laptop: the roof machine.py measured, files from the recorded run
        sizes = [(("Qwen3-1.7B" if "1.7B" in f else "Qwen3-0.6B"), next(r for r in rs if r["n_gen"] > 0)["model_size"])
                 for f, rs in measured["per_model"].items()]
        preds = predictions(sizes, bandwidth_gbs=406, tflops=48.4)
    rows = table(preds, measured)
    print(render(rows))
    extra = fit_overhead(rows[0])
    print(f"\nFirst row's unexplained time: {extra * 1000:.2f} ms per decode step.")
