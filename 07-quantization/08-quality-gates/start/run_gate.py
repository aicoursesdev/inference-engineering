"""Apply servinglab.quality.gate to llama-perplexity logs and task answers.

    uv run python run_gate.py ../../05-gguf/solution

Reads kl-<LEVEL>.log (from `llama-perplexity --kl-divergence`) and
answers-Qwen3-0.6B-<LEVEL>.json (from task_eval.py) and prints one row per
build with PASS or FAIL and the reasons.
"""
import json
import re
import sys
from pathlib import Path

from my_quality import gate  # your gate; servinglab.quality.gate is the finished one

HERE = Path(__file__).parent


def parse_kl_log(path):
    """ppl, kl and top-1 agreement from a llama-perplexity KL log."""
    text = Path(path).read_text()
    num = lambda pat: float(re.search(pat, text).group(1))
    return {"ppl": num(r"Mean PPL\(Q\)\s+:\s+([\d.]+)"),
            "ppl_base": num(r"Mean PPL\(base\)\s+:\s+([\d.]+)"),
            "kl": num(r"Mean\s+KLD:\s+([\d.]+)"),
            "top1": num(r"Same top p:\s+([\d.]+)") / 100}


def task_accuracy(level):
    path = HERE / f"answers-Qwen3-0.6B-{level}.json"
    return json.loads(path.read_text())["accuracy"]


if __name__ == "__main__":
    logs = Path(sys.argv[1])
    base_task = task_accuracy("F16")
    for level in ["Q8_0", "Q4_K_M", "Q3_K_M", "Q2_K"]:
        m = parse_kl_log(logs / f"kl-{level}.log")
        base = {"ppl": m["ppl_base"], "task": base_task}
        test = {**m, "task": task_accuracy(level)}
        ok, why = gate(base, test)
        print(f"{level:7} ppl {m['ppl']:7.2f} ({m['ppl'] / m['ppl_base']:.3f}x)  "
              f"KL {m['kl']:.4f}  top-1 {m['top1']:.3f}  "
              f"task {test['task']:.3f}  {'PASS' if ok else 'FAIL'}  "
              + "; ".join(why))
