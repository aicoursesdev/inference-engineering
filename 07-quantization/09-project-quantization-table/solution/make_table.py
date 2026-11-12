"""Build the project's quality, speed and memory table and apply the gate.

    uv run python make_table.py

Reads the logs and answers from lessons 07.5 and 07.8 (llama-perplexity,
task_eval.py) and the first-run llama-bench figures, and prints a Markdown
table with the gate's verdict and the ship rule's choice.
"""
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent.parent / "08-quality-gates" / "solution"))
import run_gate  # noqa: E402
from servinglab.quality import gate  # noqa: E402

GGUF = HERE.parent.parent / "05-gguf" / "solution"
SIZE_MB = {"F16": 1198, "Q8_0": 639, "Q4_K_M": 397, "Q3_K_M": 347, "Q2_K": 296}
DECODE = {"F16": 221.5, "Q8_0": 336.8, "Q4_K_M": 414.7, "Q3_K_M": 416.9,
          "Q2_K": 436.3}          # bench-first-run.txt, tokens/s, indicative
KV_BYTES_PER_TOKEN = 114_688      # Qwen3-0.6B in 16-bit, from its config


def rows():
    base_task = run_gate.task_accuracy("F16")
    out = [("16-bit", 1198, 221.5, 20.67, 0.0, 1.0, base_task, True, [])]
    for level in ["Q8_0", "Q4_K_M", "Q3_K_M", "Q2_K"]:
        m = run_gate.parse_kl_log(GGUF / f"kl-{level}.log")
        task = run_gate.task_accuracy(level)
        ok, why = gate({"ppl": m["ppl_base"], "task": base_task},
                       {**m, "task": task})
        out.append((level, SIZE_MB[level], DECODE[level], m["ppl"], m["kl"],
                    m["top1"], task, ok, why))
    return out


def ship_rule(table):
    """Ship the smallest build that passes the gate."""
    passing = [r for r in table if r[7]]
    return min(passing, key=lambda r: r[1])


if __name__ == "__main__":
    table = rows()
    print("| Build | File MB | Decode tok/s | Perplexity | KL | Top-1 | Task | Gate |")
    print("|---|---|---|---|---|---|---|---|")
    for n, mb, tps, ppl, kl, top1, task, ok, why in table:
        print(f"| {n} | {mb} | {tps:.0f} | {ppl:.2f} | {kl:.4f} | {top1:.3f} "
              f"| {task:.3f} | {'pass' if ok else 'FAIL'} |")
    print("\nShip:", ship_rule(table)[0])
