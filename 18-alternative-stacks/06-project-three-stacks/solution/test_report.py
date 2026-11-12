import subprocess
import sys
from pathlib import Path

import report

HERE = Path(__file__).parent


def test_cost_formula_matches_the_lesson():
    assert round(report.usd_per_m(1, 2.0, 1000), 4) == 0.5556
    assert round(report.usd_per_m(8, 2.0, 8000), 4) == 0.5556   # eight GPUs, eight times the tokens


def test_rows_that_miss_the_tpot_limit_are_listed_not_ranked():
    rows = [dict(stack="slow", gpus="1", usd_per_gpu_hour="2", output_tok_s="3000", tpot_p95_ms="80"),
            dict(stack="b", gpus="1", usd_per_gpu_hour="2", output_tok_s="2000", tpot_p95_ms="40"),
            dict(stack="a", gpus="1", usd_per_gpu_hour="2", output_tok_s="1000", tpot_p95_ms="30")]
    ranked, bad, _ = report.rank(rows, 50)
    assert [r["stack"] for r in ranked] == ["b", "a"]    # cheapest first
    assert [r["stack"] for r in bad] == ["slow"]


def test_the_command_line_prints_a_ranking(tmp_path):
    csv = tmp_path / "r.csv"
    csv.write_text(open(HERE / "results_template.csv").read()
                   + "vllm,H100,1,2.0,1000,300,40\nslow,H100,1,2.0,5000,300,90\n")
    out = subprocess.run([sys.executable, str(HERE / "report.py"), str(csv)],
                         capture_output=True, text=True, check=True).stdout
    assert "$ 0.556 per M output tokens" in out and "slow" in out and "not ranked" in out
