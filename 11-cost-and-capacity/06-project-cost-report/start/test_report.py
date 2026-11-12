import subprocess
import sys

ARGS = ["bench.json", "--rate", "1.00", "--prefill-tps", "907", "--api-in", "0.02", "--api-out", "0.04",
        "--peak-qps", "0.17", "--mean-in", "403", "--mean-out", "144"]


def test_the_report_has_the_lessons_three_sections_and_numbers():
    out = subprocess.run([sys.executable, "cost_report.py", *ARGS], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    for line in ("Capacity at the SLO: 2 requests/s, 165 output tokens/s.",
                 "- input $0.306 and output $1.682 per million tokens (full load)",
                 "at 30% utilisation $5.607",
                 "- 0.22 GPUs busy -> 2 with 30% headroom and 1 spare (11% busy at peak)",
                 "- break-even 52,821,997 requests a month"):
        assert line in out.stdout, line
