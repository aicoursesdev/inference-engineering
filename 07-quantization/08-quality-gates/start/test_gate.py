"""Offline checks for run_gate.py: a real log excerpt, no model needed."""
import run_gate
from my_quality import gate

LOG = """\
Mean PPL(Q)                   :  22.415193 ±   0.630015
Mean PPL(base)                :  20.671296 ±   0.567311
Mean    KLD:   0.104539 ±   0.001247
Same top p: 83.592 ± 0.310 %
"""


def test_parse_real_log_excerpt(tmp_path):
    p = tmp_path / "kl.log"
    p.write_text(LOG)
    m = run_gate.parse_kl_log(p)
    assert m == {"ppl": 22.415193, "ppl_base": 20.671296,
                 "kl": 0.104539, "top1": 0.83592}


def test_q4_style_build_fails_and_names_reasons(tmp_path):
    p = tmp_path / "kl.log"
    p.write_text(LOG)
    m = run_gate.parse_kl_log(p)
    ok, why = gate({"ppl": m["ppl_base"], "task": 0.675}, {**m, "task": 0.75})
    assert not ok and len(why) == 3
