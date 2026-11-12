import subprocess
import sys


def gate(new, *extra):
    return subprocess.run([sys.executable, "gate.py", "canary-Q8_0.json", new, *extra], capture_output=True, text=True)


def test_a_one_question_drop_passes():
    r = gate("canary-Q4_K_M.json", "--allow-drop", "3")
    assert r.returncode == 0 and "GATE: pass" in r.stdout, r.stderr


def test_a_seven_question_drop_with_stray_script_fails():
    r = gate("canary-Q2_K.json", "--allow-drop", "3")
    assert r.returncode == 1 and "GATE: FAIL" in r.stdout, r.stderr


def test_a_wider_allowance_does_not_excuse_stray_script():
    r = gate("canary-Q2_K.json", "--allow-drop", "30")
    assert r.returncode == 1 and "GATE: FAIL" in r.stdout, r.stderr
