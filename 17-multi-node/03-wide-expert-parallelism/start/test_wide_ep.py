import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent


def output(script, *args):
    """What the script prints (it needs no network, no GPU)."""
    done = subprocess.run([sys.executable, str(HERE / script), *args],
                          capture_output=True, text=True, check=True)
    return done.stdout


def test_deepseek_cost_and_margin_reproduce():
    out = output("wide_ep.py")
    assert "= $87,072   (page: $87,072)" in out
    assert "margin: 546%" in out


def test_experts_per_gpu_for_prefill_and_decode():
    out = output("wide_ep.py")
    assert "EP32: 256+32 experts / 32 GPUs = 9 routed experts per GPU" in out
    assert "EP144: 256+32 experts / 144 GPUs = 2 routed experts per GPU" in out
