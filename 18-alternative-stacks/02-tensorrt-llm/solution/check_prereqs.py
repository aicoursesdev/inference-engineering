"""Does this machine meet TensorRT-LLM's install page, as of Nov 2026?

Page: Python 3.12 and Ubuntu 24.04 (what NVIDIA tested), CUDA Toolkit 13.1
with CUDA_HOME set, PyTorch 2.10.0 built for cu130, libopenmpi-dev.
"""
import os, platform, shutil, subprocess, sys


def run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


checks = []
checks.append(("Linux", platform.system() == "Linux", platform.platform()))
checks.append(("Python 3.12", sys.version_info[:2] == (3, 12), sys.version.split()[0]))
smi = shutil.which("nvidia-smi")
gpu = run("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader") if smi else ""
checks.append(("NVIDIA GPU visible", bool(gpu.strip()), gpu.strip() or "nvidia-smi not found"))
nvcc = run("nvcc", "--version")
checks.append(("CUDA Toolkit 13.1", "release 13.1" in nvcc,
               nvcc.strip().splitlines()[-1] if nvcc else "nvcc not found"))
checks.append(("CUDA_HOME set", bool(os.environ.get("CUDA_HOME")), os.environ.get("CUDA_HOME", "unset")))
try:
    import torch
    ok = torch.__version__.startswith("2.10.0") and (torch.version.cuda or "").startswith("13.0")
    checks.append(("PyTorch 2.10.0 + cu130", ok, f"{torch.__version__}, cuda {torch.version.cuda}"))
except ImportError:
    checks.append(("PyTorch 2.10.0 + cu130", False, "not installed"))

for name, ok, detail in checks:
    print(f"{'ok  ' if ok else 'MISS'} {name:24} {detail}")
sys.exit(0 if all(ok for _, ok, _ in checks) else 1)
