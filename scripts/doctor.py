"""make doctor: check this machine, then play a recorded streamed reply.

Uses only Python's standard library, so it runs before anything else works:
no model, no GPU, no internet.
"""

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GB = 1024**3


def ok(text):
    print(f"  ok    {text}")


def warn(text):
    print(f"  !     {text}")


def note(text):
    print(f"        {text}")


def run(cmd):
    """The first line of a command's output, or None if it is missing or fails."""
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError):
        return None


def chip():
    if sys.platform == "darwin":
        return run(["sysctl", "-n", "machdep.cpu.brand_string"])
    if sys.platform.startswith("linux"):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or None


def total_ram_bytes():
    try:
        if sys.platform == "darwin":
            return int(run(["sysctl", "-n", "hw.memsize"]))
        if sys.platform.startswith("linux"):
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024  # the file counts in kB
        if sys.platform == "win32":
            import ctypes

            class Status(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("total", "a", "b", "c", "d", "e", "f")]

            s = Status()
            s.length = ctypes.sizeof(Status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            return s.total
    except Exception:
        pass
    return None


def nvidia_gpus():
    """[(name, memory in MiB)] for each visible NVIDIA GPU; [] when there is none."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return [(n.strip(), int(m)) for n, m in (line.rsplit(",", 1) for line in out.splitlines())]
    except (OSError, subprocess.SubprocessError, ValueError):
        return []


def play_recorded_reply():
    """A real reply recorded from llama-server, shown with the timing it arrived with."""
    rec = json.loads((ROOT / "data" / "recorded" / "llama-server-reply.json").read_text(encoding="utf-8"))
    m = rec["machine"]
    print(f"  Recorded {rec['recorded_at'][:10]} on {m['chip']} with llama.cpp {rec['llama_cpp']['version']}, "
          f"{rec['model']['file']}. No model is running now.")
    print(f"  Customer: {rec['request']['messages'][-1]['content']}\n")
    print("  Reply: ", end="", flush=True)
    live = sys.stdout.isatty()
    last, first_ms, pieces = 0, None, 0
    for e in rec["events"]:
        if e["data"] == "[DONE]":
            continue
        choice = json.loads(e["data"])["choices"]
        text = choice[0]["delta"].get("content") if choice and "delta" in choice[0] else None
        if not text:
            continue
        if live:
            time.sleep(min((e["t_ms"] - last) / 1000, 0.25))  # the recorded gaps, capped
        last = e["t_ms"]
        first_ms = e["t_ms"] if first_ms is None else first_ms
        pieces += 1
        print(text, end="", flush=True)
    print(f"\n\n  {pieces} streamed pieces; the first arrived {first_ms:.0f} ms after the request was sent.\n")


def main():
    print("\nInference Engineering: doctor\n")

    print("This machine")
    v = platform.python_version()
    (ok if sys.version_info >= (3, 12) else warn)(f"Python {v}" + ("" if sys.version_info >= (3, 12) else ": the course needs 3.12 or newer. Run `uv sync`."))
    ok(f"{platform.system()} {platform.release()} ({platform.machine()})")
    ok(f"chip: {chip() or 'unknown'}")
    ram = total_ram_bytes()
    ok(f"{ram / GB:.0f} GB of memory") if ram else warn("could not read the memory size")
    if ram and sys.platform == "darwin":
        note("On a Mac this is unified memory: the GPU shares it with the CPU.")

    print("\nTools on your PATH")
    for tool in ("llama-server", "llama-bench", "ollama", "uv", "git"):
        path = shutil.which(tool)
        ok(f"{tool}: {path}") if path else warn(f"{tool}: not found" + (" (needed from lesson 0.3; installing it is covered there)" if tool.startswith("llama") else ""))

    print("\nGPU")
    gpus = nvidia_gpus()
    for name, mem in gpus:
        ok(f"{name}, {mem / 1024:.0f} GiB")
    if not gpus:
        note("No NVIDIA GPU is visible (nvidia-smi not found or lists none). That is fine: most of the course runs on a laptop.")

    print("\nYour path")
    in_colab = "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ
    if gpus and all("T4" in n for n, _ in gpus):
        ok("Free T4" + (" (Colab)" if in_colab else "") + ": laptop lessons run here too; GPU lessons that fit a single 16 GB T4 run here, in float16.")
    elif gpus:
        ok("Rented GPU: everything runs here, including servers under load. Shut the machine down when you stop.")
    else:
        ok("Laptop only: laptop lessons run here; GPU lessons are read, or run later on a free T4 or a rented GPU.")

    print("\nA recorded reply (streamed, with its real timing)")
    play_recorded_reply()


if __name__ == "__main__":
    main()
