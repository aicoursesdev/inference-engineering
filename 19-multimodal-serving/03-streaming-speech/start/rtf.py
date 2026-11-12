"""Real-time factor of whisper.cpp: seconds of work per second of audio.

python rtf.py ../../../models/ggml-base.en.bin short.wav speech.wav
RTF below 1 means it keeps up with live audio; 0.02 is fifty times faster.
"""
import re
import subprocess
import sys
import wave


def seconds(path):
    with wave.open(path) as w:
        return w.getnframes() / w.getframerate()


def timings(model, wav):
    out = subprocess.run(["whisper-cli", "-m", model, "-f", wav, "-nt"],
                         capture_output=True, text=True).stderr
    get = lambda k: float(re.search(rf"{k} time =\s+([\d.]+) ms", out)[1])
    return get("encode"), get("total")


if __name__ == "__main__":
    model, clips = sys.argv[1], sys.argv[2:]
    print(f"{'clip':12s} {'audio s':>8s} {'encode ms':>10s} "
          f"{'total ms':>9s} {'RTF':>6s}")
    for wav in clips:
        timings(model, wav)  # warm-up run, discarded
        enc, tot = min(timings(model, wav) for _ in range(5))
        a = seconds(wav)
        print(f"{wav:12s} {a:8.2f} {enc:10.1f} {tot:9.0f} {tot / 1000 / a:6.3f}")
