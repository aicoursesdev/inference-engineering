import subprocess
from pathlib import Path

import rtf

HERE = Path(__file__).parent


def test_clip_lengths_come_from_the_wav_header():
    assert round(rtf.seconds(str(HERE / "short.wav")), 2) == 1.24
    assert round(rtf.seconds(str(HERE / "speech.wav")), 2) == 9.77


def test_timings_reads_whispers_own_report(monkeypatch):
    stderr = ("whisper_print_timings:   encode time =    52.1 ms /     1 runs\n"
              "whisper_print_timings:    total time =   130.0 ms\n")
    fake = subprocess.CompletedProcess([], 0, stdout="", stderr=stderr)
    monkeypatch.setattr(rtf.subprocess, "run", lambda *a, **k: fake)
    assert rtf.timings("model.bin", "x.wav") == (52.1, 130.0)
    # real-time factor: 130 ms of work for 9.77 s of audio
    assert round(130.0 / 1000 / 9.77, 3) == 0.013
