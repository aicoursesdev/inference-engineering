"""One report for three modalities: pictures, speech and embeddings.

Starts each server in turn (one model at a time), measures it, stops it,
and prints the table the project hands in. Settings come from the
environment so the same file works on any machine:
  LLAMA_SERVER, WHISPER_SERVER, MODELS (a folder of model files)
"""
import os
import subprocess
import time
import wave

import httpx

from embed_llamacpp import run as embed_run
from image_tokens import QWEN25, image_tokens
from measure_images import best
from servinglab.cost import hourly_rate
from servinglab.ledger import record

LLAMA = os.environ.get("LLAMA_SERVER", "llama-server")
WHISPER = os.environ.get("WHISPER_SERVER", "whisper-server")
MODELS = os.environ.get("MODELS", "models")
TTS_FIRST_AUDIO_S = 0.30   # an assumption until you measure a real TTS server
LLM_TEXT_ONLY = 26         # the chat template and question, in tokens


class Server:
    def __init__(self, cmd, port, health="/health"):
        self.cmd, self.url, self.health = cmd, f"http://localhost:{port}", health

    def __enter__(self):
        self.p = subprocess.Popen(self.cmd, stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                if httpx.get(self.url + self.health).status_code < 400:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(1)  # still loading
        self.p.terminate()
        raise RuntimeError("server did not start")

    def __exit__(self, *exc):
        self.p.terminate()
        self.p.wait()


def usd(seconds, gpu="L4"):
    """Seconds of work priced at a rented GPU's cheapest listed rate."""
    raise NotImplementedError("TODO 19.6: seconds x the cheapest listed hourly rate of the GPU / 3600 (servinglab.cost.hourly_rate), see the lesson's 'Build it' section")


def predicted_prompt_tokens(w, h):
    """What a picture question should report: the image's tokens, the text
    prompt, and the 2 start and end markers."""
    raise NotImplementedError("TODO 19.6: image_tokens for the picture, plus LLM_TEXT_ONLY, plus 2 markers, see the lesson's 'Build it' section")


def voice_turn_ms(asr_s, llm_s, tts_s=TTS_FIRST_AUDIO_S):
    """Time to the first word of a spoken answer, in milliseconds."""
    raise NotImplementedError("TODO 19.6: recognition + first LLM token + first audio, in milliseconds, see the lesson's 'Build it' section")


def pictures():
    cmd = [LLAMA, "-m", f"{MODELS}/Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf",
           "--mmproj", f"{MODELS}/mmproj-Qwen2.5-VL-3B-Instruct-Q8_0.gguf",
           "-c", "4096", "-np", "1", "--port", "8091"]
    rows = []
    with Server(cmd, 8091) as s:
        text_n, text_t = best(s.url)
        for w, h in [(448, 448), (1280, 720), (1920, 1080)]:
            n, t = best(s.url, f"img_{w}x{h}.png")
            predicted = predicted_prompt_tokens(w, h)
            rows.append((f"{w}x{h}", n, predicted, t, usd(t)))
    return text_t, rows


def speech():
    out = []
    cmd = [WHISPER, "-m", f"{MODELS}/ggml-base.en.bin", "--port", "8095"]
    with Server(cmd, 8095, health="/") as s:
        for clip in ("short.wav", "speech.wav"):
            secs = wave.open(clip).getnframes() / 16000
            times = []
            for _ in range(6):
                t0 = time.perf_counter()
                r = httpx.post(s.url + "/inference", timeout=60,
                               files={"file": open(clip, "rb")},
                               data={"response_format": "json"})
                times.append(time.perf_counter() - t0)
            t = min(times[1:])  # drop the warm-up, keep the quietest run
            out.append((clip, secs, t, t / secs, r.json()["text"].strip()))
    return out


def embeddings():
    emb = os.environ["EMBED_GGUF"]
    cmd = [LLAMA, "-m", emb, "--embeddings", "--pooling", "last",
           "-b", "4096", "-ub", "4096", "-np", "1", "--port", "8092"]
    with Server(cmd, 8092) as s:
        embed_run(8, 16)
        return max((embed_run(16) for _ in range(3)), key=lambda r: r[1])


if __name__ == "__main__":
    text_ttft, pics = pictures()
    asr = speech()
    docs_s, toks_s = embeddings()
    print("PICTURES        tokens (predicted)   TTFT s   $ per question")
    for name, n, pred, t, c in pics:
        print(f"  {name:12s} {n:6d} ({pred:6d})   {t:6.2f}   {c:.6f}")
    print("SPEECH          audio s   work s    RTF   text")
    for clip, secs, t, rtf, text in asr:
        print(f"  {clip:12s} {secs:7.2f}  {t:7.3f}  {rtf:6.3f}   {text[:40]}")
    print(f"EMBEDDINGS      {docs_s:.0f} docs/s, {toks_s:.0f} tokens/s at batch 16")
    budget_ms = voice_turn_ms(asr[0][2], text_ttft)
    print(f"VOICE TURN      {asr[0][2]*1000:.0f} ASR + {text_ttft*1000:.0f} LLM"
          f" + {TTS_FIRST_AUDIO_S*1000:.0f} TTS (assumed) = {budget_ms:.0f} ms")
    for name, n, pred, t, c in pics:
        record({"lesson": "19.6", "engine": "llama.cpp", "model": "Qwen2.5-VL-3B-Q4_K_M",
                "hardware": "M5-Max", "change": f"image {name}", "ttft_ms": round(t * 1000),
                "note": f"{n} prompt tokens; laptop seconds priced at L4"})
    record({"lesson": "19.6", "engine": "whisper.cpp", "model": "base.en",
            "hardware": "M5-Max", "change": "transcribe speech.wav",
            "ttft_ms": round(asr[1][2] * 1000), "note": f"RTF {asr[1][3]:.3f}"})
    record({"lesson": "19.6", "engine": "llama.cpp", "model": "Qwen3-Embedding-0.6B-Q8_0",
            "hardware": "M5-Max", "change": "embed batch 16", "tok_per_s": round(toks_s)})
