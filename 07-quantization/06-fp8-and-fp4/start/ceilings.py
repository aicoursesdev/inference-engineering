"""Ceilings for one 8.03B-parameter model on each chip, by number format.

    uv run python ceilings.py

decode (tokens/s, one user)  = bandwidth / weight bytes: any chip can STORE
                               the format and convert it, so it always counts
prefill (tokens/s)           = dense TFLOPS / (2 x parameters): only if the
                               chip has tensor cores for the format (else "-")
These are ceilings, not predictions (Module 02).
"""
from servinglab.data import hardware

PARAMS = 8.03e9                                      # Llama 3.1 8B, from its config
BITS = {"fp16": 16, "fp8": 8, "fp4": 4.5}            # 4.5 = NVFP4 with its scales


def row(name):
    h = hardware(name)
    out = [f"{name:14}"]
    for fmt, bits in BITS.items():
        tflops = h.get(f"{fmt}_tflops")
        decode = h["bandwidth_gbs"] * 1e9 / (PARAMS * bits / 8)
        prefill = f"{tflops * 1e12 / (2 * PARAMS):8,.0f}" if tflops else "       -"
        out.append(f"{fmt}: {decode:5,.0f} dec {prefill} pre".ljust(30))
    return "  ".join(out)


if __name__ == "__main__":
    for chip in ["T4", "L4", "A100-80GB-SXM", "H100-SXM", "B200", "MI300X"]:
        print(row(chip))
