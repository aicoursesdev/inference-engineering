"""Number formats from their bit layouts, and their rounding error on a
real Qwen3-0.6B weight matrix. Run:

    uv run --with gguf --with ml_dtypes python formats.py
"""
import numpy as np
from gguf import GGUFReader
from ml_dtypes import float8_e4m3fn, float8_e5m2

from servinglab.quality import rms_error, round_to_bits

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6])      # E2M1 magnitudes
GRID = np.concatenate([-FP4[:0:-1], FP4])         # both signs


def to_fp4(x):
    """Round each value to the nearest FP4 (E2M1) value."""
    return GRID[np.abs(x[..., None] - GRID).argmin(-1)]


def mxfp4(w, block=32):
    """OCP MX: a power-of-two scale (E8M0) per 32 values, FP4 elements."""
    b = w.reshape(-1, block)
    amax = np.abs(b).max(1, keepdims=True)
    scale = 2.0 ** (np.floor(np.log2(np.maximum(amax, 1e-30))) - 2)  # emax of E2M1 is 2
    return (to_fp4(np.clip(b / scale, -6, 6)) * scale).reshape(w.shape)


def nvfp4(w, block=16):
    """NVIDIA: an FP8 (E4M3) scale per 16 values, plus one FP32 per tensor."""
    b = w.reshape(-1, block)
    tensor = np.abs(w).max() / (6 * 448)
    scale = (np.abs(b).max(1, keepdims=True) / 6 / tensor).astype(float8_e4m3fn)
    scale = scale.astype(np.float32) * tensor
    scale[scale == 0] = 1.0
    return (to_fp4(np.clip(b / scale, -6, 6)) * scale).reshape(w.shape)


def fp8(w, dtype):
    return w.astype(dtype).astype(np.float32)


def load():
    r = GGUFReader("models/Qwen3-0.6B-F16.gguf")
    t = next(t for t in r.tensors if t.name == "blk.10.ffn_down.weight")
    raw = np.ascontiguousarray(t.data).view(np.uint16).astype(np.uint32) << 16
    return raw.view(np.float32).reshape(1024, 3072)


if __name__ == "__main__":
    w = load()
    std = w.std()
    rows = {"FP8 E4M3 (no scale)": fp8(w, float8_e4m3fn),
            "FP8 E5M2 (no scale)": fp8(w, float8_e5m2),
            "INT8, groups of 32": round_to_bits(w, 8, 32),
            "INT4, groups of 32": round_to_bits(w, 4, 32),
            "MXFP4 (block 32, E8M0)": mxfp4(w),
            "NVFP4 (block 16, E4M3)": nvfp4(w)}
    for name, q in rows.items():
        e = rms_error(w, q)
        print(f"{name:26} rms error {e:.5f}  ({100 * e / std:4.1f}% of std)")
