"""Read a model's weights from a .safetensors file, with NumPy only.

A safetensors file is: 8 bytes (the length N of the header), N bytes of JSON
(each tensor's dtype, shape and byte range), then the raw numbers. That is
simple enough to read by hand, and doing so avoids a library that wants
PyTorch to open bfloat16 files, which NumPy has no type for.
"""

import json
import struct

import numpy as np

_PLAIN = {"F32": np.float32, "F16": np.float16}


def bf16_to_f32(raw):
    """bfloat16 is the top 16 bits of a float32: shift them back into place."""
    return (raw.astype(np.uint32) << 16).view(np.float32)


def read_safetensors(path):
    """Return {tensor name: float32 array}. Handles F32, F16 and BF16."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        header.pop("__metadata__", None)
        out = {}
        for name, t in header.items():
            start, end = t["data_offsets"]
            f.seek(base + start)
            raw = f.read(end - start)
            if t["dtype"] == "BF16":
                x = bf16_to_f32(np.frombuffer(raw, dtype=np.uint16))
            elif t["dtype"] in _PLAIN:
                x = np.frombuffer(raw, dtype=_PLAIN[t["dtype"]])
            else:
                raise ValueError(f"{name}: dtype {t['dtype']} not supported")
            out[name] = x.astype(np.float32).reshape(t["shape"])
    return out
