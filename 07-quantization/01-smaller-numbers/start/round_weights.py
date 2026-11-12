"""Round real Qwen3-0.6B weights to 8 and 4 bits and measure the error."""
import numpy as np
from gguf import GGUFReader

from servinglab.quality import rms_error, round_to_bits

r = GGUFReader("models/Qwen3-0.6B-F16.gguf")
t = next(t for t in r.tensors if t.name == "blk.10.ffn_down.weight")
# Qwen3 is trained in bfloat16, and the file keeps it: the top 16 bits of a
# float32. Shift them back into place to read real numbers.
raw = np.ascontiguousarray(t.data).view(np.uint16).astype(np.uint32) << 16
w = raw.view(np.float32).reshape(1024, 3072)
print(t.name, w.shape, "std", round(float(w.std()), 4),
      "max", round(float(np.abs(w).max()), 3))
for bits, group in [(8, None), (8, 32), (4, None), (4, 32)]:
    err = rms_error(w, round_to_bits(w, bits, group))
    print(f"{bits} bits, {'one scale' if not group else f'groups of {group}'}:"
          f" rms error {err:.5f} ({100 * err / w.std():.1f}% of the std)")
