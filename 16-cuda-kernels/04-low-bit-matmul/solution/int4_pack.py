"""4-bit weights with a scale per group, packed eight to a 32-bit word.

Same layout and bit tricks as w4_matvec.cu, in NumPy, so you can check the
kernel's arithmetic before you rent a GPU.
  q in 0..15 stands for (q - 8) * scale.  Nibble i of a word is bits 4i..4i+3.
"""
import numpy as np

from servinglab.data import hardware

GROUP = 128


def quantize(w, group=GROUP):
    """w: (N, K) float -> q (N, K) uint8 in 0..15, scales (N, K // group)."""
    n, k = w.shape
    g = w.reshape(n, k // group, group)
    scales = np.abs(g).max(axis=2) / 7.0              # symmetric, -7..7
    q = np.clip(np.round(g / scales[..., None]), -8, 7) + 8
    return q.reshape(n, k).astype(np.uint8), scales.astype(np.float32)


def pack(q):
    """(N, K) nibbles -> (N, K // 8) uint32, first value in the low bits."""
    n, k = q.shape
    q = q.reshape(n, k // 8, 8).astype(np.uint32)
    word = np.zeros((n, k // 8), dtype=np.uint32)
    for i in range(8):
        word |= q[:, :, i] << np.uint32(4 * i)
    return word


def unpack(word):
    return np.stack([(word >> np.uint32(4 * i)) & np.uint32(0xF)
                     for i in range(8)], axis=2).reshape(word.shape[0], -1)


def matvec_warp_emulation(word, scales, x, group=GROUP, lanes=32):
    """What the kernel does: one 'warp' per row; lane l takes words
    l, l + 32, ...; each unpacks, dequantizes and multiplies in 'registers';
    the partial sums are added at the end (the shuffle reduction)."""
    n, words = word.shape
    y = np.zeros(n)
    for row in range(n):
        partial = np.zeros(lanes)
        for lane in range(lanes):
            for j in range(lane, words, lanes):
                packed = int(word[row, j])
                k0 = j * 8
                scale = scales[row, k0 // group]
                for i in range(8):
                    q = (packed >> (4 * i)) & 0xF
                    partial[lane] += (q - 8) * scale * x[k0 + i]
        y[row] = partial.sum()
    return y


def bits_per_weight(group=GROUP, scale_bits=16):
    return 4 + scale_bits / group


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, k = 8, 1024
    w = rng.normal(size=(n, k)).astype(np.float32)
    x = rng.normal(size=k).astype(np.float32)
    q, s = quantize(w)
    word = pack(q)
    assert np.array_equal(unpack(word), q)
    deq = (q.astype(np.float32) - 8).reshape(n, k // GROUP, GROUP) * s[..., None]
    deq = deq.reshape(n, k)
    y_ref, y_q = w @ x, matvec_warp_emulation(word, s, x)
    print("emulated kernel equals dequantize-then-matmul:",
          np.allclose(y_q, deq @ x, atol=1e-4))
    print(f"mean |weight error| / mean |weight|: "
          f"{np.abs(deq - w).mean() / np.abs(w).mean():.3f}")
    print(f"output error vs fp32 weights: {np.abs(y_q - y_ref).max():.3f} "
          f"(outputs are about {np.abs(y_ref).mean():.1f})")
    print("bits per weight with group 128 and fp16 scales:", bits_per_weight())
    rows, cols = 14336, 4096          # one Llama 3.1 8B MLP matrix
    fp16 = rows * cols * 2
    int4 = rows * cols * bits_per_weight() / 8
    print(f"fp16 {fp16 / 2**20:.1f} MiB, int4 {int4 / 2**20:.1f} MiB, "
          f"ratio {fp16 / int4:.3f}")
    for name, chip in (("T4", "T4"), ("H100 SXM", "H100-SXM")):
        gbs = hardware(chip)["bandwidth_gbs"]
        print(f"{name}: read it once in {fp16 / (gbs * 1e9) * 1e6:.0f} us (fp16), "
              f"{int4 / (gbs * 1e9) * 1e6:.0f} us (int4)")
