"""Correctness first, then the benchmark: the 4-bit kernel against the dense
fp16 matvec PyTorch would run. Needs a CUDA GPU.   python bench_w4.py"""
import numpy as np
import torch

import int4_pack
from harness import markdown_table, summarize, timeit
from w4_ext import load

N, K = 14336, 4096                      # one Llama 3.1 8B MLP matrix


def main():
    ext = load()
    rng = np.random.default_rng(0)
    w = rng.normal(size=(N, K)).astype(np.float32) * 0.02
    q, scales = int4_pack.quantize(w)
    word = torch.from_numpy(int4_pack.pack(q).view(np.int32)).cuda()
    sc = torch.from_numpy(scales).cuda()
    x = torch.randn(K, device="cuda")

    deq = ((q.astype(np.float32) - 8).reshape(N, K // 128, 128) * scales[..., None]).reshape(N, K)
    ref = torch.from_numpy(deq).cuda() @ x
    torch.testing.assert_close(ext.w4_matvec(word, sc, x), ref, atol=1e-2, rtol=1e-3)

    w16 = torch.from_numpy(w).cuda().half()
    x16 = x.half()
    rows = []
    for name, fn in (("dense fp16 matvec (torch)", lambda: w16 @ x16),
                     ("4-bit matvec (ours)", lambda: ext.w4_matvec(word, sc, x))):
        rows.append({"name": name, **summarize(timeit(fn))})
    print(markdown_table(rows))


if __name__ == "__main__":
    main()
