"""Check, then time, the kernel against PyTorch. Needs a CUDA GPU.

    python bench.py          # prints a table and writes results.json
"""
import json

import torch
import triton

from rmsnorm_kernel import rmsnorm, rmsnorm_torch
from roofline import achieved, rmsnorm_flops, rmsnorm_traffic

HIDDEN = 4096            # Llama 3.1 8B's hidden size
EPS = 1e-5               # and its rms_norm_eps


def main(tokens=(1, 16, 256, 4096), dtype=torch.float16):
    torch.manual_seed(0)
    rows = []
    for m in tokens:
        x = torch.randn(m, HIDDEN, device="cuda", dtype=dtype)
        w = torch.randn(HIDDEN, device="cuda", dtype=dtype)
        # 1. Correct before fast.
        torch.testing.assert_close(rmsnorm(x, w, EPS), rmsnorm_torch(x, w, EPS),
                                   atol=1e-2, rtol=1e-2)
        # 2. Time both the same way. do_bench warms up and repeats (its warmup
        #    and rep are milliseconds, not counts); we keep the median.
        for name, fn in (("triton", rmsnorm), ("torch", rmsnorm_torch)):
            ms = triton.testing.do_bench(lambda: fn(x, w, EPS),
                                         return_mode="median")
            gbs, tflops = achieved(m, HIDDEN, x.element_size(), ms)
            ai = rmsnorm_flops(m, HIDDEN) / rmsnorm_traffic(m, HIDDEN, x.element_size())
            rows.append({"label": f"{name} m={m}", "ms": ms, "gbs": gbs,
                         "tflops": tflops, "ai": ai})
            print(f"{name:<7} m={m:<5} {ms * 1000:8.1f} us  {gbs:8.1f} GB/s")
    json.dump(rows, open("results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
