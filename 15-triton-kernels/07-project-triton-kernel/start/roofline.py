"""Roofline arithmetic and the chart: achieved against peak.

attainable(ai) = min(peak_flops, ai * peak_bandwidth), with ai in FLOP/byte.
"""
import json
import sys

from servinglab.data import hardware

# Datasheet numbers live in data/hardware.json, with a source URL and date for
# each: T4 320 GB/s and 65 TFLOPS FP16 (NVIDIA T4 page); H100 SXM 3,350 GB/s
# and 989.5 dense TFLOPS (the datasheet's 1,979 is with sparsity, so halved,
# as lesson 02.6 teaches).
HARDWARE = {
    label: {"gbs": hardware(name)["bandwidth_gbs"],
            "tflops": hardware(name)["fp16_tflops"]}
    for label, name in (("T4", "T4"), ("H100 SXM", "H100-SXM"))
}


def ridge(gbs, tflops):
    """FLOP per byte where the sloped roof meets the flat one."""
    return tflops * 1e12 / (gbs * 1e9)


def attainable_tflops(ai, gbs, tflops):
    return min(tflops, ai * gbs * 1e9 / 1e12)


def rmsnorm_traffic(m, n, itemsize):
    """Bytes a fused RMSNorm must move: x in, y out, weight in."""
    return (2 * m * n + n) * itemsize


def rmsnorm_flops(m, n):
    """About 4 per element: square, add into the sum, divide, scale."""
    return 4 * m * n


def achieved(m, n, itemsize, ms):
    """(GB/s, TFLOPS) a kernel reached when it took ms milliseconds."""
    seconds = ms * 1e-3
    return (rmsnorm_traffic(m, n, itemsize) / seconds / 1e9,
            rmsnorm_flops(m, n) / seconds / 1e12)


def plot(results, hardware, path="roofline.png"):
    """results: [{"label", "ai", "tflops"}]. Needs matplotlib."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    gbs, tflops = hardware["gbs"], hardware["tflops"]
    ai = np.logspace(-1, 3, 200)
    roof = [attainable_tflops(a, gbs, tflops) for a in ai]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(ai, roof, color="black", label="roofline")
    for r in results:
        ax.scatter([r["ai"]], [r["tflops"]], label=r["label"])
    ax.set_xlabel("arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("TFLOPS")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    return path


if __name__ == "__main__":
    # python roofline.py results.json "H100 SXM"
    rows = json.load(open(sys.argv[1]))
    print(plot(rows, HARDWARE[sys.argv[2]]))
