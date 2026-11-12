"""Arithmetic intensity (FLOPs per byte) of three operations, and what a chip can do with each.

    uv run python intensity.py
"""

from servinglab.data import hardware


def vector_add(n, width=4):
    """c = a + b over n numbers of `width` bytes: n adds; read two arrays, write one."""
    return n / (3 * n * width)


def matmul(n, width=2):
    """An n by n matrix times an n by n matrix: n*n outputs of n multiply-adds (2 FLOPs each);
    read two matrices, write one."""
    return 2 * n**3 / (3 * n * n * width)


def matvec(n, width=2):
    """An n by n matrix times one vector (a decode step): 2 FLOPs per weight, every weight read once."""
    return 2 * n * n / (n * n * width + 2 * n * width)  # the vector in and out are tiny beside the matrix


def attainable(intensity, peak_tflops, bandwidth_gbs):
    """TFLOPS the chip can reach at this intensity: the memory slope until the flat compute roof."""
    return min(peak_tflops, intensity * bandwidth_gbs / 1000)


if __name__ == "__main__":
    h100 = hardware("H100-SXM")
    peak, bw = h100["fp16_tflops"], h100["bandwidth_gbs"]
    print(f"H100: peak {peak} TFLOPS, {bw} GB/s, ridge {peak * 1000 / bw:.0f} FLOPs per byte")
    for name, i in [("vector add (float32)", vector_add(2**20)), ("matrix x vector, 4096 (float16)", matvec(4096)),
                    ("matrix x matrix, 4096 (float16)", matmul(4096))]:
        got = attainable(i, peak, bw)
        print(f"{name:<34} {i:>7.2f} FLOPs/byte -> {got:>7.1f} TFLOPS ({got / peak:.1%} of peak)")
