"""Arithmetic intensity (FLOPs per byte) of three operations, and what a chip can do with each.

    uv run python intensity.py
"""

from servinglab.data import hardware


def vector_add(n, width=4):
    """c = a + b over n numbers of `width` bytes: n adds; read two arrays, write one."""
    raise NotImplementedError(
        "TODO 02.3: return the adds per byte moved: n adds over two arrays read and one written, see the lesson's 'Arithmetic intensity' section")


def matmul(n, width=2):
    """An n by n matrix times an n by n matrix: n*n outputs of n multiply-adds (2 FLOPs each);
    read two matrices, write one."""
    raise NotImplementedError(
        "TODO 02.3: return the FLOPs per byte moved: 2 * n**3 over the bytes of two matrices read and one written, see the lesson's 'Arithmetic intensity' section")


def matvec(n, width=2):
    """An n by n matrix times one vector (a decode step): 2 FLOPs per weight, every weight read once."""
    raise NotImplementedError(
        "TODO 02.3: return the FLOPs per byte moved: 2 FLOPs per weight over the matrix read once plus the small vectors, see the lesson's 'Arithmetic intensity' section")


def attainable(intensity, peak_tflops, bandwidth_gbs):
    """TFLOPS the chip can reach at this intensity: the memory slope until the flat compute roof."""
    raise NotImplementedError(
        "TODO 02.3: return the smaller of the compute roof and intensity times bandwidth, in TFLOPS, see the lesson's 'The roofline' section")


if __name__ == "__main__":
    h100 = hardware("H100-SXM")
    peak, bw = h100["fp16_tflops"], h100["bandwidth_gbs"]
    print(f"H100: peak {peak} TFLOPS, {bw} GB/s, ridge {peak * 1000 / bw:.0f} FLOPs per byte")
    for name, i in [("vector add (float32)", vector_add(2**20)), ("matrix x vector, 4096 (float16)", matvec(4096)),
                    ("matrix x matrix, 4096 (float16)", matmul(4096))]:
        got = attainable(i, peak, bw)
        print(f"{name:<34} {i:>7.2f} FLOPs/byte -> {got:>7.1f} TFLOPS ({got / peak:.1%} of peak)")
