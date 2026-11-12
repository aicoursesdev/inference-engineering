"""What a Triton launch does, in plain NumPy: no GPU, no Triton needed.

`launch` runs the same body once per program id, the way the interpreter
does, and `touched` records which elements each program read.
"""
import numpy as np


def add_program(pid, x, y, out, n, block_size, touched):
    offsets = pid * block_size + np.arange(block_size)   # tl.arange
    mask = offsets < n                                    # the guard
    touched[pid] = offsets[mask].tolist()
    out[offsets[mask]] = x[offsets[mask]] + y[offsets[mask]]


def launch(x, y, block_size):
    n = len(x)
    out = np.zeros_like(x)
    touched = {}
    for pid in range(-(-n // block_size)):                # cdiv(n, block)
        add_program(pid, x, y, out, n, block_size, touched)
    return out, touched


if __name__ == "__main__":
    x = np.arange(10, dtype=np.float32)
    out, touched = launch(x, x * 10, block_size=4)
    for pid, elems in touched.items():
        print(f"program {pid} handled elements {elems}")
    print("result:", out)
