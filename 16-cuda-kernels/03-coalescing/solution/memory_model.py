"""How a warp touches memory, counted exactly, with no GPU.

Global memory is served in 32-byte sectors; shared memory has 32 banks of
4-byte words. Both facts are from NVIDIA's CUDA C++ Best Practices Guide.
"""
import numpy as np

WARP, SECTOR, BANKS, WORD = 32, 32, 32, 4


def sectors_touched(byte_addresses):
    """32-byte sectors one warp's load or store needs."""
    return len({a // SECTOR for a in byte_addresses})


def stride_pattern(stride_elems, elem_bytes=4, start=0):
    """Lane i reads element start + i * stride."""
    return [(start + lane * stride_elems) * elem_bytes for lane in range(WARP)]


def load_efficiency(byte_addresses, elem_bytes=4):
    """Useful bytes over bytes moved."""
    return WARP * elem_bytes / (sectors_touched(byte_addresses) * SECTOR)


def bank_conflict_degree(word_indices):
    """How many separate passes shared memory needs: the most distinct words
    that fall in one bank. Lanes reading the same word share one broadcast."""
    per_bank = {}
    for w in set(word_indices):
        per_bank.setdefault(w % BANKS, set()).add(w)
    return max(len(v) for v in per_bank.values())


def tile_column_read(pad):
    """Tile of floats, shape [32][32 + pad]; lane x reads tile[x][0]."""
    return [x * (32 + pad) for x in range(WARP)]


def emulate_tiled_transpose(a, tile=32):
    """Run the CUDA transpose's index arithmetic for every block and thread,
    in the kernel's two phases (write the tile, barrier, read the tile)."""
    n = a.shape[0]
    out = np.zeros_like(a)
    for by in range(0, n, tile):
        for bx in range(0, n, tile):
            sh = np.zeros((tile, tile), dtype=a.dtype)
            for ty in range(tile):                     # phase 1
                for tx in range(tile):
                    x, y = bx + tx, by + ty
                    if x < n and y < n:
                        sh[ty, tx] = a[y, x]
            for ty in range(tile):                     # phase 2, after the barrier
                for tx in range(tile):
                    ox, oy = by + tx, bx + ty
                    if ox < n and oy < n:
                        out[oy, ox] = sh[tx, ty]
    return out


if __name__ == "__main__":
    print("stride  sectors  efficiency")
    for s in (1, 2, 4, 8, 32):
        addrs = stride_pattern(s)
        print(f"{s:>6}  {sectors_touched(addrs):>7}  {load_efficiency(addrs):>9.1%}")
    print("misaligned by one float:",
          sectors_touched(stride_pattern(1, start=1)), "sectors")
    n = 8192
    print(f"naive transpose write, stride {n}:",
          sectors_touched(stride_pattern(n)), "sectors for 128 useful bytes")
    for pad in (0, 1):
        print(f"tile[32][{32 + pad}] column read: "
              f"{bank_conflict_degree(tile_column_read(pad))}-way")
    a = np.arange(70 * 70).reshape(70, 70)
    print("emulated transpose correct:", np.array_equal(emulate_tiled_transpose(a), a.T))
