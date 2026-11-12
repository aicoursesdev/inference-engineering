"""Count the elements an unfused and a fused kernel move through memory.

Every step of an unfused softmax or RMSNorm is its own kernel: it reads its
inputs from the warehouse (HBM) and writes its result back. A fused kernel
reads the input once and writes the output once. This file only counts.
"""


def softmax_unfused(m, n):
    """Steps of softmax over each of m rows of n numbers, as (name, reads, writes)."""
    return [
        ("row max",       m * n,     m),
        ("x - max",       m * n + m, m * n),
        ("exp",           m * n,     m * n),
        ("row sum",       m * n,     m),
        ("divide by sum", m * n + m, m * n),
    ]


def rmsnorm_unfused(m, n):
    """x * rsqrt(mean(x*x) + eps) * weight, written as separate PyTorch ops."""
    return [
        ("x * x",            m * n,     m * n),
        ("mean",             m * n,     m),
        ("+ eps",            m,         m),
        ("rsqrt",            m,         m),
        ("x * rsqrt",        m * n + m, m * n),
        ("* weight",         m * n + n, m * n),
    ]


def fused(m, n, extra_reads=0):
    """One kernel: read x once (plus a weight vector), write y once."""
    return [("fused", m * n + extra_reads, m * n)]


def total(steps):
    return sum(r + w for _, r, w in steps)


def report(title, steps, fused_steps, bytes_per_element=2):
    unfused, one = total(steps), total(fused_steps)
    print(title)
    for name, r, w in steps:
        print(f"  {name:<14} reads {r:>10,}  writes {w:>10,}")
    print(f"  unfused total {unfused:,} elements = "
          f"{unfused * bytes_per_element / 2**20:.1f} MiB at "
          f"{bytes_per_element} bytes each")
    print(f"  fused total   {one:,} elements = "
          f"{one * bytes_per_element / 2**20:.1f} MiB")
    print(f"  traffic ratio {unfused / one:.2f}x")


if __name__ == "__main__":
    # 4,096 rows (tokens) of 1,024 numbers: the hidden size of Qwen3-0.6B
    m, n = 4096, 1024
    report("softmax", softmax_unfused(m, n), fused(m, n))
    report("RMSNorm", rmsnorm_unfused(m, n), fused(m, n, extra_reads=n))
