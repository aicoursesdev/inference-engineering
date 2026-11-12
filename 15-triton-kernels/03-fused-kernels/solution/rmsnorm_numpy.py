"""The same RMSNorm in NumPy: the numbers the kernel must match."""
import numpy as np


def rmsnorm(x, weight, eps=1e-6):
    ms = (x * x).mean(axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * weight


if __name__ == "__main__":
    x = np.array([[1.0, -2.0, 3.0, -4.0]])
    w = np.ones(4)
    print("mean of squares:", (x * x).mean())          # 7.5
    print("rms:", np.sqrt((x * x).mean() + 1e-6))
    print("output:", rmsnorm(x, w).round(4))
