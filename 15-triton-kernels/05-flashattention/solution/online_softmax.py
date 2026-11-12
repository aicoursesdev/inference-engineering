"""Softmax in one pass: keep a running max m and a running sum l.

When a bigger number arrives, the old sum was measured against a smaller
max, so multiply it by exp(old_max - new_max) to put it on the new scale.
"""
import numpy as np


def softmax_three_pass(x):
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()


def softmax_online(x):
    m, l = -np.inf, 0.0
    for v in x:
        m_new = max(m, v)
        l = l * np.exp(m - m_new) + np.exp(v - m_new)
        m = m_new
    return np.exp(x - m) / l, m, l


if __name__ == "__main__":
    x = np.array([2.0, 1.0, 4.0, 3.0])
    m, l = -np.inf, 0.0
    for v in x:
        m_new = max(m, v)
        l = l * np.exp(m - m_new) + np.exp(v - m_new)
        m = m_new
        print(f"saw {v}: max {m}, sum {l:.4f}")
    print("online :", softmax_online(x)[0].round(4))
    print("3-pass :", softmax_three_pass(x).round(4))
