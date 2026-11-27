import numpy as np


def solve(x, W, b, A, B, alpha):
    x, W, b, A, B = (np.asarray(v, dtype=np.float64) for v in (x, W, b, A, B))
    if x.ndim != 2 or W.ndim != 2 or A.ndim != 2 or B.ndim != 2 or b.ndim != 1:
        raise ValueError("x, W, A, B must be 2-D and b 1-D")
    out_f, in_f = W.shape
    r = A.shape[0]
    if r < 1 or x.shape[1] != in_f or b.shape[0] != out_f or A.shape[1] != in_f or B.shape != (out_f, r):
        raise ValueError("inconsistent shapes")
    scale = alpha / r
    y = x @ W.T + b + scale * ((x @ A.T) @ B.T)
    W_merged = W + scale * (B @ A)
    return y, W_merged, r * (in_f + out_f), out_f * in_f
