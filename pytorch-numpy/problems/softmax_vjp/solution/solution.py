import numpy as np


def solve(x, g):
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    s = e / e.sum(axis=1, keepdims=True)
    dx = s * (g - (g * s).sum(axis=1, keepdims=True))
    return (s, dx)
