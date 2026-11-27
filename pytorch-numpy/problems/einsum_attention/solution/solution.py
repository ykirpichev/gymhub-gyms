import numpy as np


def solve(q, k, v):
    q = np.asarray(q, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    s = np.einsum("bihd,bjhd->bhij", q, k) / np.sqrt(q.shape[-1])
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(-1, keepdims=True)
    return np.einsum("bhij,bjhd->bihd", p, v)
