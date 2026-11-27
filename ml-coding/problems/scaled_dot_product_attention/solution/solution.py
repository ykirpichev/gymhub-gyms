import numpy as np


def solve(q, k, v, mask=None):
    q, k, v = (np.asarray(a, dtype=np.float64) for a in (q, k, v))
    scores = q @ np.swapaxes(k, -1, -2) / np.sqrt(q.shape[-1])
    if mask is not None:
        scores = np.where(np.asarray(mask, dtype=bool), scores, -np.inf)
    row_max = scores.max(axis=-1, keepdims=True)
    row_max[~np.isfinite(row_max)] = 0.0  # fully masked rows: exp(-inf) = 0 everywhere
    e = np.exp(scores - row_max)
    total = e.sum(axis=-1, keepdims=True)
    weights = e / np.where(total == 0, 1.0, total)
    return weights @ v, weights
