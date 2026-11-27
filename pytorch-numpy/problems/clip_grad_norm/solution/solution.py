import numpy as np


def solve(grads, max_norm):
    arrays = [np.asarray(g, dtype=np.float64) for g in grads]
    norm = float(np.sqrt(sum(float((a * a).sum()) for a in arrays)))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)
        arrays = [a * scale for a in arrays]
    return (arrays, norm)
