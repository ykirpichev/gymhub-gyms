import numpy as np


def solve(W):
    w = np.asarray(W, dtype=np.float64)
    peak = np.abs(w).max(axis=1)
    scales = np.where(peak > 0, peak / 127.0, 1.0)
    q = np.clip(np.round(w / scales[:, None]), -127, 127).astype(np.int64)
    return (q.tolist(), scales.tolist())
