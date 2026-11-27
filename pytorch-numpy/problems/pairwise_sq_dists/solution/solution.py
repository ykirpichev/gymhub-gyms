import numpy as np


def solve(X, Y):
    x = np.asarray(X, dtype=np.float64)
    y = np.asarray(Y, dtype=np.float64)
    d = (x * x).sum(1)[:, None] - 2 * x @ y.T + (y * y).sum(1)[None, :]
    return np.maximum(d, 0.0)
