import numpy as np


def solve(X, y, epochs):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.all((y == 1) | (y == -1)):
        raise ValueError("labels must be -1 or +1")
    w = np.zeros(X.shape[1])
    b = 0.0
    updates = 0
    for _ in range(epochs):
        for xi, yi in zip(X, y):
            if yi * (xi @ w + b) <= 0:  # a zero margin is a mistake
                w += yi * xi
                b += yi
                updates += 1
    return w, b, updates
