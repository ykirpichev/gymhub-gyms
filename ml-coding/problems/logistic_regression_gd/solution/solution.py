import numpy as np


def sigmoid(z):
    e = np.exp(-np.abs(z))  # in (0, 1]: never overflows
    return np.where(z >= 0, 1.0 / (1.0 + e), e / (1.0 + e))


def solve(X, y, lr, steps, l2=0.0, threshold=0.5):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    if y.shape != (n,) or not np.all((y == 0) | (y == 1)):
        raise ValueError("y must hold N labels, each 0 or 1")
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        err = sigmoid(X @ w + b) - y
        w = w - lr * (X.T @ err / n + l2 * w)
        b = b - lr * err.mean()
    z = X @ w + b
    # -[y log p + (1 - y) log(1 - p)] = log(1 + e^z) - y z, with log(1 + e^z) = logaddexp(0, z)
    loss = float(np.mean(np.logaddexp(0.0, z) - y * z) + 0.5 * l2 * (w @ w))
    preds = (sigmoid(z) >= threshold).astype(np.int64)
    return w, float(b), loss, preds
