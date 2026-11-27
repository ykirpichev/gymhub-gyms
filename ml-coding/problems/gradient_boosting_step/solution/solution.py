import numpy as np


def solve(X, y, F, learning_rate):
    X = np.asarray(X, dtype=np.float64)
    residual = np.asarray(y, dtype=np.float64) - np.asarray(F, dtype=np.float64)

    candidates = []  # (sse, feature, threshold, left_mean, right_mean), in feature-then-threshold order
    for j in range(X.shape[1]):
        values = np.unique(X[:, j])  # sorted distinct values
        for t in (values[:-1] + values[1:]) / 2:
            left = X[:, j] <= t
            rl, rr = residual[left], residual[~left]
            sse = ((rl - rl.mean()) ** 2).sum() + ((rr - rr.mean()) ** 2).sum()
            candidates.append((sse, j, float(t), rl.mean(), rr.mean()))

    if not candidates:
        mean = float(residual.mean())
        feature, threshold, left_value, right_value = -1, 0.0, mean, mean
        update = np.full(len(residual), mean)
    else:
        best = min(c[0] for c in candidates)
        _, feature, threshold, left_value, right_value = next(c for c in candidates if c[0] <= best + 1e-9)
        update = np.where(X[:, feature] <= threshold, left_value, right_value)

    F_new = np.asarray(F, dtype=np.float64) + learning_rate * update
    return feature, threshold, float(left_value), float(right_value), F_new
