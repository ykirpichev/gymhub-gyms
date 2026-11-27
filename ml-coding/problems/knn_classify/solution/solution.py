import numpy as np


def solve(X_train, y_train, X_test, k):
    A = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.int64)
    B = np.asarray(X_test, dtype=np.float64)
    if not 1 <= k <= len(A):
        raise ValueError("k must be between 1 and the number of training points")
    # Squared distances, M x N, without an M x N x D intermediate.
    d2 = (B ** 2).sum(axis=1)[:, None] + (A ** 2).sum(axis=1)[None, :] - 2.0 * B @ A.T
    # A stable sort keeps the lower training index first among equal distances.
    nearest = np.argsort(d2, axis=1, kind="stable")[:, :k]
    votes = y[nearest]
    n_labels = int(y.max()) + 1
    # argmax over the counts returns the smallest label among those with the most votes.
    return np.array([np.bincount(row, minlength=n_labels).argmax() for row in votes], dtype=np.int64)
