import numpy as np


def solve(X, k):
    data = np.asarray(X, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("X must be N x D")
    n, d = data.shape
    if not _is_int(k) or k < 1 or k > d or k > n - 1:
        raise ValueError("k must be an integer in 1 .. min(D, N-1)")
    centered = data - data.mean(axis=0)
    if np.all(centered == 0):
        raise ValueError("every row is equal; the components are undefined")
    # Eigenvectors of the Gram matrix, largest eigenvalue first. Dividing by N or N-1
    # does not change the vectors or the explained-variance ratios.
    values, vectors = np.linalg.eigh(centered.T @ centered)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, None)
    components = _orient(vectors[:, order][:, :k].T)
    return components, values[:k] / values.sum()


def _orient(components):
    oriented = np.array(components, dtype=np.float64, copy=True)
    for row in oriented:
        index = int(np.argmax(np.abs(row)))
        if row[index] < 0:
            row *= -1
    return oriented


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
