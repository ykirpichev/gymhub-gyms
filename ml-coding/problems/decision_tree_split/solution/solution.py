import numpy as np

TIE_TOL = 1e-12


def solve(X, y):
    features = np.asarray(X, dtype=np.float64)
    labels = np.asarray(y)
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("X must be N x D and y must have N integer labels")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("X must have at least one row and one column")
    labels = labels.astype(np.int64, copy=False)
    parent = _gini(labels)
    best = None  # (impurity, feature, threshold), scanned in tie-break order
    for feature in range(features.shape[1]):
        values = np.unique(features[:, feature])
        for threshold in (values[:-1] + values[1:]) / 2.0:
            left = features[:, feature] <= threshold
            impurity = (left.sum() * _gini(labels[left]) + (~left).sum() * _gini(labels[~left])) / len(labels)
            candidate = (float(impurity), feature, float(threshold))
            if best is None or candidate[0] < best[0] - TIE_TOL:
                best = candidate
    if best is None or best[0] >= parent - TIE_TOL:
        return None
    return best[1], best[2], best[0]


def _gini(labels):
    _, counts = np.unique(labels, return_counts=True)
    proportions = counts / counts.sum()
    return float(1.0 - np.sum(proportions ** 2))
