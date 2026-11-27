import numpy as np


def solve(X, centroids, max_iter):
    points = np.asarray(X, dtype=np.float64)
    centers = np.asarray(centroids, dtype=np.float64).copy()
    if points.ndim != 2 or centers.ndim != 2 or points.shape[1] != centers.shape[1]:
        raise ValueError("X is N x D and centroids is K x D")
    if points.shape[0] == 0 or centers.shape[0] == 0:
        raise ValueError("need at least one point and one centroid")
    if not _is_int(max_iter) or max_iter < 0:
        raise ValueError("max_iter must be an integer >= 0")

    for _ in range(int(max_iter)):
        labels = _assign(points, centers)
        updated = centers.copy()
        for cluster in range(len(centers)):
            members = points[labels == cluster]
            if len(members):
                updated[cluster] = members.mean(axis=0)
        if np.max(np.abs(updated - centers)) <= 1e-8:
            centers = updated
            break
        centers = updated
    return centers, _assign(points, centers)


def _assign(points, centers):
    # argmin keeps the lowest centroid index when squared distances tie.
    distance = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distance, axis=1)


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)
