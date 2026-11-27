"""Grader for kmeans. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def _is_int(value):
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _assign(points, centers):
    distance = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(distance, axis=1)


def reference(X, centroids, max_iter):
    points = np.asarray(X, dtype=np.float64)
    centers = np.asarray(centroids, dtype=np.float64).copy()
    if points.ndim != 2 or centers.ndim != 2 or points.shape[1] != centers.shape[1]:
        return "ValueError"
    if points.shape[0] == 0 or centers.shape[0] == 0:
        return "ValueError"
    if not _is_int(max_iter) or max_iter < 0:
        return "ValueError"
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


def call(module, args):
    """A negative max_iter or a shape mismatch must raise ValueError."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if want == "ValueError" or got == "ValueError":
        if got != want:
            raise AssertionError(
                "expected ValueError: max_iter >= 0, and X and centroids are matrices with the same number of features"
                if want == "ValueError" else "raised ValueError, but the inputs are valid")
        return
    if not isinstance(got, tuple) or len(got) != 2:
        raise AssertionError(f"return (centroids, labels), got {type(got).__name__}")
    try:
        default_compare(got[0], want[0], 1e-8, 1e-6, "centroids")
    except AssertionError as exc:
        raise AssertionError(
            "centroids differ. An empty cluster keeps its previous centroid, "
            "and an equal distance goes to the lowest centroid index.\n" + str(exc)) from None
    try:
        default_compare(got[1], want[1], 0, 0, "labels")
    except AssertionError as exc:
        raise AssertionError(
            "labels differ. On an equal squared distance, choose the lowest centroid index.\n" + str(exc)) from None


def _random(seed, n, d, k, max_iter, name):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(k, d)) * 3.0
    drawn = rng.integers(0, k, size=n)
    points = centers[drawn] + rng.normal(size=(n, d)) * 0.4
    initial = points[rng.choice(n, size=k, replace=False)]
    return {"_name": name, "X": points.tolist(), "centroids": initial.tolist(), "max_iter": max_iter}


def functional_cases():
    early = {"X": [[0.0], [1.0], [2.0], [3.0], [10.0]], "centroids": [[0.0], [1.0]]}
    return [
        {"_name": "example", "X": [[0.0, 0.0], [1.0, 0.0], [8.0, 0.0], [9.0, 0.0]],
         "centroids": [[0.0, 0.0], [9.0, 0.0]], "max_iter": 10},
        {"_name": "equidistant: lowest centroid index",
         "X": [[0.0, 0.0], [2.0, 0.0]], "centroids": [[1.0, 0.0], [1.0, 0.0]], "max_iter": 5},
        {"_name": "empty cluster keeps its centroid",
         "X": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
         "centroids": [[0.0, 0.0], [10.0, 10.0], [0.1, 0.1]], "max_iter": 10},
        {"_name": "max_iter 0 returns the initial centroids",
         "X": [[0.0, 0.0], [1.0, 0.0], [8.0, 0.0], [9.0, 0.0]],
         "centroids": [[0.0, 0.0], [9.0, 0.0]], "max_iter": 0},
        {"_name": "max_iter 1 stops early", **early, "max_iter": 1},
        {"_name": "same data run to convergence", **early, "max_iter": 20},
        {"_name": "one cluster", "X": [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
         "centroids": [[0.0, 0.0]], "max_iter": 5},
        {"_name": "duplicate points, unused centroid stays",
         "X": [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]], "centroids": [[1.0, 1.0], [5.0, 5.0]], "max_iter": 4},
        _random(1, 40, 3, 3, 10, "random 40x3, k=3"),
        _random(2, 120, 6, 5, 25, "random 120x6, k=5"),
        {"_name": "negative max_iter", "X": [[0.0], [1.0]], "centroids": [[0.0]], "max_iter": -1},
        {"_name": "feature dimension mismatch", "X": [[0.0, 1.0], [1.0, 0.0]],
         "centroids": [[0.0], [1.0]], "max_iter": 3},
    ]


def build() -> Problem:
    return Problem(
        slug="kmeans",
        title="k-means clustering",
        difficulty="medium",
        summary="Lloyd's algorithm from given centroids, keeping empty clusters and breaking distance ties by index.",
        params=["X", "centroids", "max_iter"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "clustering", "unsupervised"],
    )
