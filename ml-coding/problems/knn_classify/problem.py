"""Grader for knn_classify. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem

TIME_LIMIT_MS = {"python": 2000}


def reference(X_train, y_train, X_test, k):
    a = np.asarray(X_train, dtype=np.float64)
    labels = np.asarray(y_train, dtype=np.int64)
    b = np.asarray(X_test, dtype=np.float64)
    if k < 1 or k > a.shape[0]:
        return "ValueError"
    out = np.empty(b.shape[0], dtype=np.int64)
    for start in range(0, b.shape[0], 256):
        chunk = b[start:start + 256]
        d2 = ((chunk[:, None, :] - a[None, :, :]) ** 2).sum(axis=2)
        order = np.argsort(d2, axis=1, kind="stable")[:, :k]
        for row, idx in enumerate(order):
            counts = np.bincount(labels[idx], minlength=labels.max() + 1)
            out[start + row] = int(np.flatnonzero(counts == counts.max())[0])
    return out


def call(module, args):
    """k outside [1, N] must raise ValueError; anything else is returned for comparison."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(want, str) or isinstance(got, str):
        if not isinstance(got, str) or got != want:
            raise AssertionError("expected ValueError: k must be between 1 and the number of training points"
                                 if isinstance(want, str) else "raised ValueError, but k is valid")
        return
    default_compare(got, want, 0, 0, "labels")


def _random(seed, n, m, d, classes, k, name):
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((classes, d)) * 2.0
    y = rng.integers(0, classes, n)
    x = centers[y] + rng.standard_normal((n, d))
    test = centers[rng.integers(0, classes, m)] + 1.5 * rng.standard_normal((m, d))
    return {"_name": name, "X_train": x.tolist(), "y_train": y.tolist(), "X_test": test.tolist(), "k": k}


def functional_cases():
    ring = [[3.0, 4.0], [-3.0, 4.0], [5.0, 0.0], [0.0, -5.0], [-4.0, -3.0], [4.0, -3.0], [-5.0, 0.0], [0.0, 5.0]]
    return [
        {"_name": "example", "X_train": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [5.0, 5.0], [6.0, 5.0]],
         "y_train": [0, 0, 1, 1, 1], "X_test": [[0.2, 0.1], [5.5, 4.0]], "k": 3},
        {"_name": "k = 1 returns the nearest label", "X_train": [[0.0], [10.0], [20.0]], "y_train": [4, 7, 2],
         "X_test": [[1.0], [9.0], [16.0], [100.0]], "k": 1},
        {"_name": "equal distances: lower training index wins", "X_train": [[0.0, 0.0], [2.0, 0.0], [1.0, 1.0]],
         "y_train": [5, 3, 9], "X_test": [[1.0, 0.0]], "k": 1},
        {"_name": "8 points at distance 5, k = 3", "X_train": ring, "y_train": [2, 2, 1, 0, 0, 1, 1, 3],
         "X_test": [[0.0, 0.0]], "k": 3},
        {"_name": "vote tie: smallest label, not the nearest", "X_train": [[0.0], [1.0], [3.0], [4.0]],
         "y_train": [3, 1, 3, 1], "X_test": [[0.0], [4.0]], "k": 2},
        {"_name": "k = N: majority of the whole set", "X_train": [[0.0], [1.0], [2.0], [3.0], [4.0]],
         "y_train": [2, 0, 2, 0, 1], "X_test": [[-10.0], [10.0]], "k": 5},
        {"_name": "test point equals a training point", "X_train": [[1.0, 2.0, 3.0], [1.0, 2.0, 3.5], [9.0, 9.0, 9.0]],
         "y_train": [0, 1, 1], "X_test": [[1.0, 2.0, 3.0]], "k": 1},
        _random(1, 60, 15, 4, 3, 5, "random 60 train, 15 test, k=5"),
        _random(2, 2000, 400, 10, 5, 7, "random 2000 train, 400 test, k=7"),
        {"_name": "k = 0", "X_train": [[0.0], [1.0]], "y_train": [0, 1], "X_test": [[0.5]], "k": 0},
        {"_name": "k > N", "X_train": [[0.0], [1.0]], "y_train": [0, 1], "X_test": [[0.5]], "k": 3},
    ]


def performance_case(seed: int = 0):
    return _random(seed, 5000, 1000, 16, 10, 15, "perf")


def build() -> Problem:
    return Problem(
        slug="knn_classify",
        title="k-nearest neighbours classifier",
        difficulty="medium",
        summary="Vectorised pairwise distances, top-k neighbours and a majority vote with deterministic tie-breaking.",
        params=["X_train", "y_train", "X_test", "k"],
        reference=reference,
        functional_cases=functional_cases,
        performance_case=performance_case,
        call=call,
        compare=compare,
        time_limit_ms=TIME_LIMIT_MS,
        tags=["numpy", "classification", "distance", "vectorization"],
    )
