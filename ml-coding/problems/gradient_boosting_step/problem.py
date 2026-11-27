"""Grader for gradient_boosting_step. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem

TIE_TOL = 1e-9


def reference(X, y, F, learning_rate):
    X = np.asarray(X, dtype=np.float64)
    F = np.asarray(F, dtype=np.float64)
    r = np.asarray(y, dtype=np.float64) - F
    best = None  # (sse, feature, threshold, left, right) of the first candidate within TIE_TOL of the minimum
    scored = []
    for j in range(X.shape[1]):
        values = np.unique(X[:, j])
        for t in (values[:-1] + values[1:]) / 2:
            mask = X[:, j] <= t
            left, right = r[mask].mean(), r[~mask].mean()
            sse = np.sum((r[mask] - left) ** 2) + np.sum((r[~mask] - right) ** 2)
            scored.append((sse, j, float(t), float(left), float(right)))
    if not scored:
        m = float(r.mean())
        return -1, 0.0, m, m, F + learning_rate * m
    low = min(s[0] for s in scored)
    best = next(s for s in scored if s[0] <= low + TIE_TOL)
    _, j, t, left, right = best
    return j, t, left, right, F + learning_rate * np.where(X[:, j] <= t, left, right)


def compare(got, want, case):
    from gymkit.runner import default_compare

    if not isinstance(got, tuple) or len(got) != 5:
        raise AssertionError(f"return a tuple (feature, threshold, left_value, right_value, F_new), "
                             f"got {type(got).__name__}")
    feature = int(got[0]) if isinstance(got[0], (int, np.integer)) else got[0]
    if feature != want[0]:
        raise AssertionError(f"feature: got {got[0]!r}, expected {want[0]} "
                             "(minimum SSE; on a tie, the lowest feature index, then the lowest threshold)")
    default_compare(got[1], want[1], 1e-9, 1e-9, "threshold (midpoint between neighbouring distinct values)")
    for name, g, w in (("left_value", got[2], want[2]), ("right_value", got[3], want[3])):
        if isinstance(g, (int, float, np.number)) and w != 0 and np.isclose(float(g), -w, atol=1e-9, rtol=1e-7):
            raise AssertionError(f"{name}: got {float(g):.6g}, expected {w:.6g}; the sign is flipped. "
                                 "The negative gradient of squared loss is y - F, not F - y")
        default_compare(g, w, 1e-9, 1e-7, name)
    default_compare(got[4], want[4], 1e-9, 1e-7, "F_new (F + learning_rate * stump(X))")


def _random(seed, n, d, decimals, lr, name):
    rng = np.random.default_rng(seed)
    X = np.round(rng.uniform(-3, 3, (n, d)), decimals)
    y = np.sin(X[:, 0]) + 0.5 * X[:, -1] + 0.1 * rng.standard_normal(n)
    F = np.full(n, float(y.mean())) + 0.2 * rng.standard_normal(n)
    return {"_name": name, "X": X.tolist(), "y": np.round(y, 4).tolist(), "F": np.round(F, 4).tolist(),
            "learning_rate": lr}


def functional_cases():
    return [
        {"_name": "example", "X": [[1.0, 5.0], [2.0, 3.0], [3.0, 4.0], [4.0, 1.0]], "y": [1.0, 2.0, 6.0, 7.0],
         "F": [2.0, 2.0, 4.0, 4.0], "learning_rate": 0.5},
        {"_name": "residuals, not targets", "X": [[0.0], [1.0], [2.0], [3.0]], "y": [10.0, 10.0, 10.0, 10.0],
         "F": [9.0, 9.0, 12.0, 12.0], "learning_rate": 1.0},
        {"_name": "tie: lowest threshold wins", "X": [[0.0], [1.0], [2.0], [3.0]], "y": [0.0, 1.0, 1.0, 0.0],
         "F": [0.0, 0.0, 0.0, 0.0], "learning_rate": 1.0},
        {"_name": "tie: lowest feature wins", "X": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], "y": [0.0, 0.0, 3.0],
         "F": [0.0, 0.0, 0.0], "learning_rate": 1.0},
        {"_name": "duplicates and unsorted values", "X": [[3.0], [1.0], [1.0], [2.0], [3.0]],
         "y": [4.0, 1.0, 1.5, 1.0, 5.0], "F": [1.0, 1.0, 1.0, 1.0, 1.0], "learning_rate": 0.3},
        {"_name": "second feature splits best", "X": [[0.0, 10.0], [1.0, 20.0], [2.0, 10.0], [3.0, 20.0]],
         "y": [1.0, 5.0, 1.0, 5.0], "F": [0.0, 0.0, 0.0, 0.0], "learning_rate": 0.1},
        {"_name": "no split possible: constant features", "X": [[2.0, 7.0], [2.0, 7.0], [2.0, 7.0]],
         "y": [1.0, 2.0, 6.0], "F": [1.0, 1.0, 1.0], "learning_rate": 0.5},
        {"_name": "single sample", "X": [[4.0]], "y": [3.0], "F": [1.0], "learning_rate": 0.1},
        {"_name": "perfect fit already: zero residuals", "X": [[0.0], [1.0]], "y": [2.0, 3.0], "F": [2.0, 3.0],
         "learning_rate": 0.5},
        _random(1, 40, 3, 1, 0.1, "random 40x3, many duplicate values"),
        _random(2, 300, 6, 3, 0.05, "random 300x6"),
    ]


def build() -> Problem:
    return Problem(
        slug="gradient_boosting_step",
        title="Gradient boosting step",
        difficulty="hard",
        summary="One boosting round for squared loss: fit a regression stump to the negative gradient and add it with shrinkage.",
        params=["X", "y", "F", "learning_rate"],
        reference=reference,
        functional_cases=functional_cases,
        compare=compare,
        tags=["numpy", "ensembles", "regression trees"],
    )
