"""Grader for linear_regression_gd. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def reference(X, y, lr, steps):
    x = np.asarray(X, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    if t.shape != (n,):
        return "ValueError"
    w, b = np.zeros(d), 0.0
    losses = []
    for _ in range(steps + 1):
        residual = x @ w + b - t
        losses.append(np.mean(residual ** 2))
        if len(losses) > steps:
            break
        w = w - lr * 2.0 * (x.T @ residual) / n
        b = b - lr * 2.0 * residual.mean()
    return w, float(b), np.array(losses)


def call(module, args):
    """A y of the wrong length must raise ValueError; anything else is returned for comparison."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(want, str) or isinstance(got, str):
        if not isinstance(got, str) or got != want:
            raise AssertionError("expected ValueError: y has a different length than X" if isinstance(want, str)
                                 else "raised ValueError, but X and y match")
        return
    if not isinstance(got, tuple) or len(got) != 3:
        raise AssertionError(f"return a tuple (w, b, losses), got {type(got).__name__}")
    default_compare(got[0], want[0], 1e-8, 1e-6, "w")
    default_compare(got[1], want[1], 1e-8, 1e-6, "b")
    default_compare(got[2], want[2], 1e-8, 1e-6, "losses")


def _random(seed, n, d, lr, steps, name, offset=0.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d))
    w = rng.standard_normal(d)
    y = x @ w + offset + 0.1 * rng.standard_normal(n)
    return {"_name": name, "X": x.tolist(), "y": y.tolist(), "lr": lr, "steps": steps}


def functional_cases():
    return [
        {"_name": "example", "X": [[1.0], [2.0], [3.0]], "y": [2.0, 4.0, 6.0], "lr": 0.1, "steps": 2},
        {"_name": "zero steps", "X": [[1.0, 2.0], [3.0, 4.0]], "y": [1.0, -1.0], "lr": 0.1, "steps": 0},
        {"_name": "bias only: constant target, zero features", "X": [[0.0], [0.0], [0.0], [0.0]],
         "y": [5.0, 5.0, 5.0, 5.0], "lr": 0.25, "steps": 3},
        {"_name": "single example", "X": [[2.0, -1.0]], "y": [3.0], "lr": 0.05, "steps": 10},
        {"_name": "intercept needed", "X": [[0.0], [1.0], [2.0], [3.0]], "y": [10.0, 12.0, 14.0, 16.0],
         "lr": 0.1, "steps": 200},
        {"_name": "features on very different scales", "X": [[1.0, 1000.0], [2.0, 3000.0], [3.0, 2000.0], [4.0, 5000.0]],
         "y": [1.0, 2.0, 3.0, 4.0], "lr": 1e-7, "steps": 50},
        {"_name": "learning rate too large diverges", "X": [[1.0], [2.0], [3.0]], "y": [1.0, 2.0, 3.0],
         "lr": 0.5, "steps": 20},
        _random(1, 20, 3, 0.05, 100, "random 20x3", offset=2.0),
        _random(2, 500, 10, 0.01, 300, "random 500x10", offset=-3.0),
        {"_name": "y length mismatch", "X": [[1.0], [2.0]], "y": [1.0, 2.0, 3.0], "lr": 0.1, "steps": 1},
    ]


def build() -> Problem:
    return Problem(
        slug="linear_regression_gd",
        title="Linear regression with gradient descent",
        difficulty="medium",
        summary="Full-batch gradient descent on mean squared error with a bias term, returning the loss history.",
        params=["X", "y", "lr", "steps"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "regression", "optimization", "vectorization"],
    )
