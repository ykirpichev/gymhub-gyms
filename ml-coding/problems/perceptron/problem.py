"""Grader for perceptron. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def reference(X, y, epochs):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.all((y == 1) | (y == -1)):
        return "ValueError"
    w = np.zeros(X.shape[1])
    b = 0.0
    updates = 0
    for _ in range(epochs):
        for i in range(len(y)):
            if y[i] * (X[i] @ w + b) <= 0:
                w = w + y[i] * X[i]
                b += y[i]
                updates += 1
    return w, float(b), updates


def call(module, args):
    """Invalid labels must raise ValueError; anything else is returned for comparison."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(want, str) or isinstance(got, str):
        if got != want:
            raise AssertionError("expected ValueError for a label that is not -1 or +1" if want == "ValueError"
                                 else "raised ValueError, but every label is -1 or +1")
        return
    if not isinstance(got, tuple) or len(got) != 3:
        raise AssertionError(f"return a tuple (w, b, n_updates), got {type(got).__name__}")
    hint = ""
    if int(got[2]) != want[2]:
        hint = (" (a margin of exactly 0 counts as a mistake: the condition is y*(w.x+b) <= 0)"
                if int(got[2]) < want[2] else "")
        raise AssertionError(f"n_updates: got {got[2]!r}, expected {want[2]}{hint}")
    default_compare(got[0], want[0], 1e-9, 1e-9, "w")
    default_compare(got[1], want[1], 1e-9, 1e-9, "b")


def _random(seed, n, d, epochs, separable, name):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    if separable:
        true_w = rng.standard_normal(d)
        score = X @ true_w + 0.3
        keep = np.abs(score) > 0.2  # leave a gap so the data is strictly separable
        X, score = X[keep], score[keep]
        y = np.where(score > 0, 1, -1)
    else:
        y = rng.choice([-1, 1], size=n)
    return {"_name": name, "X": np.round(X, 3).tolist(), "y": y.tolist(), "epochs": epochs}


def functional_cases():
    return [
        {"_name": "example", "X": [[2.0, 1.0], [-1.0, -2.0], [1.0, -1.0]], "y": [1, -1, -1], "epochs": 3},
        {"_name": "zero margin is a mistake", "X": [[1.0, 0.0], [0.0, 1.0]], "y": [1, 1], "epochs": 1},
        {"_name": "point on the boundary after an update", "X": [[1.0], [-1.0], [1.0]], "y": [1, 1, 1], "epochs": 2},
        {"_name": "zero epochs", "X": [[1.0, 2.0]], "y": [1], "epochs": 0},
        {"_name": "separable: stops updating", "X": [[3.0, 1.0], [2.0, 2.0], [-1.0, -3.0], [-2.0, -1.0]],
         "y": [1, 1, -1, -1], "epochs": 20},
        {"_name": "not separable (XOR): never stops", "X": [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
         "y": [-1, 1, 1, -1], "epochs": 10},
        {"_name": "sample order matters", "X": [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], "y": [-1, 1, 1], "epochs": 4},
        {"_name": "label 0 is invalid", "X": [[1.0], [2.0]], "y": [1, 0], "epochs": 1},
        _random(1, 40, 3, 15, True, "random separable 40x3"),
        _random(2, 60, 5, 8, False, "random noisy labels 60x5"),
    ]


def build() -> Problem:
    return Problem(
        slug="perceptron",
        title="Perceptron learning rule",
        difficulty="easy",
        summary="Online perceptron with labels in {-1, +1}: update on every mistake, including a zero margin, for a fixed number of epochs.",
        params=["X", "y", "epochs"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "online learning", "linear classification"],
    )
