"""Grader for linear_svm. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def reference(X, y, lam, lr, steps):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if not np.all((y == 1) | (y == -1)):
        return "ValueError"
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(steps):
        active = y * (X @ w + b) < 1
        grad_w = lam * w - (y[active] @ X[active]) / n
        grad_b = -y[active].sum() / n
        w, b = w - lr * grad_w, b - lr * grad_b
    objective = np.maximum(0.0, 1 - y * (X @ w + b)).mean() + 0.5 * lam * (w @ w)
    return w, float(b), float(objective)


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
        raise AssertionError(f"return a tuple (w, b, objective), got {type(got).__name__}")
    default_compare(got[0], want[0], 1e-6, 1e-6, "w")
    default_compare(got[1], want[1], 1e-6, 1e-6, "b")
    default_compare(got[2], want[2], 1e-6, 1e-6, "objective")


def _random(seed, n, d, lam, lr, steps, flip, name):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d))
    y = np.where(X @ rng.standard_normal(d) + 0.5 > 0, 1, -1)
    y[rng.random(n) < flip] *= -1
    return {"_name": name, "X": X.tolist(), "y": y.tolist(), "lam": lam, "lr": lr, "steps": steps}


def functional_cases():
    return [
        {"_name": "example", "X": [[2.0, 0.0], [0.0, 1.0], [-1.0, -1.0]], "y": [1, 1, -1],
         "lam": 0.1, "lr": 0.5, "steps": 2},
        {"_name": "zero steps", "X": [[1.0, 2.0], [3.0, -1.0]], "y": [1, -1], "lam": 0.5, "lr": 0.1, "steps": 0},
        # After one step w = b = 0.5, so the margin is exactly 1: no further hinge gradient.
        {"_name": "margin exactly 1 is not a violation", "X": [[1.0]], "y": [1], "lam": 0.0, "lr": 0.5, "steps": 3},
        {"_name": "only violators pull on w", "X": [[4.0, 0.0], [0.0, 1.0], [-4.0, 0.0], [0.0, -0.5]],
         "y": [1, 1, -1, -1], "lam": 0.0, "lr": 0.25, "steps": 5},
        {"_name": "bias is not regularized", "X": [[1.0], [2.0], [3.0]], "y": [1, 1, 1],
         "lam": 2.0, "lr": 0.1, "steps": 30},
        {"_name": "strong regularization shrinks w", "X": [[1.0, 1.0], [2.0, 0.5], [-1.0, -1.5], [-2.0, 0.0]],
         "y": [1, 1, -1, -1], "lam": 5.0, "lr": 0.05, "steps": 100},
        {"_name": "label 0 is invalid", "X": [[1.0], [2.0]], "y": [1, 0], "lam": 0.1, "lr": 0.1, "steps": 1},
        _random(1, 30, 2, 0.01, 0.1, 200, 0.0, "random separable 30x2"),
        _random(2, 120, 6, 0.1, 0.05, 300, 0.1, "random noisy 120x6"),
        _random(3, 80, 4, 1.0, 0.02, 150, 0.2, "random strong lambda 80x4"),
    ]


def build() -> Problem:
    return Problem(
        slug="linear_svm",
        title="Linear SVM with hinge loss",
        difficulty="medium",
        summary="Train a primal linear SVM with full-batch subgradient descent on mean hinge loss plus an L2 penalty on the weights.",
        params=["X", "y", "lam", "lr", "steps"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "optimization", "linear classification"],
    )
