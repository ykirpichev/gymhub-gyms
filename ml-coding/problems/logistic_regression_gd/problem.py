"""Grader for logistic_regression_gd. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def _sigmoid(z):
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def reference(X, y, lr, steps, l2, threshold):
    x = np.asarray(X, dtype=np.float64)
    t = np.asarray(y, dtype=np.float64)
    n, d = x.shape
    if t.shape != (n,) or np.any((t != 0) & (t != 1)):
        return "ValueError"
    w, b = np.zeros(d), 0.0
    for _ in range(steps):
        g = _sigmoid(x @ w + b) - t
        grad_w = x.T @ g / n + l2 * w
        grad_b = g.sum() / n
        w, b = w - lr * grad_w, b - lr * grad_b
    z = x @ w + b
    # Per-example BCE: softplus(z) - y*z, softplus(z) = max(z, 0) + log1p(exp(-|z|)).
    nll = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z))) - t * z
    loss = nll.mean() + 0.5 * l2 * float(np.dot(w, w))
    preds = (_sigmoid(z) >= threshold).astype(np.int64)
    return w, float(b), float(loss), preds


def call(module, args):
    """Labels other than 0/1 must raise ValueError. Overflow is an error, so exp(1e6) cannot slip by."""
    try:
        with np.errstate(over="raise", invalid="raise"):
            return module.solve(*args)
    except ValueError:
        return "ValueError"
    except FloatingPointError as exc:
        return f"FloatingPointError: {exc}"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(got, str) and got.startswith("FloatingPointError"):
        raise AssertionError(f"{got}. Overflow or NaN inside your code: keep every exp() argument <= 0, "
                             "and compute the loss from z instead of log(p)")
    if isinstance(want, str) or isinstance(got, str):
        if not isinstance(got, str) or got != want:
            raise AssertionError("expected ValueError: labels must be 0 or 1" if isinstance(want, str)
                                 else "raised ValueError, but every label is 0 or 1")
        return
    if not isinstance(got, tuple) or len(got) != 4:
        raise AssertionError(f"return a tuple (w, b, loss, preds), got {type(got).__name__}")
    default_compare(got[0], want[0], 1e-8, 1e-6, "w")
    default_compare(got[1], want[1], 1e-8, 1e-6, "b")
    loss = got[2]
    if isinstance(loss, (float, int, np.floating)) and not np.isfinite(loss):
        raise AssertionError(f"loss: got {loss}, expected {want[2]:.6g} (log of a probability that rounded to 0 or 1?)")
    default_compare(loss, want[2], 1e-8, 1e-6, "loss")
    default_compare(got[3], want[3], 0, 0, "preds")


def _random(seed, n, d, lr, steps, l2, threshold, name, scale=1.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)) * scale
    w = rng.standard_normal(d)
    p = 1.0 / (1.0 + np.exp(-(x @ w / scale + 0.5)))
    y = (rng.random(n) < p).astype(int)
    return {"_name": name, "X": x.tolist(), "y": y.tolist(), "lr": lr, "steps": steps, "l2": l2, "threshold": threshold}


def _case(name, X, y, lr, steps, l2=0.0, threshold=0.5):
    return {"_name": name, "X": X, "y": y, "lr": lr, "steps": steps, "l2": l2, "threshold": threshold}


def functional_cases():
    return [
        _case("example", [[1.0], [2.0], [-1.0], [-2.0]], [1, 1, 0, 0], 0.5, 1),
        _case("zero steps: p = 0.5 everywhere, ties go to 1", [[3.0, -1.0], [0.5, 2.0]], [0, 1], 0.1, 0),
        _case("only the bias moves", [[0.0], [0.0], [0.0], [0.0]], [1, 1, 1, 0], 1.0, 25),
        _case("separable data, many steps", [[-3.0, 1.0], [-1.0, 0.5], [1.0, -0.5], [2.0, 1.0], [4.0, 0.0]],
              [0, 0, 1, 1, 1], 1.0, 200),
        _case("L2 shrinks the weights", [[-3.0, 1.0], [-1.0, 0.5], [1.0, -0.5], [2.0, 1.0], [4.0, 0.0]],
              [0, 0, 1, 1, 1], 1.0, 200, l2=0.3),
        _case("high threshold", [[1.0], [2.0], [3.0], [-1.0], [0.5]], [1, 1, 1, 0, 0], 0.2, 30, threshold=0.9),
        _case("huge features: |z| in the millions", [[5000.0, -2000.0], [-8000.0, 1000.0], [3000.0, 9000.0],
                                                     [-4000.0, -7000.0]], [1, 0, 0, 1], 0.1, 15),
        _case("huge features, misclassified points", [[1e4], [-1e4], [2e4], [-3e4]], [0, 1, 1, 0], 0.01, 5),
        _random(1, 30, 3, 0.3, 100, 0.0, 0.5, "random 30x3"),
        _random(2, 400, 8, 0.5, 300, 0.01, 0.35, "random 400x8, L2, threshold 0.35"),
        _random(3, 200, 5, 1e-3, 50, 0.1, 0.5, "random 200x5, features x1000", scale=1000.0),
        _case("label 2", [[1.0], [2.0]], [1, 2], 0.1, 5),
        _case("label -1 (use 0/1, not -1/+1)", [[1.0], [2.0]], [1, -1], 0.1, 5),
    ]


def build() -> Problem:
    return Problem(
        slug="logistic_regression_gd",
        title="Logistic regression with gradient descent",
        difficulty="hard",
        summary="Full-batch gradient descent on L2-regularised binary cross-entropy with a sigmoid and loss that never overflow.",
        params=["X", "y", "lr", "steps", "l2", "threshold"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "classification", "optimization", "numerical stability"],
    )
