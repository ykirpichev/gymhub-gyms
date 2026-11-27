"""Grader for matrix_factorization. Private: learners see PROBLEM.md, HINTS.md and the starter only."""

import numpy as np

from gymkit.spec import Problem


def reference(ratings, P, Q, lr, lam, epochs):
    P = np.array(P, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)
    if P.shape[1] != Q.shape[1]:
        return "ValueError"
    for _ in range(epochs):
        for u, i, r in ratings:
            pu, qi = P[u].copy(), Q[i].copy()
            err = r - pu @ qi
            P[u] = pu + lr * (err * qi - lam * pu)
            Q[i] = qi + lr * (err * pu - lam * qi)
    residuals = np.array([r - P[u] @ Q[i] for u, i, r in ratings])
    return P, Q, float(np.sqrt(np.mean(residuals ** 2)))


def call(module, args):
    """A rank mismatch must raise ValueError; anything else is returned for comparison."""
    try:
        return module.solve(*args)
    except ValueError:
        return "ValueError"


def compare(got, want, case):
    from gymkit.runner import default_compare

    if isinstance(want, str) or isinstance(got, str):
        if got != want:
            raise AssertionError("expected ValueError: P and Q have different ranks" if want == "ValueError"
                                 else "raised ValueError, but P and Q have the same rank")
        return
    if not isinstance(got, tuple) or len(got) != 3:
        raise AssertionError(f"return a tuple (P, Q, rmse), got {type(got).__name__}")
    default_compare(got[0], want[0], 1e-6, 1e-6, "P")
    default_compare(got[1], want[1], 1e-6, 1e-6, "Q")
    default_compare(got[2], want[2], 1e-6, 1e-6, "rmse (over the observed ratings only)")


def _random(seed, n_users, n_items, rank, density, epochs, lam, name):
    rng = np.random.default_rng(seed)
    true_p = rng.standard_normal((n_users, rank))
    true_q = rng.standard_normal((n_items, rank))
    mask = rng.random((n_users, n_items)) < density
    users, items = np.nonzero(mask)
    order = rng.permutation(len(users))
    ratings = [[int(users[j]), int(items[j]), round(float(true_p[users[j]] @ true_q[items[j]] + 3.0), 2)]
               for j in order]
    return {"_name": name, "ratings": ratings,
            "P": (rng.standard_normal((n_users, rank)) * 0.5).tolist(),
            "Q": (rng.standard_normal((n_items, rank)) * 0.5).tolist(),
            "lr": 0.02, "lam": lam, "epochs": epochs}


def functional_cases():
    small = [[0, 0, 5.0], [0, 1, 3.0], [1, 0, 4.0]]
    return [
        {"_name": "example", "ratings": small, "P": [[1.0], [1.0]], "Q": [[1.0], [0.5]],
         "lr": 0.1, "lam": 0.1, "epochs": 1},
        {"_name": "zero epochs: RMSE of the initial factors", "ratings": small, "P": [[1.0], [2.0]],
         "Q": [[2.0], [1.0]], "lr": 0.1, "lam": 0.1, "epochs": 0},
        # User 2 and item 2 have no ratings: their factors must not change (no regularization either).
        {"_name": "unobserved rows are untouched (cold start)", "ratings": small,
         "P": [[1.0, 0.5], [0.5, 1.0], [0.3, -0.2]], "Q": [[1.0, -0.5], [0.5, 0.5], [0.7, 0.1]],
         "lr": 0.05, "lam": 0.2, "epochs": 5},
        {"_name": "both factors use pre-update values", "ratings": [[0, 0, 4.0]], "P": [[1.0, 2.0]],
         "Q": [[0.5, -1.0]], "lr": 0.2, "lam": 0.0, "epochs": 1},
        {"_name": "no regularization", "ratings": small, "P": [[0.5, 0.1], [0.4, -0.3]],
         "Q": [[0.6, 0.2], [0.3, 0.5]], "lr": 0.05, "lam": 0.0, "epochs": 50},
        {"_name": "strong regularization", "ratings": small, "P": [[0.5, 0.1], [0.4, -0.3]],
         "Q": [[0.6, 0.2], [0.3, 0.5]], "lr": 0.05, "lam": 2.0, "epochs": 50},
        {"_name": "order of triples matters", "ratings": [[1, 0, 4.0], [0, 1, 3.0], [0, 0, 5.0]],
         "P": [[1.0], [1.0]], "Q": [[1.0], [0.5]], "lr": 0.1, "lam": 0.1, "epochs": 1},
        {"_name": "rank mismatch", "ratings": small, "P": [[1.0], [1.0]], "Q": [[1.0, 0.0], [0.5, 0.0]],
         "lr": 0.1, "lam": 0.1, "epochs": 1},
        _random(1, 8, 6, 1, 0.5, 30, 0.01, "random rank 1, 8x6"),
        _random(2, 40, 30, 3, 0.2, 40, 0.05, "random rank 3, 40x30"),
        _random(3, 25, 25, 5, 0.3, 20, 0.1, "random rank 5, 25x25"),
    ]


def build() -> Problem:
    return Problem(
        slug="matrix_factorization",
        title="Matrix factorization with SGD",
        difficulty="medium",
        summary="Fit user and item factors to the observed ratings only, with SGD and L2 regularization, and report the training RMSE.",
        params=["ratings", "P", "Q", "lr", "lam", "epochs"],
        reference=reference,
        functional_cases=functional_cases,
        call=call,
        compare=compare,
        tags=["numpy", "recommendation", "sgd"],
    )
