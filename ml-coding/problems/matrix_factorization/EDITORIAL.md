#### The idea
Only the observed ratings enter the loss, `Σ (r − pᵤ·qᵢ)² + λ(‖pᵤ‖² + ‖qᵢ‖²)`. SGD takes one triple at a time; the
gradient of that triple's term touches only `P[u]` and `Q[i]`. The one trap is the order of the two updates: both use
the factors from **before** this triple, so save `p` before overwriting `P[u]`.

#### A solution
```python
import numpy as np


def solve(ratings, P, Q, lr, lam, epochs):
    P = np.array(P, dtype=np.float64)
    Q = np.array(Q, dtype=np.float64)
    if P.shape[1] != Q.shape[1]:
        raise ValueError("P and Q must have the same rank")
    for _ in range(epochs):
        for u, i, r in ratings:
            err = r - P[u] @ Q[i]
            p_old = P[u].copy()
            P[u] += lr * (err * Q[i] - lam * P[u])
            Q[i] += lr * (err * p_old - lam * Q[i])
    errors = [r - P[u] @ Q[i] for u, i, r in ratings]
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    return P, Q, rmse
```

`np.array(P, ...)` copies, so the caller's lists are never modified.

#### What it costs
Each epoch is `O(|ratings|·k)`, independent of the full `n_users × n_items` matrix, which is the point: rating
matrices are 99% empty. The alternative is ALS (alternating least squares): fix `Q`, solve a small ridge regression for
every user in closed form, then swap. ALS parallelises well and handles implicit feedback (Spark's recommender uses
it); SGD is simpler and works for any differentiable loss.

#### Pitfalls
- Updating `Q[i]` with the new `P[u]`.
- Treating missing entries as zero ratings, which teaches the model that every user dislikes everything they have not
  seen.
- Computing the RMSE over the whole matrix.
- Modifying `P` and `Q` in place when the caller passed arrays.
