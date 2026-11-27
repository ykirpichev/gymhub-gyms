# Matrix factorization with SGD

Write `solve(ratings, P, Q, lr, lam, epochs)` that fits a low-rank model `rating(u, i) ≈ P[u] · Q[i]` to the observed ratings with stochastic gradient descent.

- `ratings`: a non-empty list of observed triples `[u, i, r]`: integer user index `u`, integer item index `i`, float rating `r`. Every other user/item pair is **unobserved** (not a zero rating).
- `P`: the initial user factors, `n_users × k`. `Q`: the initial item factors, `n_items × k`. If their ranks `k` differ, raise `ValueError`.
- `lr > 0`, `lam ≥ 0`, `epochs ≥ 0` (integer).

For each epoch, visit the triples in the given order. For triple `(u, i, r)`, with `p = P[u]` and `q = Q[i]` taken **before** this update:

```
e    = r - p · q
P[u] = p + lr · (e · q - lam · p)
Q[i] = q + lr · (e · p - lam · q)
```

Only the rows `P[u]` and `Q[i]` of the current triple change. A user or item with no ratings keeps its initial factors.

Return a tuple `(P, Q, rmse)`: the trained factors as NumPy arrays of the input shapes (don't modify the inputs), and `rmse = sqrt(mean of (r - P[u] · Q[i])²)` over the observed triples only, using the final factors.

```
solve([[0, 0, 5], [0, 1, 3], [1, 0, 4]], P=[[1], [1]], Q=[[1], [0.5]], lr=0.1, lam=0.1, epochs=1)
-> (array([[1.49135], [1.35279]]), array([[1.6371], [0.815395]]), 2.074926)
```

The first triple has `e = 5 - 1·1 = 4`, so `P[0] = 1 + 0.1·(4·1 - 0.1·1) = 1.39`. `Q[0]` uses the old `P[0] = 1` and also becomes `1.39`.

**Interview focus:** training and evaluating on observed entries only, L2 regularization on the factors, the latent rank `k`, updating both factors from pre-update values, and why cold-start users and items get no signal.

**Follow-ups** to answer out loud once your solution passes:

- How do you recommend for a brand-new user with no ratings?
- How would you handle implicit feedback (clicks) instead of explicit ratings?
- How does alternating least squares differ from this gradient update?
