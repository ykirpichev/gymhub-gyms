## Hint 1
Never build the dense `n_users × n_items` matrix and treat its missing cells as 0: the loss, the updates and the RMSE all run over the list of observed triples only.

## Hint 2
The gradient of `(r - p·q)² / 2 + lam/2 · (|p|² + |q|²)` with respect to `p` is `-(e·q) + lam·p`, and symmetrically for `q`. If you update `P[u]` in place first, `Q[i]` would see the new `P[u]`. Keep a copy of the old row.

## Hint 3
Copy `P` and `Q` into float arrays. Then for each epoch and each `(u, i, r)`: compute `e`, save `p_old = P[u].copy()`, update `P[u]`, then update `Q[i]` with `p_old`. At the end, compute the residuals on the triples and return the RMSE.
