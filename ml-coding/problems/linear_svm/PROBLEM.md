# Linear SVM with hinge loss

Write `solve(X, y, lam, lr, steps)` that trains a linear SVM in the primal by full-batch subgradient descent.

- `X`: an `N × D` list of lists (or array) of floats. `N ≥ 1`, `D ≥ 1`.
- `y`: `N` labels, each `-1` or `+1`. Any other value raises `ValueError`.
- `lam`: the regularization strength `λ ≥ 0`. `lr`: the learning rate `> 0`. `steps`: an integer `≥ 0`.

With `f(x) = w · x + b`, minimize

```
J(w, b) = (1/N) Σᵢ max(0, 1 - y[i] · f(X[i]))  +  (λ/2) · ||w||²
```

The bias `b` is **not** regularized. Start from `w = 0`, `b = 0` and take `steps` full-batch steps. At each step, using the current `(w, b)`:

- sample `i` is a margin violation when `y[i] · f(X[i]) < 1`. At exactly `1` it contributes nothing to the subgradient;
- `g_w = λ · w - (1/N) Σ_{violations} y[i] · X[i]`
- `g_b = -(1/N) Σ_{violations} y[i]`
- `w ← w - lr · g_w`, `b ← b - lr · g_b`, both computed from the values before the step.

Return a tuple `(w, b, objective)`: `w` is a NumPy array of shape `(D,)`, `b` a float, and `objective` the float `J(w, b)` at the final parameters.

```
solve([[2, 0], [0, 1], [-1, -1]], [1, 1, -1], lam=0.1, lr=0.5, steps=2)
-> (array([0.64166667, 0.65]), 0.16666667, 0.10282292)
```

In step 1 every margin is 0, so all three samples are violations: `w = [0.5, 0.3333]`, `b = 0.1667`. In step 2 the first sample has margin `1.1667 ≥ 1` and drops out.

**Interview focus:** the hinge loss, which samples are margin violations (`< 1`, not `<= 1`), the regularization gradient `λw` (not applied to `b`), and the 1/N in the mean.

**Follow-ups** to answer out loud once your solution passes:

- How does the kernel trick extend this to non-linear boundaries, and what does it cost at scale?
- What does the regularization strength do to the margin?
- Why are only the support vectors needed at prediction time?
