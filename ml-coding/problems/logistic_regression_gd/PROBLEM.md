# Logistic regression with gradient descent

Write `solve(X, y, lr, steps, l2=0.0, threshold=0.5)` that trains a binary logistic-regression model by full-batch gradient descent.

- `X`: an `N × D` list of lists (or array) of floats. `N ≥ 1`, `D ≥ 1`.
- `y`: `N` labels. Every label must be `0` or `1`; otherwise (for example `2`, or `-1` from a ±1 encoding) raise `ValueError`.
- `lr`: the learning rate. `steps`: the number of updates, an integer `≥ 0`.
- `l2`: the L2 strength `λ ≥ 0`. It applies to `w` only, not to the bias.
- `threshold`: a probability in `(0, 1)`.

The model is `p = σ(z)`, `z = X·w + b`, `σ(z) = 1 / (1 + e^(−z))`. Start from `w = 0`, `b = 0`. The objective is

```
J(w, b) = −(1/N) Σᵢ [ yᵢ log pᵢ + (1 − yᵢ) log(1 − pᵢ) ] + (λ/2) ‖w‖²
```

Each step computes the gradient of `J` on all `N` rows at the current parameters, then updates `w` and `b` together: `w ← w − lr·∂J/∂w`, `b ← b − lr·∂J/∂b`.

Return a tuple `(w, b, loss, preds)`:

- `w`: a NumPy array of shape `(D,)`; `b`: a Python float.
- `loss`: `J` at the returned parameters, a Python float.
- `preds`: an integer array of shape `(N,)` for the training rows: `1` where `p ≥ threshold`, else `0`.

Features can be in the thousands, so `|z|` reaches millions. Then `e^(−z)` overflows and `p` rounds to exactly 0 or 1, so `log p` is `−inf`. Your sigmoid, gradients and loss must stay finite and accurate. The grader runs your code under `np.errstate(over="raise", invalid="raise")`, so an overflowing `exp` fails the test even when the result would come out right.

```
solve([[1.0], [2.0], [-1.0], [-2.0]], [1, 1, 0, 0], 0.5, 1)
-> (array([0.375]), 0.0, 0.4550, array([1, 1, 0, 0]))
```

**Interview focus:** a sigmoid that never overflows, cross-entropy computed from `z` (log-sum-exp form) instead of `log(σ(z))`, the gradient `Xᵀ(p − y)/N + λw`, validating 0/1 targets, and the decision threshold.

**Follow-ups** to answer out loud once your solution passes:

- The classes are perfectly separable. What happens to the weights without regularization?
- How do you extend this to many classes?
- How would you calibrate the output probabilities?
