## Hint 1
The gradient is the same as for linear regression with `p − y` in place of the residual: `∂J/∂w = Xᵀ(p − y)/N + λw` and `∂J/∂b = mean(p − y)`.

## Hint 2
For a stable sigmoid, only ever exponentiate a non-positive number. With `e = exp(−|z|)`: `σ(z) = 1/(1 + e)` when `z ≥ 0`, and `e/(1 + e)` when `z < 0`. Note that `np.where(z >= 0, 1/(1 + np.exp(-z)), ...)` still evaluates `np.exp(-z)` for every element.

## Hint 3
Rewrite the per-example loss in terms of `z`: `−[y log σ(z) + (1 − y) log(1 − σ(z))] = log(1 + e^z) − y·z`. Compute `log(1 + e^z)` as `max(z, 0) + log1p(exp(−|z|))`, or with `np.logaddexp(0, z)`.
