# Linear regression with gradient descent

Write `solve(X, y, lr, steps)` that fits `ŷ = X·w + b` by full-batch gradient descent on the mean squared error.

- `X`: an `N × D` list of lists (or array) of floats. `N ≥ 1`, `D ≥ 1`.
- `y`: `N` float targets. If `len(y) != N`, raise `ValueError`.
- `lr`: the learning rate, a positive float.
- `steps`: the number of updates, an integer `≥ 0`.

Start from `w = 0` (all `D` entries) and `b = 0`. The loss is `L(w, b) = (1/N) Σᵢ (xᵢ·w + b − yᵢ)²`. Each step computes the gradient of `L` on **all** `N` rows at the current parameters, then updates `w` and `b` together:

```
w ← w − lr · ∂L/∂w        b ← b − lr · ∂L/∂b
```

Return a tuple `(w, b, losses)`:

- `w`: a NumPy array of shape `(D,)`.
- `b`: a Python float.
- `losses`: `steps + 1` floats (a list or array): `losses[t]` is `L` after `t` updates, so `losses[0]` is the loss at zero and `losses[-1]` the loss of the returned parameters.

Do not rescale the features and do not stop early. With a learning rate that is too large the loss grows at every step. Return those numbers as they are.

```
solve([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0], 0.1, 2)
-> (array([1.6711]), 0.6933, [18.6667, 0.2963, 0.0734])
```

**Interview focus:** the MSE gradient (including the factor 2 and the 1/N), updating the bias, vectorising over the batch, and how feature scale and the learning rate decide convergence.

**Follow-ups** to answer out loud once your solution passes:

- When would you use the closed-form solution instead, and when can't you?
- What happens with two perfectly correlated features?
- How do you pick the learning rate, and what does feature scaling do to convergence?
