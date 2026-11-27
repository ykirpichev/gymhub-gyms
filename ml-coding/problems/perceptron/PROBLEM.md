# Perceptron learning rule

Write `solve(X, y, epochs)` that trains a perceptron online.

- `X`: an `N × D` list of lists (or array) of floats. `N ≥ 1`, `D ≥ 1`.
- `y`: `N` labels, each `-1` or `+1`. Any other value raises `ValueError`.
- `epochs`: an integer `≥ 0`, the number of passes over the data.

Start from `w = 0` (length `D`) and `b = 0`. In each epoch, visit the samples in the given order `i = 0, …, N-1`. Sample `i` is a **mistake** when `y[i] · (w · X[i] + b) <= 0`; a margin of exactly `0` counts as a mistake. On a mistake, and only then, update `w ← w + y[i] · X[i]` and `b ← b + y[i]` before moving to the next sample. Always run all `epochs`, even when an epoch makes no mistakes.

Return a tuple `(w, b, n_updates)`:

- `w`: a NumPy array of shape `(D,)`.
- `b`: a float.
- `n_updates`: an int, the total number of updates over all epochs.

```
solve([[2, 1], [-1, -2], [1, -1]], [1, -1, -1], 3)
-> (array([1., 2.]), 0.0, 2)
```

The first sample is a mistake because `w` and `b` start at zero (margin 0). The third sample is misclassified next. After that, every sample has a positive margin, so nothing more changes.

**Interview focus:** the update condition (only on mistakes, with `<= 0`), the online order of updates, convergence on separable data and endless updates on non-separable data, and the decision boundary `w · x + b = 0`.

**Follow-ups** to answer out loud once your solution passes:

- What does the perceptron convergence theorem say, and what does it need?
- How is the perceptron different from logistic regression trained with SGD?
- How would you get a margin out of it? (The averaged or voted perceptron.)
