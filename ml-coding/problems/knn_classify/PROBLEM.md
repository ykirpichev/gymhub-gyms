# k-nearest neighbours classifier

Write `solve(X_train, y_train, X_test, k)` that labels each test point by a majority vote of its `k` nearest training points.

- `X_train`: an `N × D` list of lists (or array) of floats. `N ≥ 1`, `D ≥ 1`.
- `y_train`: `N` integer labels, each `≥ 0`.
- `X_test`: an `M × D` list of lists of floats. `M ≥ 1`.
- `k`: an integer. If `k < 1` or `k > N`, raise `ValueError`.

Return a NumPy integer array of shape `(M,)`.

Distance is Euclidean. Ties are broken like this, so the answer is unique:

1. **Equal distances.** Rank training points by distance, and among equal distances by training index, lower first. The `k` neighbours are the first `k` in that order.
2. **Equal votes.** If several labels share the highest count among the `k` neighbours, return the **smallest** of those labels (not the label of the nearest neighbour).

The hidden tests include thousands of training and test points. Compute all distances with array operations, not a Python loop over pairs.

```
solve([[0, 0], [1, 0], [0, 1], [5, 5], [6, 5]], [0, 0, 1, 1, 1], [[0.2, 0.1], [5.5, 4.0]], 3)
-> array([0, 1])
```

**Interview focus:** vectorised pairwise distances (`‖a‖² + ‖b‖² − 2a·b`), selecting the top `k`, deterministic tie-breaking, and the `O(M·N)` time and memory cost of brute force.

**Follow-ups** to answer out loud once your solution passes:

- N is ten million. How do you make queries fast? (KD-trees for low dimensions, approximate indexes for high.)
- Why does k-NN degrade in high dimensions?
- How would you weight neighbours by distance?
