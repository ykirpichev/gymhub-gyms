# Decision-tree split

Write `solve(X, y)` that finds the best single split of numeric features by weighted Gini impurity.

- `X`: an `N × D` list of lists (or array) of floats. `N ≥ 1`, `D ≥ 1`.
- `y`: `N` integer labels. They need not be `0 .. C-1`. If `len(y) != N`, or `X` is not a matrix, raise `ValueError`.

Gini impurity of a node with class counts `n_c` and `n` rows is `1 − Σ_c (n_c / n)²`. A split's score is the **weighted** impurity

```
(n_left * gini_left + n_right * gini_right) / n
```

not the average of the two sides and not the reduction from the parent.

On feature `j`, sort the **distinct** values and take the midpoint of each consecutive pair. That is the only candidate threshold. A feature with one distinct value contributes none. The left side is every row with `x[j] ≤ threshold`. No training value equals a midpoint, so `<` and `≤` select the same rows, but the threshold you return must be the midpoint.

Choose the minimum score. If several scores are within `1e-12` absolute of the minimum, choose the lowest feature index, then the lowest threshold.

Return `(feature, threshold, impurity)`: an int, a float and a float. `impurity` is the weighted Gini of the chosen split.

Return `None` when no candidate exists (one row, or every feature is constant) or when no candidate is strictly below the parent's Gini (every label is equal, or every split leaves the class mix unchanged). A difference of `1e-12` or less does not count as an improvement.

```
solve([[1, 2], [2, 1], [3, 2], [4, 1]], [0, 0, 1, 1])
-> (0, 2.5, 0.0)
```

Feature 0 at 2.5 separates the two classes. The parent Gini is 0.5; the split's weighted Gini is 0, which is what you return.

**Interview focus:** candidate thresholds, weighted impurity, and stopping when no split improves.

**Follow-ups** to answer out loud once your solution passes:

- How would you find the best threshold in O(N log N) per feature instead of O(N²)?
- How do trees handle missing values in practice?
- Why do single trees overfit, and how do random forests and boosting each fix it?
