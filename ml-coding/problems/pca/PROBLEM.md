# Principal component analysis

Write `solve(X, k)` for PCA on a data matrix.

- `X`: an `N × D` list of lists (or array) of floats. `N ≥ 2`, `D ≥ 1`.
- `k`: an integer, how many components to return.

Raise `ValueError` if `k < 1`, if `k > D`, or if `k > N - 1`. Also raise `ValueError` when every row of `X` is equal: after centering, the covariance is zero and no direction is defined.

Center each column (subtract its mean). Then take a thin SVD of the centered matrix, or the eigendecomposition of `Xc.T @ Xc`. Either one. Components are unit vectors in feature space, one per row, ordered by **decreasing** variance. The hidden tests keep a clear gap between those variances, so the directions are unique up to sign.

Sign: in each component, look at the entry with the largest absolute value. If several entries tie, use the smallest index. If that entry is negative, multiply the whole component by `-1`.

Explained-variance ratio of a component is its variance divided by the **total** variance of all `D` components (the zero ones count in the total, and they add nothing). Do not renormalise the `k` ratios to sum to 1. Using `N` or `N - 1` in the covariance does not change the ratio.

Return `(components, explained_variance_ratio)`:

- `components`: a float array of shape `(k, D)`.
- `explained_variance_ratio`: a float array of shape `(k,)`.

```
solve([[0, 10], [1, 10], [2, 10], [3, 11]], 1)
-> (array([[0.953144, 0.302516]]), array([0.952362]))
```

The cloud is almost horizontal, but the column means are about `(1.5, 10.25)`. Centering is what makes the first component point along the variation rather than toward the origin. The figures are rounded to 6 decimals.

**Interview focus:** centering, ordering components by variance, and the explained-variance ratio.

**Follow-ups** to answer out loud once your solution passes:

- Why compute PCA with an SVD of the centered data instead of the covariance matrix?
- How do you choose the number of components?
- When is PCA a bad idea? (Non-linear structure, features on different scales, supervised goals.)
