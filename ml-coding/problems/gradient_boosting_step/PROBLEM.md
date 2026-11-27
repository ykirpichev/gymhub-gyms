# Gradient boosting step

Write `solve(X, y, F, learning_rate)` that takes one boosting step for squared loss.

- `X`: an `N × D` list of lists (or array) of floats. `N ≥ 1`, `D ≥ 1`.
- `y`: `N` float targets.
- `F`: `N` floats, the current prediction of each row.
- `learning_rate`: a float. The stump is multiplied by this before it is added.

The loss on one row is `½(y − F)²`, so the negative gradient — the residual to fit — is `r = y − F`. Do not fit `y`, and do not fit `F − y`.

Fit one regression stump to `r`. It tests a single feature: rows with `x[j] ≤ threshold` go left, the rest go right. Each leaf value is the **mean of the residuals** on that side. The score of a split is the residual sum of squares around those two means. Choose the smallest score.

Candidate thresholds on feature `j` are the midpoints between consecutive **distinct** values of that column, after sorting. A column with only one distinct value has no candidate. The rows are not pre-sorted.

If several scores are within `1e-9` absolute of the minimum, choose the lowest feature index, then the lowest threshold.

When no feature has a candidate (`N = 1`, or every column is constant), there is no split. Use feature `-1`, threshold `0.0`, and set both leaves to the mean of `r`. The update is that constant on every row.

Otherwise the update of a row is the leaf it falls into. In either case

```
F_new = F + learning_rate * update
```

Return `(feature, threshold, left_value, right_value, F_new)`:

- `feature`: an int, or `-1` when there is no split.
- `threshold`: a float.
- `left_value`, `right_value`: floats, the leaf means of the residuals.
- `F_new`: a float array of shape `(N,)`.

A legal split is returned even when every residual is already 0. Its leaves are 0, so `F` does not move. `feature` is `-1` only when no threshold exists.

```
solve([[1.0, 5.0], [2.0, 3.0], [3.0, 4.0], [4.0, 1.0]], [1.0, 2.0, 6.0, 7.0], [2.0, 2.0, 4.0, 4.0], 0.5)
-> (0, 2.5, -0.5, 2.5, array([1.75, 1.75, 5.25, 5.25]))
```

Residuals are `[-1, 0, 2, 3]`. Feature 0 at 2.5 puts the first two on the left (mean `-0.5`) and the last two on the right (mean `2.5`). Learning rate `0.5` moves `F` by half of that.

**Interview focus:** negative gradients of squared loss, a one-split weak learner, and learning-rate shrinkage.

**Follow-ups** to answer out loud once your solution passes:

- How does the next step change for logistic loss instead of squared loss?
- What do the learning rate and the number of trees trade off?
- Why do XGBoost and LightGBM use second-order (Hessian) information and histogram-based splits?
