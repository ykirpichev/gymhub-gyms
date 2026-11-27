# k-means clustering

Write `solve(X, centroids, max_iter)` that runs Lloyd's algorithm from the given centroids.

- `X`: an `N × D` list of lists (or array) of floats. `N ≥ 1`, `D ≥ 1`.
- `centroids`: a `K × D` list of lists of floats. `K ≥ 1`, and the second dimension equals `D`.
- `max_iter`: an integer `≥ 0`, the maximum number of assignment-plus-update rounds.

Raise `ValueError` if `max_iter < 0`, if either array is not a matrix, if a dimension is 0, or if the two matrices do not share `D`.

Distance is squared Euclidean. Each round:

1. Assign every point to the nearest centroid. If several centroids are equally far, choose the **lowest index**.
2. Replace each centroid by the mean of the points assigned to it. A cluster with no points **keeps its current centroid**.
3. If every coordinate of every centroid changed by at most `1e-8`, stop.

`max_iter = 0` does no update and returns the centroids you were given. After the loop, assign every point once more, so the labels match the centroids you return.

Return `(centroids, labels)`:

- `centroids`: a float array of shape `(K, D)`.
- `labels`: an integer array of shape `(N,)`.

```
solve([[0, 0], [1, 0], [8, 0], [9, 0]], [[0, 0], [9, 0]], 10)
-> (array([[0.5, 0.0], [8.5, 0.0]]), array([0, 0, 1, 1]))
```

The first two points fall into cluster 0 and the last two into cluster 1. One update reaches the means, and the assignment does not change.

**Interview focus:** initialization (passed in), the assignment and update steps, and empty clusters.

**Follow-ups** to answer out loud once your solution passes:

- How would you choose k?
- Explain k-means++ initialization and why it helps.
- Your features have very different scales. What happens, and what do you do?
