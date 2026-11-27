## Hint 1
Subtract `X.mean(axis=0)` before any decomposition. Without that, the leading direction points from the origin toward the cloud, which is not the principal component.

## Hint 2
`np.linalg.svd(Xc, full_matrices=False)` returns `Vt` whose rows are components, already in decreasing singular-value order. Eigenvectors of `Xc.T @ Xc` come out in the opposite order, so reverse them. The ratio is `s[:k]**2 / s**2.sum()` (or the same with eigenvalues). The divisor is every component, including ones you do not return.

## Hint 3
SVD and eigendecomposition can disagree on sign. `j = np.argmax(np.abs(component))` is the largest-absolute entry, and the lowest index when several tie. If `component[j] < 0`, multiply that row by `-1`.
