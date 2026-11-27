# Pairwise squared distances

Write `solve(X, Y)` that returns the `N × M` matrix of squared Euclidean distances between the rows of `X` (`N × D`) and the rows of `Y` (`M × D`), as a NumPy array.

Use the expansion $\|x - y\|^2 = \|x\|^2 - 2\,x \cdot y + \|y\|^2$, which costs one matrix product instead of an `N × M × D` intermediate. Rounding can make a distance come out slightly negative; clamp the result at 0.

```
solve([[0, 0], [1, 1]], [[1, 0]])  ->  array([[1.], [1.]])
```

#### What interviewers look for

- Broadcasting `(N, 1) + (1, M)` for the norms, and `X @ Y.T` for the cross term.
- Knowing the memory cost of the direct `(X[:, None] - Y[None]) ** 2` version: `N × M × D` floats.
- The clamp, and why the expansion loses precision when points are far from the origin but close to each other.

#### Follow-ups

- How would you get the `k` nearest neighbours of each row of `X` from this matrix? (`np.argpartition`.)
- `N = M = 10^6`. What do you do? (Chunk over `X`, or an approximate index.)
- How does this relate to `torch.cdist`?
