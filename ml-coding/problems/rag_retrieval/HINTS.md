## Hint 1
Normalize the rows of both matrices once; then one matrix product `qn @ dn.T` gives the whole `Q × N` score table. For a zero row, divide by 1 instead of 0 (it stays a zero row, so its scores are 0).

## Hint 2
`np.argsort(-scores, axis=1, kind="stable")` sorts each row by score descending and keeps equal scores in index order. `np.argsort(scores)[::-1]` does not: it puts the higher index first among ties.

## Hint 3
Sorting all `N` is `O(N log N)` per query. For `k ≪ N`, `np.argpartition` finds the top `k` in `O(N)`, but it does not keep ties at the cut-off in index order, so you must handle the boundary. Then use `np.take_along_axis` to gather the scores.
