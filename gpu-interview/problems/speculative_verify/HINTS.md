## Hint 1

Write a helper `sample(d, v)` with a running cumulative sum. Use it for both the residual and the bonus token.

## Hint 2

If `p[i][x] ≥ q[i][x]` the token is always accepted, since `u[i] < 1`.
