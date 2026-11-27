# Batched attention scores with einsum

Write `solve(q, k, v)` for tensors shaped `(B, T, H, D)`: batch, time, heads, head dimension (the layout projections produce before any transpose). Compute, per batch and head:

- scores `s[b, h, i, j] = q[b, i, h, :] · k[b, j, h, :] / sqrt(D)`,
- `p = softmax(s)` over `j`,
- `out[b, i, h, :] = Σ_j p[b, h, i, j] · v[b, j, h, :]`.

Return `out` in the input layout `(B, T, H, D)`, as a NumPy array. Use einsum; no Python loops over batch, time or heads.

```
solve([[[[1, 0]], [[0, 1]]]], [[[[1, 0]], [[0, 1]]]], [[[[1, 2]], [[3, 4]]]])
-> array([[[[1.6604, 2.6604]], [[2.3396, 3.3396]]]])
```

#### What interviewers look for

- Writing both einsums directly in the input layout (`"bihd,bjhd->bhij"`, `"bhij,bjhd->bihd"`) instead of transposing.
- A numerically stable softmax: subtract the max over the last axis.
- Being able to say the shapes of every intermediate.

#### Follow-ups

- Add a causal mask. Where does it go, and with what value?
- Grouped-query attention has `Hk` key and value heads, fewer than `H`. How does the einsum change?
- Which intermediate is the memory problem at long `T`, and what does flash attention do about it?
