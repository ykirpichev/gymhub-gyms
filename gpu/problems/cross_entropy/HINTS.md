# Hints for cross-entropy

Each `## Hint` section is revealed one at a time by `gpu-gym hint cross_entropy`.

## Hint 1

Start with one thread per row. Loop once to find `m`, the row's maximum. Loop again to sum `expf(x - m)`. Then `loss[row] = m + logf(sum) - logits[row * V + (int)targets[row]];`.

## Hint 2

One thread cannot stream a 32,768-long row fast enough. Give each row a block: `grid = rows`, `block = 256` or more, and let each thread walk `j = threadIdx.x, threadIdx.x + blockDim.x, ...`. Reduce the maximum across the block, then reduce the sum, as in `row_softmax`.

## Hint 3

Merge both passes: each thread keeps its own `(m, s)` while it walks the row, updating it with `if (x > m) { s = s * expf(m - x); m = x; } s += expf(x - m);`. Then combine the per-thread states with the rule from the statement, first inside each warp with `__shfl_xor_sync`, then across warps through shared memory. Start `m` at `-INFINITY` and guard the `-inf - -inf` case when a thread has no elements.

## Hint 4

Load `float4`s when `V % 4 == 0`, and use `__expf` once the result is correct. In Triton, loop over the row in chunks of `BLOCK_SIZE` with a vector of running maxima and sums, `m_new = tl.maximum(m, x)`, `s = s * tl.exp(m - m_new) + tl.exp(x - m_new)`, then reduce the vectors at the end. Pad masked lanes with `-float("inf")`.
