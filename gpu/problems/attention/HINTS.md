# Hints for attention

Each `## Hint` section is revealed one at a time by `gpu-gym hint attention`.

## Hint 1

Get a correct version first. In the starter, one thread owns one query row `i`. Loop over `j` once to find the max score, again to get `sum` of `expf(s - max)`, and a third time to add `expf(s - max) / sum * V[j * d + k]` into `O[i * d + k]` for each `k`. Recompute the score `s` each time from `Q` and `K`. Zero the `O` row first: it starts as NaN.

## Hint 2

Merge the three loops into one with the online softmax. Keep `m` (running max), `l` (running sum), and `acc[k]` (unnormalized output, `d <= 128` floats in a local array). For each key `j`: `m_new = fmaxf(m, s); scale = expf(m - m_new); p = expf(s - m_new); l = l * scale + p; acc[k] = acc[k] * scale + p * V[j * d + k];`. At the end, `O[i * d + k] = acc[k] / l`. Start with `m = -INFINITY` and `l = 0`.

## Hint 3

Now tile it. A block takes 64 query rows. Load a tile of 64 key rows of `K` and `V` into shared memory, compute the 64 × 64 score tile with every thread doing a small piece, then apply the online update to whole tiles: one max and one rescale per query row per key tile, not per key. All the threads working on the same query row must agree on `m` and `l`, so reduce across them with warp shuffles.

## Hint 4

Give each thread a small block of scores (for example 4 query rows by 4 keys) and a small block of the output, like a register-tiled matmul. Store `K` transposed in shared memory so the score loop reads it without bank conflicts. Fold `1 / sqrt(d)` and `log2(e)` into one scale and use `exp2f`. Template on `d` (64 and 128) so the loops over `k` unroll.
