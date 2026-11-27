# Hints for top-k per row

Each `## Hint` section is revealed one at a time by `gpu-gym hint top_k`.

## Hint 1

One thread per row first. Keep a local array `float best[64]` filled with `-INFINITY`, sorted from largest to smallest. For each `x` in the row, if `x > best[k - 1]`, shift the smaller entries down by one and insert `x` at its place. At the end copy `best[0 .. k)` to `Y`.

## Hint 2

Give each row a block of 128 threads. Each thread runs the Hint 1 insertion over columns `threadIdx.x, threadIdx.x + 128, ...` and keeps its own top `k`. Write all 128 lists to shared memory (`128 * 64 * 4` bytes is 32 KiB), then let thread 0 merge them. Correct, and the scan is now parallel.

## Hint 3

Merge in a tree instead: in round `s`, thread `t` with `t % (2 * s) == 0` merges its list with the list of thread `t + s`, keeping the top `k` of the two. Merging two sorted lists of `k` takes `k` steps with two cursors. After `log2(128) = 7` rounds thread 0 holds the answer.

## Hint 4

The per-element insertion is the slow part when `k` is 50: most values are smaller than the current `k`-th best and are rejected by one comparison, but each accepted one costs up to `k` moves. Cut the candidates first: find a threshold below which no value can be in the answer, for example the `k`-th largest of a sample, or by radix select on the top 8 bits, and insert only values above it. In Triton, `tl.sort` on a block-sized chunk plus a running merge of the top `k` is a good start.
