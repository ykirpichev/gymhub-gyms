# Hints for MoE grouped matmul

Each `## Hint` section is revealed one at a time by `gpu-gym hint moe_matmul`.

## Hint 1

Correct first: with one thread per `(t, n)`, read `int e = (int)expert[t];` and sum `X[t * K + k] * W[(e * K + k) * N + n]` over `k`. Note that consecutive threads take consecutive `n`, so the warp's loads from `W` are contiguous.

## Hint 2

Group the tokens. A first kernel counts tokens per expert with `atomicAdd(&count[e], 1)`. Compute `start[e]` as the prefix sum of `count` (with `E` at most a few dozen, one thread can do it). A second kernel gives each token a slot, `perm[start[e] + atomicAdd(&fill[e], 1)] = t`. The order inside a group does not matter.

## Hint 3

Now run your tiled matmul on each group. Build a small table of row tiles on the host or in a kernel: for each expert, `ceil(count[e] / BLOCK_M)` entries of `(e, first row in perm)`. Launch `grid = (number of row tiles, ceil(N / BLOCK_N))`. A block loads the `BLOCK_M` rows `X[perm[r]]` of its tile into shared memory, multiplies by tiles of `W[e]`, and writes `Y[perm[r]]`. Rows past the end of the group are masked.

## Hint 4

In Triton the grouping is three lines of PyTorch in `solve`: `ids = expert.long()`, `order = torch.argsort(ids)`, `counts = torch.bincount(ids, minlength=E)`. Build the tile table with `torch.repeat_interleave` and pass it to the kernel, which loads `rows = tl.load(order + tile_start + offs_m)` and uses them as row pointers into `X` and `Y`. Use `tl.dot(a, b, input_precision="ieee")`.
