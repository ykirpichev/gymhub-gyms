# Hints for causal flash attention

Each `## Hint` section is revealed one at a time by `gpu-gym hint causal_attention`.

## Hint 1

A correct first version: one thread per `(h, i)` row. Loop `j` from `0` to `i` (inclusive) twice, once for the max score and once for the sum of `exp(s - max)` while accumulating `exp(s - max) * V[j]` into `d` registers or a local array of 128 floats. Divide by the sum at the end. Pointers for head `h` are `Q + h * L * d` and so on.

## Hint 2

Turn the two loops into one with the online softmax: keep `m`, `l` and `acc[d]`. For each `j`, `m_new = max(m, s)`, `alpha = expf(m - m_new)`, `p = expf(s - m_new)`, then `l = l * alpha + p` and `acc = acc * alpha + p * V[j]`.

## Hint 3

Tile it. Use a 2D grid: `blockIdx.x` is the query tile, `blockIdx.y` is the head. Load the tile of `BLOCK_N` query rows into shared memory once, then for each key tile `kt = 0 .. qt` (inclusive, using the same tile size for both) load `K` and `V` tiles into shared memory and update every query row's state. Only the diagonal tile `kt == qt` needs the `j > i` mask.

## Hint 4

In Triton the loop is `for start in range(0, (qt + 1) * BLOCK_N, BLOCK_M)`, with `BLOCK_N` a multiple of `BLOCK_M`. Compute `s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale`, then `s = tl.where(offs_n[:, None] >= start + offs_m[None, :], s, float("-inf"))`, and mask key columns at or past `L` the same way. Launch the longest query tiles first by reversing the program id: `qt = tl.num_programs(0) - 1 - tl.program_id(0)`.
