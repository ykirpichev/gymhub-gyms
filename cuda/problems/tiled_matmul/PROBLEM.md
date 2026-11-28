# Tiled matmul, tile by tile

Write a matrix multiply that is organized like a tiled CUDA kernel, and count its global loads. `a` is `n × k`, `b` is
`k × m` (lists of rows), and `t` is the tile size; `n`, `m` and `k` are multiples of `t`.

Compute C one `t × t` output tile at a time. For each output tile, walk along K in steps of `t`: "load" the `t × t` tile
of A and the `t × t` tile of B (count `2 * t * t` global loads per step), then accumulate the products from those tiles.

Return `[c, loads]`.

Example: `a = [[1, 2], [3, 4]]`, `b = [[5, 6], [7, 8]]`, `t = 1`: `c = [[19, 22], [43, 50]]`; 4 output tiles × 2 steps ×
2 loads = 16 loads. With `t = 2`: one tile, one step, 8 loads.
