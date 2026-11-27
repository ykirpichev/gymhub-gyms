# Tiled matmul traffic

A tiled matmul `C = A @ B` (`A` is `M × K`, `B` is `K × N`) gives each thread block one `BM × BN` tile of `C`. To compute it, the block reads a `BM × K` strip of `A` and a `K × BN` strip of `B` from global memory (in chunks along `K`, but every element of both strips once). Assume no caching between blocks.

Write `solve(M, N, K, BM, BN, elem_bytes)` that returns a tuple `(bytes_read, intensity)`:

- `bytes_read`: total bytes all blocks read from global memory, an int.
- `intensity`: $2MNK$ FLOPs divided by `bytes_read` plus the bytes of writing `C` once (`M * N * elem_bytes`), a float.

Raise `ValueError` unless `M` is divisible by `BM` and `N` by `BN`.

```
solve(4096, 4096, 4096, 128, 128, 2)  ->  (2147483648, 63.01...)
```

#### What interviewers look for

- The formula $\text{bytes} = MNK \left(\frac{1}{BN} + \frac{1}{BM}\right) \cdot \text{elem\_bytes}$, and reading it: bigger tiles, fewer bytes.
- Comparing the intensity with the ridge point: with 128 × 128 tiles and no L2 reuse it is 63, far below 333, so real kernels also rely on L2 hits between neighbouring blocks.
- Why the tile size cannot grow forever: shared memory, registers per thread, and enough blocks to fill every SM.

#### Follow-ups

- Make the tiles 256 × 128. What happens to the intensity, and what to occupancy?
- How does block scheduling order (for example grouping blocks so neighbours share strips of `A`) raise L2 hit rates?
- Where does the next level of reuse come from inside the block? (Each thread holding a small output tile in registers.)
