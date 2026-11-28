#### Idea
The loop nest is the kernel: `(i0, j0)` is a thread block, the `k0` loop is the walk along K with a shared-memory tile
of A and B per step, and the inner `(i, j)` loops are the threads. Loads are `2·t²` per step and there are
`(n/t)(m/t)(k/t)` steps, so `2·n·m·k / t` in total: the factor `t` is the saving.

#### In CUDA
The "load" becomes each thread copying one element of each tile into `__shared__` arrays, followed by
`__syncthreads()`; the accumulation reads only shared memory. A second `__syncthreads()` stops fast threads from
overwriting the tile while slow ones still read it.

#### Pitfalls
- Resetting the accumulator per K step instead of per output: the partial sums must add up across `k0`.
- Counting loads per thread instead of per tile.
