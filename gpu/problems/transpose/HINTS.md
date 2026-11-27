# Hints for matrix transpose

Each `## Hint` section is revealed one at a time by `gpu-gym hint transpose`.

## Hint 1

In the starter, `i` is the row of `A` and `j` is its column. Inside `if (i < M && j < N)`, write `C[j * M + i] = A[i * N + j];`. That passes the tests.

Triton: build 2D offsets from `offs_m[:, None]` and `offs_n[None, :]`, load a `BLOCK_M × BLOCK_N` tile of `A` with a mask, and store it to `C` using `offs_n[None, :] * M + offs_m[:, None]` as the address. Triton reorders the tile through shared memory for you.

## Hint 2

The naive kernel reads coalesced, because `threadIdx.x` moves along a row of `A`. Its writes are strided by `M`. Swapping `i` and `j` only moves the problem to the reads.

Use a tile: `__shared__ float tile[32][32];` with a `dim3(32, 32)` block. Each thread reads one element of `A` into `tile[threadIdx.y][threadIdx.x]`, then `__syncthreads()`. For the write, swap the block's coordinates (`blockIdx.x` and `blockIdx.y` trade roles) and read the tile transposed, `tile[threadIdx.x][threadIdx.y]`, so that `threadIdx.x` again moves along a row of `C`. Keep the bounds checks on both the read and the write: the `63x65` test has partial tiles. Update the grid in `solve` to match.

## Hint 3

Reading `tile[threadIdx.x][threadIdx.y]` makes the 32 threads of a warp read one column of the tile. Shared memory has 32 banks, and addresses 32 floats apart hit the same bank, so this is a 32-way bank conflict. Pad the tile by one column, `__shared__ float tile[32][33];`. The column now spans all 32 banks and the conflict disappears.

## Hint 4

Use a `dim3(32, 8)` block that still moves a 32 × 32 tile: each thread loops over four rows, `for (int r = 0; r < 32; r += 8)`, on both the load and the store. Each thread does more work, there are fewer threads per tile, and the loads of one thread are independent, so more of them are in flight at once.
