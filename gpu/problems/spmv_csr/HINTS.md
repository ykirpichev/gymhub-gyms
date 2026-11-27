# Hints for sparse matvec

Each `## Hint` section is revealed one at a time by `gpu-gym hint spmv_csr`.

## Hint 1

One thread per row: read `int start = (int)row_ptr[r], end = (int)row_ptr[r + 1];`, then sum `values[p] * x[(int)col_idx[p]]` for `p` in `[start, end)` and write `y[r]`. Every row writes `y[r]`, including empty rows, which write 0.

## Hint 2

One warp per row: `r = (blockIdx.x * blockDim.x + threadIdx.x) / 32`, `lane = threadIdx.x % 32`. Lane `l` sums positions `start + l, start + l + 32, ...`. Reduce the 32 partial sums with `__shfl_down_sync(0xffffffff, v, offset)` for offsets 16 down to 1, and let lane 0 write `y[r]`. Launch 32 times as many threads as rows.

## Hint 3

The warp-per-row kernel wastes lanes on short rows. Pick the group size from the average row length `nnz / rows`: 2, 4, 8, 16 or 32 lanes per row, using `__shfl_down_sync` with a width argument. Rows of a handful of entries get a few lanes each and a warp serves many rows at once.

## Hint 4

For the very long rows, no fixed group size is right. Split the nonzeros evenly instead: give each block the same number of nonzeros, find the first row of its range with a binary search over `row_ptr`, and add a row's partial sums across blocks with `atomicAdd(&y[r], v)` (zero `y` first). This is the "merge-based" or "nonzero-split" SpMV. In Triton, a program per row with a loop over `BLOCK_SIZE` chunks of the row is a good first version.
