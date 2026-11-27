#### The idea
Rows have different lengths. One thread per row (*scalar CSR*) is simple but uncoalesced and unbalanced. One warp
per row (*vector CSR*) reads each row's nonzeros contiguously and reduces with shuffles; it is the better default
unless rows are very short.

#### A solution
```cuda
__device__ float warp_sum(float v) {          // lane 0 ends up with the warp's total
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__global__ void spmv_warp(const float* row_ptr, const float* col_idx, const float* values,
                          const float* x, float* y, int rows) {
    int r = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (r >= rows) return;
    int begin = (int)row_ptr[r], end = (int)row_ptr[r + 1];
    float s = 0.0f;
    for (int p = begin + lane; p < end; p += 32) s += values[p] * x[(int)col_idx[p]];
    s = warp_sum(s);
    if (lane == 0) y[r] = s;                 // an empty row writes 0
}
```
Launch 8 warps per block and `ceil(rows / 8)` blocks. PyTorch: build
`torch.sparse_csr_tensor(row_ptr.long(), col_idx.long(), values, (rows, cols))` and multiply by `x`.

#### What it costs
About 12 bytes per nonzero (value, column index, a gathered `x`) for 2 FLOPs: firmly memory-bound. The gather of `x`
is irregular; it is fast when the column indices of nearby rows overlap (L1/L2 hits) and slow when they are random.

#### Pitfalls
- Empty rows must still write `y[r] = 0`; `y` is not zeroed for you.
- A few very long rows (power-law graphs) leave one warp working while others idle. Fixes: split long rows across
  several warps, or a merge-path scheme that splits the nonzeros evenly.
- Duplicate column indices in a row are legal: they just add.
