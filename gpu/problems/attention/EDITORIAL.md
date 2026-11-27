#### The idea
Two correct designs:
1. **Three kernels:** `S = QKᵀ/√d` with your matmul, a row softmax, then `O = PV`. Simple, but `S` is `N × M`
   floats written once and read twice.
2. **Fused (flash):** never store `S`. Walk over the keys keeping a running max `m`, running sum `l` and an output
   accumulator per query row; when the max grows, rescale `l` and the accumulator by `exp(m_old − m_new)`.

A compact fused kernel gives each query row to one warp; each lane owns `d / 32` output columns (`d <= 128`).

#### A solution
```cuda
__device__ float warp_allsum(float v) {          // every lane gets the total
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    return v;
}

__global__ void attn_kernel(const float* Q, const float* K, const float* V, float* O, int N, int M, int d) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32, lane = threadIdx.x % 32;
    if (row >= N) return;
    const float* q = Q + (size_t)row * d;
    float scale = rsqrtf((float)d), m = -INFINITY, l = 0.0f, acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < M; ++j) {
        const float* k = K + (size_t)j * d;
        float s = 0.0f;
        for (int c = lane; c < d; c += 32) s += q[c] * k[c];
        s = warp_allsum(s) * scale;
        float m_new = fmaxf(m, s), alpha = expf(m - m_new), p = expf(s - m_new);
        l = l * alpha + p;
        for (int t = 0, c = lane; c < d; ++t, c += 32) acc[t] = acc[t] * alpha + p * V[(size_t)j * d + c];
        m = m_new;
    }
    for (int t = 0, c = lane; c < d; ++t, c += 32) O[(size_t)row * d + c] = acc[t] / l;
}
// launch: 4 warps per block, (N + 3) / 4 blocks
```

#### What it costs
`4Nd·M` FLOPs. The three-kernel version moves the `N × M` matrix through HBM three times; the fused one moves only
`Q, K, V, O`. This warp-per-row kernel still re-reads `K` and `V` once per query row (from L2, mostly). Real
FlashAttention loads a *tile* of keys and values into shared memory once and lets a whole block of query rows use
it, with the two products done as small matmuls on Tensor Cores.

#### Pitfalls
- Rescaling `l` but not the accumulator (or the reverse) when the max changes.
- Starting `m` at 0 instead of `-INFINITY`: fine on random tests, but when every score is very negative,
  `exp(s - 0)` underflows to 0 and the row ends as `0 / 0`.
- TF32 in `tl.dot` or `torch.matmul`: about 1e-3 off; the tolerance is 1e-4. Use `input_precision="ieee"` in Triton.
