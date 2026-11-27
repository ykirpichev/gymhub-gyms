#### The idea
One block per row. Three sweeps over the row, each a block-wide reduction or a map: the max, the sum of
`exp(x - max)`, then the normalized write. The row is re-read from L1/L2, so the cost is close to one read and one
write.

#### A solution
```cuda
__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

// Every thread gets the block's total. blockDim.x is a multiple of 32.
__device__ float block_sum(float v) {
    __shared__ float part[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_sum(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? part[threadIdx.x] : 0.0f;
    if (warp == 0) v = warp_sum(v);
    if (threadIdx.x == 0) part[0] = v;
    __syncthreads();
    float total = part[0];
    __syncthreads();                            // part[] is reused by the next call
    return total;
}

__device__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(0xffffffff, v, o));
    return v;
}

__device__ float block_max(float v) {           // same shape as block_sum
    __shared__ float part[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    v = warp_max(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    v = threadIdx.x < blockDim.x / 32 ? part[threadIdx.x] : -INFINITY;
    if (warp == 0) v = warp_max(v);
    if (threadIdx.x == 0) part[0] = v;
    __syncthreads();
    float m = part[0];
    __syncthreads();
    return m;
}

__global__ void softmax_kernel(const float* X, float* Y, int cols) {
    const float* x = X + (size_t)blockIdx.x * cols;
    float* y = Y + (size_t)blockIdx.x * cols;
    float m = -INFINITY;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, x[c]);
    m = block_max(m);
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += expf(x[c] - m);
    float inv = 1.0f / block_sum(s);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[c] = expf(x[c] - m) * inv;
}

extern "C" void solve(const float* X, float* Y, int rows, int cols) {
    softmax_kernel<<<rows, 256>>>(X, Y, cols);
    cudaDeviceSynchronize();
}
```

#### What it costs
8 bytes per element of compulsory traffic; memory-bound. The **online softmax** merges the first two sweeps: keep a
running max `m` and sum `l`, and when a new value raises the max, rescale `l` by `exp(m_old - m_new)`. Two
`(m, l)` pairs combine the same way, so it also works across threads. It is the trick flash attention is built on.

#### Pitfalls
- Skipping the max subtraction: `exp` overflows to `inf` above about 88 and the row becomes NaN.
- Dividing inside the loop instead of multiplying by one reciprocal: slower, same answer.
- Very short rows (say 32 columns) waste a 256-thread block; one warp per row is better there.
