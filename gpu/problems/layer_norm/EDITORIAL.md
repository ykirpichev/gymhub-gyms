#### The idea
One block per row, like `row_softmax`. Two reductions: the mean, then the variance around it. Then one map that
normalizes, scales by `gamma` and shifts by `beta`.

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

__global__ void layer_norm_kernel(const float* X, const float* gamma, const float* beta, float* Y,
                                  int cols, float eps) {
    const float* x = X + (size_t)blockIdx.x * cols;
    float* y = Y + (size_t)blockIdx.x * cols;
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += x[c];
    float mean = block_sum(s) / cols;
    float q = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) { float d = x[c] - mean; q += d * d; }
    float rstd = rsqrtf(block_sum(q) / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        y[c] = (x[c] - mean) * rstd * gamma[c] + beta[c];
}
```
Launch `rows` blocks of 256 threads. PyTorch: `F.layer_norm(X, (cols,), gamma, beta, eps)` copied into `Y`.

#### What it costs
Memory-bound: 8 bytes per element plus `gamma` and `beta`, which every row shares and which stay in cache.

The two-pass variance is the safe one. The one-pass formula `E[x²] − E[x]²` subtracts two large, nearly equal
numbers when the mean is big compared with the spread, and loses most of its digits. If you want one pass, use
**Welford's** running mean and M2, which also merge across threads.

#### Pitfalls
- Dividing by `cols - 1`: layer norm uses the population variance.
- Putting `eps` outside the square root changes the result for small variances.
- Forgetting that `gamma` and `beta` index by column, not by row.
