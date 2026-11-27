#### The idea
Layer norm without the mean: one reduction (the mean of squares), then a scale. One block per row.

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

__global__ void rms_norm_kernel(const float* X, const float* weight, float* Y, int cols, float eps) {
    const float* x = X + (size_t)blockIdx.x * cols;
    float* y = Y + (size_t)blockIdx.x * cols;
    float q = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) q += x[c] * x[c];
    float r = rsqrtf(block_sum(q) / cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[c] = x[c] * r * weight[c];
}
```
Launch `rows` blocks of 256 threads. PyTorch: `X * torch.rsqrt(X.pow(2).mean(-1, keepdim=True) + eps) * weight`.

#### What it costs
8 bytes per element; memory-bound. In a real model RMSNorm is fused with the residual add before it (read `x` and
the residual once, write both the sum and the normalized output), because two separate kernels would move the
tensor twice.

#### Pitfalls
- Using the mean of `x` instead of the mean of `x²`.
- Low-precision inputs: in BF16 models the sum of squares is still accumulated in FP32.
