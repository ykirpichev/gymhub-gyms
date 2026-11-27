#### The idea
For each row, the loss is `logsumexp(row) − row[target]`. One block per row computes the max, then the sum of
`exp(x − max)`, and thread 0 writes `max + log(sum) − x[target]`. The full softmax is never written.

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

__global__ void ce_kernel(const float* logits, const float* targets, float* loss, int V) {
    const float* x = logits + (size_t)blockIdx.x * V;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < V; j += blockDim.x) m = fmaxf(m, x[j]);
    m = block_max(m);
    float s = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) s += expf(x[j] - m);
    s = block_sum(s);
    if (threadIdx.x == 0) loss[blockIdx.x] = m + logf(s) - x[(int)targets[blockIdx.x]];
}
```
Launch `rows` blocks of 256 (or 512 for a 50,000-word vocabulary). PyTorch:
`loss.copy_(F.cross_entropy(logits, targets.long(), reduction="none"))`.

#### What it costs
One read of the logits, `4 · rows · V` bytes, which is large: for a 50,000-token vocabulary and 8,192 tokens that is
1.6 GB in FP32. Training frameworks fuse this kernel with the backward pass (the gradient is
`softmax − onehot`) and some never materialize the logits at all, computing the loss chunk by chunk from the
hidden states.

#### Pitfalls
- `log(softmax(x))[target]` computed literally: the softmax underflows to 0 for very negative logits and the log is
  `-inf`. Work in log space as above.
- Skipping the max: rows with logits near 1,000 overflow.
- Targets arrive as floats; convert with `(int)`.
