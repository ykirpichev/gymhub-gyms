#### The idea
Each output `y[r]` is a dot product of one row with `x`. Give each row to a **warp**: the 32 lanes read 32
consecutive elements of the row at a time, which coalesces, then combine their partial sums with shuffles. One thread
per row would make each warp read 32 different rows at once, strided by `K`.

#### A solution
```cuda
__device__ float warp_sum(float v) {          // lane 0 ends up with the warp's total
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    return v;
}

__global__ void matvec_kernel(const float* A, const float* x, float* y, int M, int K) {
    int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (row >= M) return;                       // the whole warp leaves together
    const float* a = A + (size_t)row * K;
    float s = 0.0f;
    for (int k = lane; k < K; k += 32) s += a[k] * x[k];
    s = warp_sum(s);
    if (lane == 0) y[row] = s;
}

extern "C" void solve(const float* A, const float* x, float* y, int M, int K) {
    int threads = 256;                          // 8 rows per block
    matvec_kernel<<<(M + 7) / 8, threads>>>(A, x, y, M, K);
    cudaDeviceSynchronize();
}
```
PyTorch: `torch.mv(A, x, out=y)`.

#### What it costs
The matrix dominates: `4MK` bytes for `2MK` FLOPs, intensity 1/2. Memory-bound, so the target is `4MK / bandwidth`.
`x` is re-read by every row but stays in L1/L2. This is exactly the shape of batch-1 LLM decoding, which is why
decoding speed is set by how fast the weights stream from memory.

#### Pitfalls
- One thread per row: correct and several times slower, because the loads are strided.
- Very short rows (`K < 32`) leave lanes idle; then several rows per warp is better.
- `float4` loads help further, but need `K % 4 == 0` and 16-byte alignment; handle the tail separately.
