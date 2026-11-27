#### The idea
Each output is a dot product of `K` weights with a window of `A`. Neighbouring outputs share `K - 1` of their inputs,
so a block loads its inputs once into shared memory, with a halo of `K - 1` extra values, and every thread reads its
window from there. The weights are the same for everyone: load them once per block too.

#### A solution
```cuda
__global__ void conv1d_kernel(const float* A, const float* W, float* C, int N, int K) {
    extern __shared__ float s[];
    float* a = s;                                   // blockDim.x + K - 1 inputs
    float* w = s + blockDim.x + K - 1;              // K weights
    int base = blockIdx.x * blockDim.x;
    for (int t = threadIdx.x; t < blockDim.x + K - 1; t += blockDim.x)
        a[t] = base + t < N ? A[base + t] : 0.0f;
    for (int t = threadIdx.x; t < K; t += blockDim.x) w[t] = W[t];
    __syncthreads();
    int i = base + threadIdx.x;
    if (i < N - K + 1) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += a[threadIdx.x + k] * w[k];
        C[i] = acc;
    }
}

extern "C" void solve(const float* A, const float* W, float* C, int N, int K) {
    int threads = 256, out = N - K + 1;
    size_t shared = (threads + 2 * K - 1) * sizeof(float);
    conv1d_kernel<<<(out + threads - 1) / threads, threads, shared>>>(A, W, C, N, K);
    cudaDeviceSynchronize();
}
```

#### What it costs
`2K` FLOPs per output against 8 bytes of compulsory traffic (one read, one write). With the benchmark's `K = 7`
that is under 2 FLOPs per byte: memory-bound, so the win from shared memory is mostly fewer *requests*; L1 would
catch many of the re-reads anyway. For large `K` the kernel becomes compute-bound and the shared tile matters more.

#### Pitfalls
- The halo: the last threads of a block need `K - 1` inputs past the block's own range.
- `__syncthreads()` inside `if (i < N - K + 1)`: threads of the last block skip it and the block hangs or races.
  Load and sync first, then branch.
- Flipping `W`. This is cross-correlation, as in deep learning libraries.
