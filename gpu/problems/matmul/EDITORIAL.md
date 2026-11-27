#### The idea
The naive kernel (one thread per output, a loop over `k`) reads each input element `N` or `M` times from global
memory. Tiling turns that into on-chip reuse: a block loads a `T × T` tile of `A` and of `B` into shared memory, every
thread reads them `T` times, and the block walks along `K` one tile pair at a time (the diagram in the statement).

#### A solution
```cuda
#define T 32

__global__ void matmul_tiled(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[T][T], Bs[T][T];
    int row = blockIdx.y * T + threadIdx.y, col = blockIdx.x * T + threadIdx.x;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[threadIdx.y][threadIdx.x] = (row < M && k0 + threadIdx.x < K) ? A[(size_t)row * K + k0 + threadIdx.x] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (k0 + threadIdx.y < K && col < N) ? B[(size_t)(k0 + threadIdx.y) * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < T; ++k) acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[(size_t)row * N + col] = acc;
}

extern "C" void solve(const float* A, const float* B, float* C, int M, int N, int K) {
    dim3 threads(T, T), blocks((N + T - 1) / T, (M + T - 1) / T);
    matmul_tiled<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```

#### What it costs
`2MNK` FLOPs over at least `4(MK + KN + MN)` bytes: for 4096 × 4096 FP32 that is an intensity above 600, far right of
the ridge point. A matmul *should* be compute-bound, and the ladder is about getting there:

| Step | What it fixes |
|---|---|
| naive, one output per thread | (baseline) every product re-reads global memory |
| shared-memory tiles (above) | global traffic drops by `T`; now limited by shared-memory loads, two per FMA |
| register tiles: each thread computes 4 × 4 or 8 × 8 outputs | one shared load feeds many FMAs |
| vectorized `float4` loads, transposed `A` tile | load instructions and bank conflicts |
| double buffering (load tile `k+1` while computing `k`) | waiting on memory between tiles |
| Tensor Cores (TF32/BF16, see `gemm_fp16`) | the FP32 FMA ceiling itself |

The tiled kernel above reaches roughly 15 to 25 percent of cuBLAS FP32; register tiling gets to 60 to 80.

#### Pitfalls
- Missing bounds on the tile loads: sizes that are not multiples of `T` read garbage, and the zero padding matters
  because padded products are summed.
- One `__syncthreads()` instead of two: the next iteration overwrites the tile while slow threads still read it.
- TF32 Tensor Cores (`torch.backends.cuda.matmul.allow_tf32`) are about 1e-3 off and fail the FP32 tolerance.
