#### The idea
FP16 inputs and an FP32 accumulator are exactly what Tensor Cores do. The WMMA API exposes them at warp level: a warp
loads a 16 × 16 tile of `A` and of `B` into *fragments*, `mma_sync` multiplies and accumulates in FP32, and the warp
walks along `K` 16 at a time. Every test size is a multiple of 16, so there are no ragged edges.

#### A solution
```cuda
#include <mma.h>
using namespace nvcuda;

// One warp per 16 x 16 tile of C. Block: 128 x 4 threads = 4 x 4 warps = a 64 x 64 tile of C.
__global__ void gemm_wmma(const __half* A, const __half* B, float* C, int M, int N, int K) {
    int tile_m = blockIdx.y * blockDim.y + threadIdx.y;
    int tile_n = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (tile_m * 16 >= M || tile_n * 16 >= N) return;          // uniform across the warp
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
    wmma::fill_fragment(c, 0.0f);
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a, A + (size_t)tile_m * 16 * K + k, K);
        wmma::load_matrix_sync(b, B + (size_t)k * N + tile_n * 16, N);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(C + (size_t)tile_m * 16 * N + tile_n * 16, c, N, wmma::mem_row_major);
}

extern "C" void solve(const __half* A, const __half* B, float* C, int M, int N, int K) {
    dim3 threads(128, 4);
    dim3 blocks((N / 16 + 3) / 4, (M / 16 + 3) / 4);
    gemm_wmma<<<blocks, threads>>>(A, B, C, M, N, K);
    cudaDeviceSynchronize();
}
```
PyTorch: `C.copy_(A.float() @ B.float())` matches the reference. `A @ B` runs on Tensor Cores and accumulates in
FP32, but rounds its output to FP16, which can miss the tolerance for large `K`. Triton: `tl.dot` on FP16 blocks
returns FP32.

#### What it costs
`2MNK` FLOPs at Tensor Core rate, roughly 8 to 16 times the FP32 FMA rate. This kernel loads its fragments straight
from global memory, so neighbouring warps re-read the same tiles from L2; staging `A` and `B` tiles in shared memory
for the whole block (and double-buffering them with `cp.async`) is the next step, and gets within reach of cuBLAS.

#### Pitfalls
- Accumulating in FP16 (`half` accumulator fragment): fast, and wrong by percents for large `K`.
- Fragment layout and leading dimensions: `A` is `M × K` row-major, so its leading dimension is `K`; `B`'s is `N`.
- WMMA needs compute capability 7.0 or newer; compile with the right `-arch`.
