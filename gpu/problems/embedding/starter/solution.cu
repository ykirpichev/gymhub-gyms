// GPU_GYM_STARTER
#include <cuda_runtime.h>

// ids holds N whole numbers in [0, V) stored as floats. table is V x D, out is N x D, row-major.
// out[i, d] = table[ids[i], d].
__global__ void embedding_kernel(const float* ids, const float* table, float* out, int N, int V,
                                 int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * D) {
    }
}

extern "C" void solve(const float* ids, const float* table, float* out, int N, int V, int D) {
    int threads = 256;
    int blocks = (N * D + threads - 1) / threads;
    embedding_kernel<<<blocks, threads>>>(ids, table, out, N, V, D);
    cudaDeviceSynchronize();
}
