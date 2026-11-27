// GPU_GYM_STARTER
#include <cuda_runtime.h>

// X is T x K, W is E x K x N, Y is T x N, all row-major. expert holds whole numbers in [0, E)
// stored as floats. Y[t, n] = sum_k X[t, k] * W[expert[t], k, n].
__global__ void moe_matmul_kernel(const float* X, const float* expert, const float* W, float* Y,
                                  int T, int E, int K, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < T * N) {
    }
}

extern "C" void solve(const float* X, const float* expert, const float* W, float* Y, int T, int E,
                      int K, int N) {
    int threads = 256;
    int blocks = (T * N + threads - 1) / threads;
    moe_matmul_kernel<<<blocks, threads>>>(X, expert, W, Y, T, E, K, N);
    cudaDeviceSynchronize();
}
