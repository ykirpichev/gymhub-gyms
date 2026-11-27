// GPU_GYM_STARTER
#include <cuda_runtime.h>
#include <math.h>

// X is rows x cols, Y is rows x k, both row-major. k <= 64.
// Y[r, :] = the k largest values of X[r, :], largest first.
__global__ void top_k_kernel(const float* X, float* Y, int rows, int cols, int k) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
    }
}

extern "C" void solve(const float* X, float* Y, int rows, int cols, int k) {
    int threads = 64;
    int blocks = (rows + threads - 1) / threads;
    top_k_kernel<<<blocks, threads>>>(X, Y, rows, cols, k);
    cudaDeviceSynchronize();
}
