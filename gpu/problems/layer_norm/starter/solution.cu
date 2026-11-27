// GPU_GYM_STARTER
#include <cuda_runtime.h>

// X and Y are row-major rows x cols. Normalize each row, then scale by gamma and shift by beta.
__global__ void layer_norm_kernel(const float* X, const float* gamma, const float* beta, float* Y,
                                  int rows, int cols, float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
    }
}

extern "C" void solve(const float* X, const float* gamma, const float* beta, float* Y, int rows,
                      int cols, float eps) {
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    layer_norm_kernel<<<blocks, threads>>>(X, gamma, beta, Y, rows, cols, eps);
    cudaDeviceSynchronize();
}
