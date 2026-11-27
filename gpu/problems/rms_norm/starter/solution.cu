// GPU_GYM_STARTER
#include <cuda_runtime.h>

// X and Y are row-major rows x cols. Y[r, c] = X[r, c] / sqrt(mean of row r squared + eps) * weight[c].
__global__ void rms_norm_kernel(const float* X, const float* weight, float* Y, int rows, int cols,
                                float eps) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
    }
}

extern "C" void solve(const float* X, const float* weight, float* Y, int rows, int cols, float eps) {
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    rms_norm_kernel<<<blocks, threads>>>(X, weight, Y, rows, cols, eps);
    cudaDeviceSynchronize();
}
