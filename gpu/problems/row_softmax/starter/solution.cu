// GPU_GYM_STARTER
#include <cuda_runtime.h>

// X and Y are row-major rows x cols. Softmax each row independently.
__global__ void softmax_kernel(const float* X, float* Y, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
    }
}

extern "C" void solve(const float* X, float* Y, int rows, int cols) {
    int threads = 128;
    int blocks = (rows + threads - 1) / threads;
    softmax_kernel<<<blocks, threads>>>(X, Y, rows, cols);
    cudaDeviceSynchronize();
}
