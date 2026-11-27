// GPU_GYM_STARTER
#include <cuda_runtime.h>

// A is row-major M x N. C is row-major N x M.
__global__ void transpose_kernel(const float* A, float* C, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < M && j < N) {
    }
}

extern "C" void solve(const float* A, float* C, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    transpose_kernel<<<blocks, threads>>>(A, C, M, N);
    cudaDeviceSynchronize();
}
