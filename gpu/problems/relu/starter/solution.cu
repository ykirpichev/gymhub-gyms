// GPU_GYM_STARTER
#include <cuda_runtime.h>

__global__ void relu_kernel(const float* A, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
    }
}

extern "C" void solve(const float* A, float* C, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(A, C, N);
    cudaDeviceSynchronize();
}
